import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamChat } from './api'
import { uid } from './format'
import type { ChatMessage, Grounding, MealChange, MemoryFact, Span } from './types'

const SESSION_KEY = 'calorai.session'

function sessionFromStorage(): string {
  const fromUrl = new URLSearchParams(location.search).get('session')
  if (fromUrl && /^[\w-]{1,64}$/.test(fromUrl)) {
    localStorage.setItem(SESSION_KEY, fromUrl)
    return fromUrl
  }
  const existing = localStorage.getItem(SESSION_KEY)
  if (existing) return existing
  const fresh = uid() + uid()
  localStorage.setItem(SESSION_KEY, fresh)
  return fresh
}

const TOOL_LABEL: Record<string, string> = {
  lookup_nutrition: 'Looking up nutrition',
  log_meal: 'Logging the meal',
  update_meal: 'Correcting the meal',
  delete_meal: 'Removing the meal',
  get_meals: 'Checking your meals',
  search_meal_history: 'Searching your history',
  get_daily_totals: 'Adding up today',
  get_user_memory: 'Recalling your preferences',
  set_user_memory: 'Remembering that',
  delete_user_memory: 'Forgetting that',
}

export function useChat(onTurnComplete: () => void) {
  const [sessionId, setSessionId] = useState(sessionFromStorage)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  // Restore the conversation for this session (receipts + traces come back from stored metadata).
  useEffect(() => {
    let alive = true
    api
      .history(sessionId)
      .then(({ messages: rows }) => {
        if (!alive) return
        setMessages(
          rows.map((r) => {
            const meta = (r.meta ?? {}) as Record<string, unknown>
            return {
              id: uid(),
              role: r.role === 'user' ? 'user' : 'assistant',
              text: r.content,
              meals: meta.meals_changed as MealChange[] | undefined,
              groundings: meta.groundings as Grounding[] | undefined,
              memories: meta.memories_used as MemoryFact[] | undefined,
              trace: meta.trace as Span[] | undefined,
              latency_ms: meta.latency_ms as number | undefined,
              mode: meta.mode as string | undefined,
            }
          }),
        )
      })
      .catch(() => alive && setMessages([]))
    return () => {
      alive = false
    }
  }, [sessionId])

  const patch = (id: string, fn: (m: ChatMessage) => ChatMessage) =>
    setMessages((all) => all.map((m) => (m.id === id ? fn(m) : m)))

  const send = useCallback(
    async (text: string, image?: File | null) => {
      if (busy || (!text.trim() && !image)) return
      const userMsg: ChatMessage = {
        id: uid(),
        role: 'user',
        text: text.trim(),
        image: image ? URL.createObjectURL(image) : undefined,
      }
      const botId = uid()
      setMessages((all) => [
        ...all,
        userMsg,
        { id: botId, role: 'assistant', text: '', streaming: true, step: 'Reading your message', meals: [] },
      ])
      setBusy(true)
      abort.current = new AbortController()
      try {
        await streamChat(
          { message: text.trim(), sessionId, image },
          (event, data) => {
            switch (event) {
              case 'step':
                patch(botId, (m) => ({ ...m, step: data.label }))
                break
              case 'retrieval':
                patch(botId, (m) => ({ ...m, groundings: data.groundings, memories: data.memories }))
                break
              case 'tool_call':
                patch(botId, (m) => ({ ...m, step: TOOL_LABEL[data.name] ?? `Running ${data.name}` }))
                break
              case 'meal':
                patch(botId, (m) => ({ ...m, meals: [...(m.meals ?? []), data as MealChange] }))
                break
              case 'token':
                patch(botId, (m) => ({ ...m, text: m.text + data.text }))
                break
              case 'error':
                patch(botId, (m) => ({ ...m, error: data.message }))
                break
              case 'done':
                patch(botId, (m) => ({
                  ...m,
                  text: data.response,
                  streaming: false,
                  step: undefined,
                  trace: data.trace,
                  latency_ms: data.latency_ms,
                  mode: data.mode,
                  groundings: data.groundings ?? m.groundings,
                  memories: data.memories_used ?? m.memories,
                  meals: data.meals_changed ?? m.meals,
                }))
                break
            }
          },
          abort.current.signal,
        )
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        patch(botId, (m) => ({
          ...m,
          streaming: false,
          step: undefined,
          text: m.text || "I couldn't reach the CalorAI server. Is the backend running on port 8000?",
          error: message,
        }))
      } finally {
        setBusy(false)
        onTurnComplete()
      }
    },
    [busy, sessionId, onTurnComplete],
  )

  const newSession = useCallback(() => {
    abort.current?.abort()
    const fresh = uid() + uid()
    localStorage.setItem(SESSION_KEY, fresh)
    setSessionId(fresh)
    setMessages([])
  }, [])

  return { messages, send, busy, newSession, sessionId }
}
