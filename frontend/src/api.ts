import type { AnalyzeResponse, Dashboard, Health, RagStats, SearchResponse } from './types'

const USER = 'default'

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

const qs = (params: Record<string, string | number | undefined | null>) =>
  new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => [k, String(v)]),
  ).toString()

export const api = {
  health: () => json<Health>('/health'),
  metrics: () => json<Record<string, Record<string, { count: number; p50_ms: number; p95_ms: number }>>>('/metrics'),
  dashboard: (date?: string) => json<Dashboard>(`/api/dashboard?${qs({ user_id: USER, date })}`),
  deleteMeal: (id: number) => json(`/api/meals/${id}?${qs({ user_id: USER })}`, { method: 'DELETE' }),
  forget: (key: string) => json(`/api/memory/${encodeURIComponent(key)}?${qs({ user_id: USER })}`, { method: 'DELETE' }),
  history: (sessionId: string) =>
    json<{ messages: { role: string; content: string; meta: Record<string, unknown> | null }[] }>(
      `/api/history?${qs({ session_id: sessionId, user_id: USER })}`,
    ),
  search: (q: string, diet?: string, allergens?: string[]) =>
    json<SearchResponse>(`/api/rag/search?${qs({ q, k: 8, diet, allergens: allergens?.join(',') })}`),
  analyze: (text: string, diet?: string, allergens?: string[]) =>
    json<AnalyzeResponse>(`/api/rag/analyze?${qs({ text, diet, allergens: allergens?.join(',') })}`),
  ragStats: () => json<RagStats>('/api/rag/stats'),
}

export type StreamHandler = (event: string, data: any) => void

/**
 * POST a chat turn and consume the Server-Sent Events stream.
 * (EventSource only supports GET, so the stream is parsed by hand.)
 */
export async function streamChat(
  input: { message: string; sessionId: string; image?: File | null },
  onEvent: StreamHandler,
  signal?: AbortSignal,
): Promise<void> {
  const form = new FormData()
  form.append('message', input.message)
  form.append('session_id', input.sessionId)
  form.append('user_id', USER)
  if (input.image) form.append('image', input.image)

  const res = await fetch('/api/chat/stream', { method: 'POST', body: form, signal })
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      let event = 'message'
      const data: string[] = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (data.length) onEvent(event, JSON.parse(data.join('\n')))
    }
  }
}
