import { ArrowUp, ImagePlus, X } from 'lucide-react'
import { useEffect, useRef, useState, type DragEvent, type KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import type { ChatMessage } from '../types'
import { Receipt } from './Receipt'
import { Trace } from './Trace'

const EXAMPLES = [
  'had 2 parathas and chai for breakfast',
  'actually that was 3 parathas not 2',
  "i'm vegetarian btw",
  'leftover biryani, maybe two thirds of the box',
  'how am I doing today?',
]

function Assistant({ m }: { m: ChatMessage }) {
  const waiting = m.streaming && !m.text
  return (
    <div className="msg msg-assistant">
      {waiting ? (
        <p className="step" role="status">
          <span className="pulse" aria-hidden />
          {m.step ?? 'Thinking'}…
        </p>
      ) : (
        <div className="prose">
          <ReactMarkdown>{m.text}</ReactMarkdown>
          {m.streaming && <span className="caret" aria-hidden />}
        </div>
      )}
      {(m.meals ?? []).map((c, i) => (
        <Receipt key={`${c.meal.id}-${c.action}-${i}`} change={c} />
      ))}
      {m.error && <p className="msg-error">{m.error}</p>}
      {!m.streaming && <Trace message={m} />}
    </div>
  )
}

function Composer({ busy, onSend }: { busy: boolean; onSend: (text: string, image?: File | null) => void }) {
  const [text, setText] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const area = useRef<HTMLTextAreaElement>(null)
  const file = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!image) return setPreview(null)
    const url = URL.createObjectURL(image)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [image])

  useEffect(() => {
    const el = area.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  const submit = () => {
    if (busy || (!text.trim() && !image)) return
    onSend(text, image)
    setText('')
    setImage(null)
    if (file.current) file.current.value = ''
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      {preview && (
        <div className="composer-photo">
          <img src={preview} alt="Photo to send" />
          <button type="button" className="icon-btn icon-btn-sm" onClick={() => setImage(null)} aria-label="Remove photo">
            <X size={14} />
          </button>
        </div>
      )}
      <div className="composer-row">
        <input
          ref={file}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => setImage(e.target.files?.[0] ?? null)}
        />
        <button type="button" className="icon-btn" onClick={() => file.current?.click()} aria-label="Attach a photo of your meal">
          <ImagePlus size={18} />
        </button>
        <textarea
          ref={area}
          rows={1}
          value={text}
          placeholder={image ? 'Add a caption — e.g. "half was my brother\'s"' : 'What did you eat?'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={(e) => {
            const img = [...e.clipboardData.files].find((f) => f.type.startsWith('image/'))
            if (img) setImage(img)
          }}
          aria-label="Message"
        />
        <button type="submit" className="send-btn" disabled={busy || (!text.trim() && !image)} aria-label="Send">
          <ArrowUp size={18} strokeWidth={2.4} />
        </button>
      </div>
      <p className="composer-hint">Enter to send · Shift+Enter for a new line · paste or drop a photo</p>
      <DropTarget onFile={setImage} />
    </form>
  )
}

function DropTarget({ onFile }: { onFile: (f: File) => void }) {
  const [active, setActive] = useState(false)
  useEffect(() => {
    const over = (e: globalThis.DragEvent) => {
      if (e.dataTransfer?.types.includes('Files')) {
        e.preventDefault()
        setActive(true)
      }
    }
    window.addEventListener('dragover', over)
    return () => window.removeEventListener('dragover', over)
  }, [])
  if (!active) return null
  const drop = (e: DragEvent) => {
    e.preventDefault()
    setActive(false)
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'))
    if (f) onFile(f)
  }
  return (
    <div className="dropzone" onDragOver={(e) => e.preventDefault()} onDragLeave={() => setActive(false)} onDrop={drop}>
      Drop the photo of your plate
    </div>
  )
}

export function Chat({
  messages,
  busy,
  onSend,
  mode,
}: {
  messages: ChatMessage[]
  busy: boolean
  onSend: (text: string, image?: File | null) => void
  mode?: string
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const last = messages[messages.length - 1]

  useEffect(() => {
    const el = scroller.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: messages.length > 2 ? 'smooth' : 'auto' })
  }, [messages.length, last?.text, last?.meals?.length, last?.streaming])

  return (
    <section className="chat" aria-label="Conversation">
      <div className="chat-scroll" ref={scroller}>
        <div className="chat-inner">
          {messages.length === 0 ? (
            <div className="empty">
              <h1>What did you eat?</h1>
              <p>
                Text it the way you'd text a friend, or send a photo. Every number is looked up in a nutrition knowledge
                base — never guessed — and every line shows where it came from.
              </p>
              {mode === 'offline' && (
                <p className="notice">
                  Running on the <b>offline engine</b>: deterministic parsing plus the same RAG grounding, no LLM. Set{' '}
                  <code>GOOGLE_API_KEY</code> to switch to the full Gemini agent (photos, free-form reasoning, memory
                  extraction).
                </p>
              )}
              <ul className="examples">
                {EXAMPLES.map((ex) => (
                  <li key={ex}>
                    <button onClick={() => onSend(ex)} disabled={busy}>
                      {ex}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            messages.map((m) =>
              m.role === 'user' ? (
                <div key={m.id} className="msg msg-user">
                  {m.image && <img src={m.image} alt="Your meal" />}
                  {m.text && <p>{m.text}</p>}
                </div>
              ) : (
                <Assistant key={m.id} m={m} />
              ),
            )
          )}
        </div>
      </div>
      <Composer busy={busy} onSend={onSend} />
    </section>
  )
}
