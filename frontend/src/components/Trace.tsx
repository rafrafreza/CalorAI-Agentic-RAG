import { ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { kcal } from '../format'
import type { ChatMessage, Decision, Grounding } from '../types'

const DECISION: Record<Decision, string> = { grounded: 'Grounded', ambiguous: 'Ambiguous', no_match: 'No match' }
const NODE_LABEL: Record<string, string> = {
  vision: 'Vision',
  retrieve: 'Retrieve',
  agent: 'Agent',
  tools: 'Tools',
}

export function DecisionChip({ decision }: { decision: Decision }) {
  return <span className={`chip chip-${decision}`}>{DECISION[decision]}</span>
}

function ChannelRanks({ g }: { g: Grounding }) {
  const top = g.candidates[0]
  if (!top) return null
  return (
    <span className="ranks">
      {(['bm25', 'fuzzy', 'dense'] as const).map((c) => (
        <span key={c} className={top.channels[c] ? 'rank' : 'rank rank-off'} title={`${c} rank`}>
          {c === 'fuzzy' ? 'phon' : c} {top.channels[c] ? `#${top.channels[c]!.rank}` : '—'}
        </span>
      ))}
    </span>
  )
}

/** "How I got this": the turn's graph spans, retrieval decisions and memory, collapsed to one line. */
export function Trace({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false)
  const spans = message.trace ?? []
  const groundings = message.groundings ?? []
  const memories = message.memories ?? []
  if (!spans.length && !groundings.length) return null

  const total = spans.reduce((s, x) => s + x.ms, 0) || 1
  const tools = spans.flatMap((s) => (Array.isArray(s.calls) ? (s.calls as string[]) : []))
  const grounded = groundings.filter((g) => g.status !== 'no_match').length
  const tokens = spans.reduce((s, x) => s + ((x.input_tokens as number) ?? 0) + ((x.output_tokens as number) ?? 0), 0)

  return (
    <div className={open ? 'trace trace-open' : 'trace'}>
      <button className="trace-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <ChevronRight size={13} className="trace-chevron" />
        <span>How I got this</span>
        <span className="trace-sep">·</span>
        <span>
          {grounded}/{groundings.length} food{groundings.length === 1 ? '' : 's'} grounded
        </span>
        {tools.length > 0 && (
          <>
            <span className="trace-sep">·</span>
            <span>{[...new Set(tools)].join(', ')}</span>
          </>
        )}
        {message.latency_ms != null && (
          <>
            <span className="trace-sep">·</span>
            <span>{(message.latency_ms / 1000).toFixed(message.latency_ms < 1000 ? 2 : 1)} s</span>
          </>
        )}
        {message.mode && message.mode !== 'gemini' && <span className="trace-mode">{message.mode}</span>}
      </button>

      {open && (
        <div className="trace-body">
          {spans.length > 0 && (
            <section>
              <h4>Graph</h4>
              <ul className="spans">
                {spans.map((s, i) => (
                  <li key={i}>
                    <span className="span-name">{NODE_LABEL[s.name] ?? s.name}</span>
                    <span className="span-bar">
                      <span style={{ width: `${Math.max(2, (s.ms / total) * 100)}%` }} />
                    </span>
                    <span className="span-ms">{s.ms < 10 ? s.ms.toFixed(1) : Math.round(s.ms)} ms</span>
                    <span className="span-attr">
                      {Array.isArray(s.calls) && (s.calls as string[]).join(', ')}
                      {Array.isArray(s.tool_calls) && (s.tool_calls as string[]).length > 0 && `→ ${(s.tool_calls as string[]).join(', ')}`}
                      {typeof s.mentions === 'number' && `${s.grounded}/${s.mentions} grounded · ${s.memories} facts`}
                      {s.mode === 'offline' && 'offline engine'}
                      {s.mode === 'degraded' && 'degraded → offline engine'}
                    </span>
                  </li>
                ))}
              </ul>
              {tokens > 0 && <p className="trace-note">{tokens.toLocaleString()} tokens</p>}
            </section>
          )}

          {groundings.length > 0 && (
            <section>
              <h4>Retrieval</h4>
              <table className="trace-table">
                <thead>
                  <tr>
                    <th>Mention</th>
                    <th>Match</th>
                    <th>Decision</th>
                    <th className="num">kcal</th>
                    <th>Channels</th>
                  </tr>
                </thead>
                <tbody>
                  {groundings.map((g, i) => (
                    <tr key={i}>
                      <td>
                        {g.quantity != null && <span className="muted">{g.quantity} {g.unit ?? ''} </span>}
                        {g.food}
                      </td>
                      <td>{g.name ?? <span className="muted">{g.closest?.[0] ? `≈ ${g.closest[0]}` : '—'}</span>}</td>
                      <td>
                        <DecisionChip decision={g.status} /> <span className="muted">{g.confidence.toFixed(2)}</span>
                      </td>
                      <td className="num">{g.nutrition_for_quantity ? kcal(g.nutrition_for_quantity.kcal) : '—'}</td>
                      <td>
                        <ChannelRanks g={g} />
                        {g.escalated && <span className="rank rank-dense">dense ↑</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {memories.length > 0 && (
            <section>
              <h4>Memory in context</h4>
              <ul className="facts">
                {memories.map((m) => (
                  <li key={m.key}>
                    <span className="fact-cat">{m.pinned ? 'pinned' : m.category}</span> {m.value}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </div>
  )
}
