import { Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api'
import { kcal, SOURCE_LABEL } from '../format'
import type { AnalyzeResponse, RagStats, SearchResponse } from '../types'
import { DecisionChip } from './Trace'

const DIETS = ['', 'vegetarian', 'vegan', 'pescatarian']
const ALLERGENS = ['peanut', 'tree_nut', 'dairy', 'gluten', 'egg', 'shellfish']
const FEATURES: [string, string][] = [
  ['exact', 'Exact name/alias'],
  ['form_f1', 'Best-form F1'],
  ['coverage', 'Token coverage'],
  ['fuzzy', 'Phonetic similarity'],
  ['fused', 'RRF fused rank'],
  ['dense', 'Embedding cosine'],
]
const EXAMPLES: { q: string; diet?: string; mode?: 'food' | 'sentence'; why: string }[] = [
  { q: 'parontha', why: 'unseen spelling' },
  { q: 'golgappa', why: 'regional synonym' },
  { q: 'chiken biryani', why: 'typo' },
  { q: 'biryani', diet: 'vegetarian', why: 'profile-aware' },
  { q: 'dragon fruit smoothie', why: 'must abstain' },
  { q: 'leftover biryani, maybe two thirds of the box', mode: 'sentence', why: 'full message' },
]
const DECISION_COPY = {
  grounded: 'Confident match — the agent logs these numbers and cites the source.',
  ambiguous: 'Near-tie between variants — the agent asks only if the calorie difference is material (>20%).',
  no_match: 'Not in the knowledge base — the agent must estimate and label it as an estimate.',
}

function Bar({ value }: { value: number }) {
  return (
    <span className="fbar">
      <span style={{ transform: `scaleX(${Math.max(0, Math.min(1, value))})` }} />
    </span>
  )
}

export function Explorer() {
  const [mode, setMode] = useState<'food' | 'sentence'>('food')
  const [q, setQ] = useState('parontha')
  const [diet, setDiet] = useState('')
  const [allergens, setAllergens] = useState<string[]>([])
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null)
  const [stats, setStats] = useState<RagStats | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.ragStats().then(setStats).catch(() => setStats(null))
  }, [])

  useEffect(() => {
    if (!q.trim()) return
    const t = setTimeout(() => {
      const run = mode === 'food' ? api.search(q, diet || undefined, allergens) : api.analyze(q, diet || undefined, allergens)
      run
        .then((r) => {
          setError(null)
          if (mode === 'food') setResult(r as SearchResponse)
          else setAnalysis(r as AnalyzeResponse)
        })
        .catch((e) => setError(e.message))
    }, 140)
    return () => clearTimeout(t)
  }, [q, mode, diet, allergens])

  const toggleAllergen = (a: string) => setAllergens((xs) => (xs.includes(a) ? xs.filter((x) => x !== a) : [...xs, a]))

  return (
    <div className="explorer">
      <div className="explorer-main">
        <header className="explorer-head">
          <h1>Knowledge Explorer</h1>
          <p>
            Every number CalorAI shows comes from this knowledge base. Query it exactly the way the agent does: three
            retrieval channels, fused with Reciprocal Rank Fusion, reranked on interpretable features, then a calibrated
            decision.
          </p>
        </header>

        <div className="search">
          <div className="segmented" role="tablist" aria-label="Query type">
            {(['food', 'sentence'] as const).map((m) => (
              <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)}>
                {m === 'food' ? 'Food' : 'Message'}
              </button>
            ))}
          </div>
          <label className="search-box">
            <Search size={16} />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={mode === 'food' ? 'Try “dhosa”' : 'Try a full message'} />
          </label>
        </div>

        <div className="filters">
          <span className="filters-label">Simulate profile</span>
          <div className="segmented segmented-sm">
            {DIETS.map((d) => (
              <button key={d || 'any'} aria-pressed={diet === d} onClick={() => setDiet(d)}>
                {d || 'Any diet'}
              </button>
            ))}
          </div>
          <div className="toggles">
            {ALLERGENS.map((a) => (
              <button key={a} aria-pressed={allergens.includes(a)} onClick={() => toggleAllergen(a)}>
                no {a.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        <ul className="examples examples-inline">
          {EXAMPLES.map((ex) => (
            <li key={ex.q + (ex.diet ?? '')}>
              <button
                onClick={() => {
                  setMode(ex.mode ?? 'food')
                  setQ(ex.q)
                  setDiet(ex.diet ?? '')
                }}
              >
                {ex.q}
                <small>{ex.diet ? `${ex.why} · ${ex.diet}` : ex.why}</small>
              </button>
            </li>
          ))}
        </ul>

        {error && <p className="msg-error">{error}</p>}

        {mode === 'food' && result && (
          <section className="results" aria-live="polite">
            <div className={`verdict verdict-${result.decision}`}>
              <DecisionChip decision={result.decision} />
              <span className="verdict-copy">{DECISION_COPY[result.decision]}</span>
              <span className="verdict-meta">
                conf {result.confidence.toFixed(2)} · {result.latency_ms.toFixed(1)} ms · {result.channels.join(' + ')}
                {result.escalated && ' · dense escalated'}
              </span>
            </div>
            {result.profile_promoted && (
              <p className="promoted">
                Profile resolved an implicit choice: the best match conflicted with the simulated diet/allergies, so the
                closest compatible variant was promoted.
              </p>
            )}
            <ol className="cands">
              {result.candidates.map((c, i) => (
                <li key={c.id} className={i === 0 ? 'cand cand-top' : 'cand'}>
                  <div className="cand-head">
                    <span className="cand-rank">{i + 1}</span>
                    <div className="cand-name">
                      <strong>{c.name}</strong>
                      <span className="muted">
                        {c.food.serving} · {kcal(c.food.per_serving.kcal)} kcal · {SOURCE_LABEL[c.food.source]} · {c.food.diet}
                      </span>
                    </div>
                    <div className="cand-score">
                      <span>{c.score.toFixed(3)}</span>
                      <Bar value={c.score} />
                    </div>
                  </div>
                  <div className="cand-body">
                    <div className="cand-channels">
                      {(['bm25', 'fuzzy', 'dense'] as const).map((ch) => (
                        <span key={ch} className={c.channels[ch] ? 'rank' : 'rank rank-off'}>
                          {ch === 'fuzzy' ? 'phonetic' : ch} {c.channels[ch] ? `#${c.channels[ch]!.rank}` : '—'}
                        </span>
                      ))}
                    </div>
                    <dl className="features">
                      {FEATURES.filter(([k]) => k in c.features).map(([k, label]) => (
                        <div key={k}>
                          <dt>{label}</dt>
                          <dd>
                            <Bar value={c.features[k]} /> <span>{c.features[k].toFixed(2)}</span>
                          </dd>
                        </div>
                      ))}
                    </dl>
                    {c.conflicts.length > 0 && <p className="conflicts">⚠ {c.conflicts.join(' · ')}</p>}
                  </div>
                </li>
              ))}
            </ol>
          </section>
        )}

        {mode === 'sentence' && analysis && (
          <section className="results" aria-live="polite">
            <div className="verdict">
              <span className="verdict-copy">
                {analysis.mentions.length} food mention{analysis.mentions.length === 1 ? '' : 's'} extracted · speculative
                retrieval runs before the LLM, so the model logs in one tool round.
              </span>
              <span className="verdict-meta">{kcal(analysis.grounded_kcal)} kcal grounded</span>
            </div>
            <table className="trace-table analysis">
              <thead>
                <tr>
                  <th>Mention</th>
                  <th>Quantity</th>
                  <th>Match</th>
                  <th>Decision</th>
                  <th className="num">kcal</th>
                </tr>
              </thead>
              <tbody>
                {analysis.groundings.map((g, i) => (
                  <tr key={i}>
                    <td>{g.food}</td>
                    <td>{g.quantity != null ? `${g.quantity} ${g.unit ?? ''}` : <span className="muted">1 serving</span>}</td>
                    <td>
                      {g.name ?? '—'}
                      {g.assumptions?.map((a) => (
                        <small key={a} className="assumption">
                          {a}
                        </small>
                      ))}
                    </td>
                    <td>
                      <DecisionChip decision={g.status} />
                    </td>
                    <td className="num">{g.nutrition_for_quantity ? kcal(g.nutrition_for_quantity.kcal) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>

      <aside className="explorer-side">
        <section className="panel-section">
          <h3>Knowledge base</h3>
          {stats ? (
            <>
              <p className="stat-hero">
                {stats.foods}
                <span>foods</span>
              </p>
              <p className="muted">{stats.surface_forms.toLocaleString()} names, aliases & transliterations</p>
              <ul className="kv">
                {Object.entries(stats.by_source).map(([k, v]) => (
                  <li key={k}>
                    <span>{SOURCE_LABEL[k] ?? k}</span>
                    <span>{v}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="empty-line">Backend offline.</p>
          )}
        </section>
        {stats && (
          <section className="panel-section">
            <h3>Channels</h3>
            <ul className="kv">
              {Object.entries(stats.channels).map(([k, v]) => (
                <li key={k}>
                  <span>{k === 'fuzzy' ? 'phonetic n-gram' : k}</span>
                  <span className={`status status-${v}`}>{v}</span>
                </li>
              ))}
            </ul>
            <h3>Calibration</h3>
            <ul className="kv">
              <li>
                <span>abstain below</span>
                <span>{String(stats.config.tau_abstain)}</span>
              </li>
              <li>
                <span>ambiguity margin</span>
                <span>{String(stats.config.ambiguity_margin)}</span>
              </li>
              <li>
                <span>dense escalates below</span>
                <span>{String(stats.config.escalate_below)}</span>
              </li>
              <li>
                <span>RRF k</span>
                <span>{String(stats.config.rrf_k)}</span>
              </li>
            </ul>
          </section>
        )}
      </aside>
    </div>
  )
}
