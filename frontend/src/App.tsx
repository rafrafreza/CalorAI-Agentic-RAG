import { PanelRight, SquarePen } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { Chat } from './components/Chat'
import { Explorer } from './components/Explorer'
import { TodayPanel } from './components/TodayPanel'
import { kcal } from './format'
import type { Dashboard, Health } from './types'
import { useChat } from './useChat'

type Tab = 'log' | 'knowledge'

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash === '#knowledge' ? 'knowledge' : 'log'))
  const [health, setHealth] = useState<Health | null>(null)
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [sheet, setSheet] = useState(false)

  const refresh = useCallback(() => {
    api.dashboard().then(setDashboard).catch(() => undefined)
  }, [])

  const { messages, send, busy, newSession } = useChat(refresh)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    refresh()
  }, [refresh])

  useEffect(() => {
    history.replaceState(null, '', tab === 'knowledge' ? '#knowledge' : location.pathname + location.search)
  }, [tab])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setSheet(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const mode = health?.llm.mode
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="wordmark">CalorAI</span>
          <span className="brand-tag">grounded nutrition agent</span>
        </div>
        <nav className="tabs" aria-label="Views">
          <button aria-current={tab === 'log' ? 'page' : undefined} onClick={() => setTab('log')}>
            Log
          </button>
          <button aria-current={tab === 'knowledge' ? 'page' : undefined} onClick={() => setTab('knowledge')}>
            Knowledge
          </button>
        </nav>
        <div className="topbar-end">
          {health ? (
            <span className={`mode mode-${mode}`} title={health.llm.chat_model ?? 'Deterministic offline engine'}>
              <span className="mode-dot" />
              <span className="mode-label">{mode === 'gemini' ? health.llm.chat_model : 'Offline engine'}</span>
              <span className="mode-kb">· {health.rag.foods} foods</span>
            </span>
          ) : (
            <span className="mode mode-down">
              <span className="mode-dot" /> <span className="mode-label">Backend unreachable</span>
            </span>
          )}
          {tab === 'log' && (
            <>
              <button className="icon-btn" onClick={newSession} aria-label="New conversation" title="New conversation">
                <SquarePen size={17} />
              </button>
              <button className="today-pill" onClick={() => setSheet(true)} aria-label="Show today">
                <PanelRight size={15} /> {dashboard ? kcal(dashboard.day.total_calories) : 'Today'}
                {dashboard && <span className="today-pill-unit">kcal</span>}
              </button>
            </>
          )}
        </div>
      </header>

      {tab === 'log' ? (
        <main className="log">
          <Chat messages={messages} busy={busy} onSend={send} mode={mode} />
          <div className={sheet ? 'today-wrap today-wrap-open' : 'today-wrap'} onClick={(e) => e.target === e.currentTarget && setSheet(false)}>
            <TodayPanel
              data={dashboard}
              onDeleteMeal={(id) => api.deleteMeal(id).then(refresh)}
              onForget={(key) => api.forget(key).then(refresh)}
            />
          </div>
        </main>
      ) : (
        <main className="knowledge">
          <Explorer />
        </main>
      )}
    </div>
  )
}
