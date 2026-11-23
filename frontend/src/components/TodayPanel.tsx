import { X } from 'lucide-react'
import { grams, kcal, longDate, timeOf, title } from '../format'
import type { Dashboard, MemoryFact } from '../types'

const CATEGORY_LABEL: Record<string, string> = {
  preference: 'Diet & allergies',
  target: 'Goals',
  shortcut: 'Shortcuts',
  personal: 'Context',
}

function Row({ label, value, goal, strong }: { label: string; value: number; goal?: number; strong?: boolean }) {
  return (
    <div className={strong ? 'nf-row nf-row-strong' : 'nf-row'}>
      <span>
        <b>{label}</b> {grams(value)}
      </span>
      <span className="nf-dv">{goal ? `${Math.round((value / goal) * 100)}%` : ''}</span>
    </div>
  )
}

function Week({ data, goal }: { data: Dashboard['trend']; goal?: number }) {
  const max = Math.max(goal ?? 0, ...data.map((d) => d.calories), 1)
  return (
    <div className="week" role="img" aria-label="Calories over the last 7 days">
      <div className="week-plot">
        {goal ? <span className="week-goal" style={{ bottom: `${(goal / max) * 100}%` }} title={`Goal ${kcal(goal)} kcal`} /> : null}
        {data.map((d, i) => (
          <span key={d.date} className={i === data.length - 1 ? 'week-col week-today' : 'week-col'}>
            <span className="week-bar" style={{ height: `${(d.calories / max) * 100}%` }} title={`${d.date}: ${kcal(d.calories)} kcal`} />
          </span>
        ))}
      </div>
      <div className="week-labels" aria-hidden>
        {data.map((d, i) => (
          <span key={d.date} className={i === data.length - 1 ? 'week-today' : undefined}>
            {new Date(`${d.date}T12:00:00`).toLocaleDateString([], { weekday: 'narrow' })}
          </span>
        ))}
      </div>
    </div>
  )
}

export function TodayPanel({
  data,
  onDeleteMeal,
  onForget,
}: {
  data: Dashboard | null
  onDeleteMeal: (id: number) => void
  onForget: (key: string) => void
}) {
  if (!data) {
    return (
      <aside className="today">
        <div className="nf nf-skeleton" aria-busy />
      </aside>
    )
  }
  const { day, meals, memory, profile } = data
  const goal = day.targets.calories
  const pct = goal ? Math.min(day.total_calories / goal, 1) : 0
  const over = goal ? day.total_calories > goal : false
  const grouped = memory.reduce<Record<string, MemoryFact[]>>((acc, m) => {
    ;(acc[m.category] ??= []).push(m)
    return acc
  }, {})

  return (
    <aside className="today" aria-label="Today">
      <section className="nf" aria-label="Today's nutrition">
        <h2 className="nf-title">Nutrition Facts</h2>
        <p className="nf-sub">
          Today · {longDate(day.date)} · {day.meal_count} meal{day.meal_count === 1 ? '' : 's'}
        </p>
        <div className="nf-rule-xl" />
        <p className="nf-caption">Calories logged</p>
        <div className="nf-hero">
          <span className="nf-cal">{kcal(day.total_calories)}</span>
          {goal ? <span className="nf-goal">of {kcal(goal)}</span> : <span className="nf-goal">no goal set</span>}
        </div>
        {goal ? (
          <div className={over ? 'budget budget-over' : 'budget'}>
            <span className="budget-fill" style={{ transform: `scaleX(${pct})` }} />
          </div>
        ) : null}
        {goal ? (
          <p className="nf-left">
            {over ? `${kcal(day.total_calories - goal)} kcal over` : `${kcal(goal - day.total_calories)} kcal left`}
          </p>
        ) : (
          <p className="nf-left">Tell me a goal, e.g. “my goal is 2000 kcal”.</p>
        )}
        <div className="nf-rule-md" />
        <p className="nf-dv-head">% of goal</p>
        <Row label="Protein" value={day.total_protein_g} goal={day.targets.protein_g} strong />
        <Row label="Carbohydrate" value={day.total_carbs_g} goal={day.targets.carbs_g} />
        <Row label="Fat" value={day.total_fat_g} goal={day.targets.fat_g} />
        <Row label="Fiber" value={day.total_fiber_g} />
        <div className="nf-rule-xl" />
        <p className="nf-foot">
          {data.grounded_share == null
            ? 'Every logged item is grounded in the knowledge base or labelled as an estimate.'
            : `${Math.round(data.grounded_share * 100)}% of today's calories come from knowledge-base entries; the rest are labelled estimates.`}
        </p>
      </section>

      <section className="panel-section">
        <h3>Meals</h3>
        {meals.length === 0 ? (
          <p className="empty-line">Nothing logged yet today.</p>
        ) : (
          <ul className="meal-list">
            {meals.map((m) => (
              <li key={m.id}>
                <span className="meal-time">{timeOf(m.timestamp)}</span>
                <span className="meal-desc">
                  <span className="meal-type">{title(m.meal_type ?? 'meal')}</span>
                  {m.description}
                </span>
                <span className="meal-kcal">{kcal(m.calories)}</span>
                <button className="icon-btn icon-btn-sm meal-del" onClick={() => onDeleteMeal(m.id)} aria-label={`Delete ${m.description}`}>
                  <X size={13} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel-section">
        <h3>Last 7 days</h3>
        <Week data={data.trend} goal={goal} />
      </section>

      <section className="panel-section">
        <h3>What I remember</h3>
        {(profile.diet || profile.allergens.length > 0) && (
          <p className="profile-line">
            Shapes retrieval: {[profile.diet && title(profile.diet), ...profile.allergens.map((a) => `no ${a.replace('_', ' ')}`)]
              .filter(Boolean)
              .join(' · ')}
          </p>
        )}
        {memory.length === 0 ? (
          <p className="empty-line">Nothing yet. Mention a diet, allergy, goal or “my usual”.</p>
        ) : (
          Object.entries(grouped).map(([cat, facts]) => (
            <div key={cat} className="fact-group">
              <p className="fact-group-label">{CATEGORY_LABEL[cat] ?? cat}</p>
              <ul className="memory-list">
                {facts.map((f) => (
                  <li key={f.key}>
                    <span>{f.value}</span>
                    <button className="icon-btn icon-btn-sm" onClick={() => onForget(f.key)} aria-label={`Forget ${f.value}`}>
                      <X size={13} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
      </section>
    </aside>
  )
}
