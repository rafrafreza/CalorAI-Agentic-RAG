import { Check, Equal } from 'lucide-react'
import { itemQuantity, kcal, SOURCE_LABEL, timeOf, title, todayISO } from '../format'
import type { MealChange } from '../types'

const STAMP: Record<MealChange['action'], string> = {
  logged: 'Logged',
  updated: 'Corrected',
  deleted: 'Removed',
}

/** A logged meal printed as an itemised receipt: every line carries its provenance. */
export function Receipt({ change }: { change: MealChange }) {
  const { meal, action, previous } = change
  const notes: string[] = []
  const footnote = (text: string) => {
    const i = notes.indexOf(text)
    if (i >= 0) return i + 1
    notes.push(text)
    return notes.length
  }

  return (
    <article className="receipt" data-action={action} aria-label={`${STAMP[action]} meal: ${meal.description}`}>
      <header className="receipt-head">
        <span className="receipt-kind">
          {title(meal.meal_type ?? 'meal')} · {timeOf(meal.timestamp)}
          {meal.date !== todayISO() && ` · ${meal.date}`}
        </span>
        <span className="stamp">{STAMP[action]}</span>
      </header>

      <ol className="receipt-lines">
        {meal.items.map((item, idx) => {
          const marks = (item.assumptions ?? []).filter((a) => !a.startsWith('Not in the knowledge base')).map(footnote)
          return (
            <li key={idx}>
              <div className="line">
                <span className="line-qty">{itemQuantity(item)}</span>
                <span className="line-name">
                  {item.name}
                  {marks.map((n) => (
                    <sup key={n}>{n}</sup>
                  ))}
                </span>
                <span className="leader" aria-hidden />
                <span className="line-kcal">{kcal(item.calories)}</span>
              </div>
              <div className={item.grounded ? 'prov prov-grounded' : 'prov prov-estimate'}>
                {item.grounded ? <Check size={11} strokeWidth={2.5} /> : <Equal size={11} strokeWidth={2.5} />}
                {item.grounded ? SOURCE_LABEL[item.source] ?? item.source : 'Model estimate · not in knowledge base'}
                {item.grounded && item.kb_id && <code>kb:{item.kb_id}</code>}
              </div>
              {(item.conflicts ?? []).map((c) => (
                <div className="prov prov-conflict" key={c}>
                  ⚠ {c}
                </div>
              ))}
            </li>
          )
        })}
      </ol>

      <footer className="receipt-foot">
        <div className="receipt-total">
          <span>Total</span>
          <span className="receipt-total-value">
            {previous && action === 'updated' && <s>{kcal(previous.calories)}</s>}
            <strong>{kcal(meal.calories)}</strong> kcal
          </span>
        </div>
        <div className="receipt-macros">
          P {Math.round(meal.protein_g)} g · C {Math.round(meal.carbs_g)} g · F {Math.round(meal.fat_g)} g
        </div>
        {notes.length > 0 && (
          <ol className="receipt-notes">
            {notes.map((n, i) => (
              <li key={n} value={i + 1}>
                {n}
              </li>
            ))}
          </ol>
        )}
      </footer>
    </article>
  )
}
