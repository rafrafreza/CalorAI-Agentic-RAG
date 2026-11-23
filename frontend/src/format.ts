import type { MealItem } from './types'

const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })

export const kcal = (n: number | null | undefined) => nf0.format(Math.round(n ?? 0))
export const grams = (n: number | null | undefined) => `${nf0.format(Math.round(n ?? 0))} g`

export const qty = (q: number) => (Number.isInteger(q) ? String(q) : q.toFixed(2).replace(/0+$/, '').replace(/\.$/, ''))

export function itemQuantity(item: Pick<MealItem, 'quantity' | 'unit'>): string {
  if (item.quantity == null) return ''
  if (!item.unit || item.unit === 'piece') return `${qty(item.quantity)} ×`
  return `${qty(item.quantity)} ${item.unit}`
}

export const SOURCE_LABEL: Record<string, string> = {
  'usda-ref': 'USDA ref',
  'recipe-avg': 'Recipe avg',
  'label-avg': 'Label avg',
  'model-estimate': 'Estimate',
}

export function timeOf(timestamp: string): string {
  const d = new Date(timestamp.replace(' ', 'T'))
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function longDate(iso: string): string {
  const d = new Date(`${iso}T12:00:00`)
  return d.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' })
}

export const title = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

/** Local calendar date as YYYY-MM-DD (the server stamps meals in local time). */
export const todayISO = () => new Date().toLocaleDateString('en-CA')

export const uid = () => Math.random().toString(36).slice(2, 10)
