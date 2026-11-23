export type MealAction = 'logged' | 'updated' | 'deleted'

export interface MealItem {
  name: string
  query?: string
  kb_id: string | null
  quantity: number | null
  unit: string | null
  grams?: number | null
  calories: number
  protein_g: number
  carbs_g: number
  fat_g: number
  fiber_g?: number
  grounded: boolean
  source: string
  confidence?: number | null
  assumptions?: string[]
  conflicts?: string[]
}

export interface Meal {
  id: number
  timestamp: string
  date: string
  meal_type: string | null
  description: string
  calories: number
  protein_g: number
  carbs_g: number
  fat_g: number
  fiber_g: number
  items: MealItem[]
  source: string
}

export interface MealChange {
  action: MealAction
  meal: Meal
  previous?: { description: string; calories: number } | null
}

export interface ChannelHit {
  rank: number
  score: number
}

export interface Candidate {
  id: string
  name: string
  score: number
  fused: number
  channels: Partial<Record<'bm25' | 'fuzzy' | 'dense', ChannelHit>>
  features: Record<string, number>
  conflicts: string[]
}

export type Decision = 'grounded' | 'ambiguous' | 'no_match'

export interface Grounding {
  food: string
  status: Decision
  confidence: number
  quantity?: number
  unit?: string
  kb_id?: string
  name?: string
  serving?: string
  nutrition_for_quantity?: { kcal: number; protein_g: number; carbs_g: number; fat_g: number; fiber_g: number; grams: number }
  source?: string
  assumptions?: string[]
  alternatives?: { kb_id: string; name: string; kcal: number }[]
  material_ambiguity?: boolean
  conflicts?: string[]
  note?: string
  closest?: string[]
  channels: string[]
  escalated: boolean
  latency_ms: number
  candidates: Candidate[]
}

export interface MemoryFact {
  key: string
  value: string
  category: 'preference' | 'target' | 'shortcut' | 'personal' | string
  pinned?: boolean
  score?: number
}

export interface Span {
  name: string
  ms: number
  [attr: string]: unknown
}

export interface DaySummary {
  date: string
  total_calories: number
  total_protein_g: number
  total_carbs_g: number
  total_fat_g: number
  total_fiber_g: number
  meal_count: number
  targets: Partial<Record<'calories' | 'protein_g' | 'carbs_g' | 'fat_g', number>>
  remaining: Partial<Record<'calories' | 'protein_g' | 'carbs_g' | 'fat_g', number>>
}

export interface Dashboard {
  date: string
  day: DaySummary
  meals: Meal[]
  trend: { date: string; calories: number; protein_g: number; meals: number }[]
  memory: MemoryFact[]
  profile: { diet: string | null; allergens: string[] }
  grounded_share: number | null
  items_logged: number
}

export interface Health {
  status: string
  version: string
  llm: { mode: 'gemini' | 'offline' | 'scripted'; chat_model: string | null }
  rag: { foods: number; channels: Record<string, string> }
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  image?: string
  streaming?: boolean
  step?: string
  meals?: MealChange[]
  groundings?: Grounding[]
  memories?: MemoryFact[]
  trace?: Span[]
  latency_ms?: number
  mode?: string
  error?: string
}

export interface SearchResponse {
  query: string
  decision: Decision
  confidence: number
  latency_ms: number
  channels: string[]
  escalated: boolean
  profile_promoted: boolean
  candidates: (Candidate & {
    food: {
      id: string
      name: string
      aliases: string[]
      description: string
      cuisine: string
      category: string
      serving: string
      per_serving: { kcal: number; protein_g: number; carbs_g: number; fat_g: number; fiber_g: number }
      diet: string
      allergens: string[]
      source: string
      source_label: string
    }
  })[]
}

export interface AnalyzeResponse {
  text: string
  mentions: { text: string; food: string; quantity: number | null; unit: string | null; vague: boolean }[]
  groundings: Grounding[]
  grounded_kcal: number
}

export interface RagStats {
  foods: number
  surface_forms: number
  by_cuisine: Record<string, number>
  by_source: Record<string, number>
  fingerprint: string
  channels: Record<string, string>
  config: Record<string, unknown>
}
