import { useEffect, useState } from 'react'
import { api, type Health } from './api'
import Ask from './pages/Ask'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'

const pages = {
  dashboard: { title: 'Dashboard', component: Dashboard },
  documents: { title: 'Documents', component: Documents },
  ask: { title: 'Ask', component: Ask },
} as const

type PageKey = keyof typeof pages

function currentPage(): PageKey {
  const key = window.location.hash.replace('#/', '')
  return key in pages ? (key as PageKey) : 'dashboard'
}

export default function App() {
  const [page, setPage] = useState<PageKey>(currentPage)
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState(false)

  useEffect(() => {
    const onHash = () => setPage(currentPage())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    api.health().then(setHealth, () => setHealthError(true))
  }, [])

  const Page = pages[page].component
  const mock = health && (health.embedding_model.startsWith('mock') || health.llm_model.startsWith('mock'))

  return (
    <div className="min-h-screen">
      <header className="border-b border-gray-200">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <span className="font-semibold">Enterprise Intelligence</span>
          <nav className="flex gap-4 text-sm">
            {(Object.keys(pages) as PageKey[]).map((key) => (
              <a
                key={key}
                href={`#/${key}`}
                className={page === key ? 'font-medium text-gray-900 underline underline-offset-4' : 'text-gray-600 hover:text-gray-900'}
              >
                {pages[key].title}
              </a>
            ))}
          </nav>
        </div>
      </header>

      {healthError && (
        <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-center text-sm text-red-800">
          Cannot reach the backend API. Is it running on port 8000?
        </div>
      )}
      {mock && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-center text-sm text-amber-900">
          Running with mock AI providers (no GEMINI_API_KEY). Search is keyword-based and answers are not generated.
        </div>
      )}

      <main className="mx-auto max-w-5xl px-4 py-6">
        <h1 className="mb-6 text-lg font-semibold">{pages[page].title}</h1>
        <Page />
      </main>

      {health && (
        <footer className="mx-auto max-w-5xl px-4 pb-6 text-xs text-gray-500">
          Embeddings: {health.embedding_model} · LLM: {health.llm_model} · Database: {health.database}
        </footer>
      )}
    </div>
  )
}
