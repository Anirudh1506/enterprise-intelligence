import { useCallback, useEffect, useState } from 'react'
import { api, type DocumentItem } from './api'

const isBusy = (d: DocumentItem) => d.status === 'UPLOADED' || d.status === 'PROCESSING'

/** Loads documents and polls every 2s while any document is still being processed. */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(
    () =>
      api
        .listDocuments()
        .then(
          (docs) => {
            setDocuments(docs)
            setError(null)
          },
          (e: Error) => setError(e.message),
        )
        .finally(() => setLoading(false)),
    [],
  )

  useEffect(() => {
    refresh()
  }, [refresh])

  const busy = documents.some(isBusy)
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(refresh, 2000)
    return () => clearInterval(timer)
  }, [busy, refresh])

  return { documents, loading, error, refresh }
}
