export type DocumentStatus = 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED'

export interface DocumentItem {
  id: string
  filename: string
  file_type: string
  content_type: string
  file_size: number
  status: DocumentStatus
  error: string | null
  page_count: number | null
  char_count: number | null
  chunk_count: number
  embedding_model: string | null
  created_at: string
  updated_at: string
  processed_at: string | null
}

export interface ChunkItem {
  id: string
  chunk_index: number
  page_number: number | null
  content: string
  embedding_dimensions: number
  embedding_preview: number[]
}

export interface DocumentDetail extends DocumentItem {
  chunks: ChunkItem[]
}

export interface Source {
  source_number: number
  cited: boolean
  chunk_id: string
  document_id: string
  filename: string
  chunk_index: number
  page_number: number | null
  content: string
  score: number
}

export interface AskResponse {
  question: string
  answer: string
  insufficient_context: boolean
  llm: { provider: string; model: string }
  sources: Source[]
}

export interface Health {
  status: string
  database: string
  embedding_model: string
  llm_model: string
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') message = body.detail
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      // non-JSON error body; keep the generic message
    }
    throw new Error(message)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  health: () => request<Health>('/health'),
  listDocuments: () => request<DocumentItem[]>('/api/v1/documents'),
  getDocument: (id: string) => request<DocumentDetail>(`/api/v1/documents/${id}`),
  deleteDocument: (id: string) => request<void>(`/api/v1/documents/${id}`, { method: 'DELETE' }),
  reprocessDocument: (id: string) =>
    request<DocumentItem>(`/api/v1/documents/${id}/reprocess`, { method: 'POST' }),
  uploadDocuments: (files: File[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return request<DocumentItem[]>('/api/v1/documents', { method: 'POST', body: form })
  },
  ask: (question: string, topK: number) =>
    request<AskResponse>('/api/v1/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK }),
    }),
}
