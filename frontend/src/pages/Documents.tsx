import { useEffect, useRef, useState } from 'react'
import { api, type DocumentDetail, type DocumentItem } from '../api'
import StatusText from '../components/StatusText'
import { formatBytes, formatDate } from '../format'
import { useDocuments } from '../useDocuments'

export default function Documents() {
  const { documents, loading, error, refresh } = useDocuments()
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function onUpload(files: FileList | null) {
    if (!files || files.length === 0) return
    setUploading(true)
    setMessage(null)
    try {
      const created = await api.uploadDocuments(Array.from(files))
      setMessage(`Uploaded ${created.length} file(s). Processing has started.`)
      await refresh()
    } catch (e) {
      setMessage(`Upload failed: ${(e as Error).message}`)
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  async function run(action: () => Promise<unknown>) {
    setMessage(null)
    try {
      await action()
    } catch (e) {
      setMessage((e as Error).message)
    }
    await refresh()
  }

  return (
    <div className="space-y-6">
      <section className="border border-gray-200 p-4">
        <label className="block text-sm font-semibold" htmlFor="upload">
          Upload documents
        </label>
        <p className="mt-1 text-sm text-gray-600">PDF, TXT or Markdown. Up to 20 files at once.</p>
        <input
          ref={inputRef}
          id="upload"
          type="file"
          multiple
          accept=".pdf,.txt,.md"
          disabled={uploading}
          onChange={(e) => onUpload(e.target.files)}
          className="mt-3 block w-full text-sm file:mr-3 file:border file:border-gray-300 file:bg-gray-50 file:px-3 file:py-1.5 file:text-sm hover:file:bg-gray-100"
        />
        {uploading && <p className="mt-2 text-sm text-gray-600">Uploading…</p>}
        {message && <p className="mt-2 text-sm">{message}</p>}
      </section>

      {error && <p className="text-sm text-red-700">Could not load documents: {error}</p>}

      {!loading && documents.length === 0 ? (
        <p className="text-sm text-gray-600">No documents uploaded yet.</p>
      ) : (
        <table className="w-full text-left text-sm">
          <thead className="border-b border-gray-200 text-gray-600">
            <tr>
              <th className="py-2 font-medium">Name</th>
              <th className="py-2 font-medium">Status</th>
              <th className="hidden py-2 font-medium md:table-cell">Type</th>
              <th className="hidden py-2 font-medium md:table-cell">Size</th>
              <th className="hidden py-2 font-medium md:table-cell">Chunks</th>
              <th className="hidden py-2 font-medium lg:table-cell">Uploaded</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {documents.map((d) => (
              <DocumentRow
                key={d.id}
                doc={d}
                open={openId === d.id}
                onToggle={() => setOpenId(openId === d.id ? null : d.id)}
                onDelete={() => {
                  if (confirm(`Delete ${d.filename}? Its chunks and embeddings will be removed.`))
                    run(() => api.deleteDocument(d.id))
                }}
                onReprocess={() => run(() => api.reprocessDocument(d.id))}
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function DocumentRow(props: {
  doc: DocumentItem
  open: boolean
  onToggle: () => void
  onDelete: () => void
  onReprocess: () => void
}) {
  const { doc, open } = props
  const busy = doc.status === 'UPLOADED' || doc.status === 'PROCESSING'

  return (
    <>
      <tr className="border-b border-gray-100 align-top">
        <td className="py-2 break-all">
          <button type="button" onClick={props.onToggle} className="text-left underline-offset-2 hover:underline">
            {doc.filename}
          </button>
          {doc.error && <div className="mt-1 text-xs text-red-700">{doc.error}</div>}
        </td>
        <td className="py-2"><StatusText status={doc.status} /></td>
        <td className="hidden py-2 uppercase md:table-cell">{doc.file_type}</td>
        <td className="hidden py-2 md:table-cell">{formatBytes(doc.file_size)}</td>
        <td className="hidden py-2 md:table-cell">{doc.status === 'COMPLETED' ? doc.chunk_count : '–'}</td>
        <td className="hidden py-2 lg:table-cell">{formatDate(doc.created_at)}</td>
        <td className="py-2 text-right whitespace-nowrap">
          {(doc.status === 'FAILED' || doc.status === 'COMPLETED') && (
            <button type="button" onClick={props.onReprocess} className="mr-3 text-gray-700 hover:underline">
              Reprocess
            </button>
          )}
          <button
            type="button"
            onClick={props.onDelete}
            disabled={busy}
            className="text-red-700 hover:underline disabled:text-gray-400 disabled:no-underline"
          >
            Delete
          </button>
        </td>
      </tr>
      {open && (
        <tr className="border-b border-gray-200 bg-gray-50">
          <td colSpan={7} className="p-4">
            <DocumentDetails id={doc.id} status={doc.status} />
          </td>
        </tr>
      )}
    </>
  )
}

function DocumentDetails({ id, status }: { id: string; status: string }) {
  const [detail, setDetail] = useState<DocumentDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  // reload when the document's status changes (e.g. PROCESSING -> COMPLETED)
  useEffect(() => {
    let active = true
    api.getDocument(id).then(
      (d) => active && setDetail(d),
      (e: Error) => active && setError(e.message),
    )
    return () => {
      active = false
    }
  }, [id, status])

  if (error) return <p className="text-sm text-red-700">{error}</p>
  if (!detail) return <p className="text-sm text-gray-600">Loading…</p>

  const meta: [string, string][] = [
    ['Document ID', detail.id],
    ['Content type', detail.content_type],
    ['Pages', detail.page_count?.toString() ?? '–'],
    ['Characters extracted', detail.char_count?.toLocaleString() ?? '–'],
    ['Embedding model', detail.embedding_model ?? '–'],
    ['Processed at', detail.processed_at ? formatDate(detail.processed_at) : '–'],
  ]

  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
        {meta.map(([k, v]) => (
          <div key={k} className="flex gap-2">
            <dt className="text-gray-600">{k}:</dt>
            <dd className="break-all">{v}</dd>
          </div>
        ))}
      </dl>

      {detail.chunks.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold">Chunks ({detail.chunks.length})</h3>
          <ol className="max-h-96 space-y-2 overflow-y-auto">
            {detail.chunks.map((c) => (
              <li key={c.id} className="border border-gray-200 bg-white p-3 text-sm">
                <div className="mb-1 text-xs text-gray-600">
                  #{c.chunk_index}
                  {c.page_number != null && ` · page ${c.page_number}`} · {c.embedding_dimensions}-dim vector [
                  {c.embedding_preview.join(', ')}, …]
                </div>
                <p className="whitespace-pre-wrap">{c.content}</p>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}
