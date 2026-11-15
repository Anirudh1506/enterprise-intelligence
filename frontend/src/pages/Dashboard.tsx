import StatusText from '../components/StatusText'
import { formatBytes, formatDate } from '../format'
import { useDocuments } from '../useDocuments'

export default function Dashboard() {
  const { documents, loading, error } = useDocuments()

  const count = (...statuses: string[]) => documents.filter((d) => statuses.includes(d.status)).length
  const stats = [
    { label: 'Total documents', value: documents.length },
    { label: 'Processed', value: count('COMPLETED') },
    { label: 'Processing', value: count('UPLOADED', 'PROCESSING') },
    { label: 'Failed', value: count('FAILED') },
  ]

  return (
    <div className="space-y-8">
      {error && <p className="text-sm text-red-700">Could not load documents: {error}</p>}

      <dl className="grid grid-cols-2 gap-px overflow-hidden border border-gray-200 bg-gray-200 sm:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="bg-white p-4">
            <dt className="text-sm text-gray-600">{s.label}</dt>
            <dd className="mt-1 text-2xl font-semibold">{loading ? '–' : s.value}</dd>
          </div>
        ))}
      </dl>

      <section>
        <h2 className="mb-2 text-sm font-semibold">Recent uploads</h2>
        {documents.length === 0 && !loading ? (
          <p className="text-sm text-gray-600">
            No documents yet. <a className="underline" href="#/documents">Upload some</a>.
          </p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="border-b border-gray-200 text-gray-600">
              <tr>
                <th className="py-2 font-medium">Name</th>
                <th className="py-2 font-medium">Status</th>
                <th className="hidden py-2 font-medium sm:table-cell">Size</th>
                <th className="hidden py-2 font-medium sm:table-cell">Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {documents.slice(0, 8).map((d) => (
                <tr key={d.id} className="border-b border-gray-100">
                  <td className="py-2 break-all">{d.filename}</td>
                  <td className="py-2"><StatusText status={d.status} /></td>
                  <td className="hidden py-2 sm:table-cell">{formatBytes(d.file_size)}</td>
                  <td className="hidden py-2 sm:table-cell">{formatDate(d.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
