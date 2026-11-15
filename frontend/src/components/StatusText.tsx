import type { DocumentStatus } from '../api'

const statusColor: Record<DocumentStatus, string> = {
  UPLOADED: 'text-gray-600',
  PROCESSING: 'text-blue-700',
  COMPLETED: 'text-green-700',
  FAILED: 'text-red-700',
}

export default function StatusText({ status }: { status: DocumentStatus }) {
  return <span className={`text-xs font-medium ${statusColor[status]}`}>{status}</span>
}
