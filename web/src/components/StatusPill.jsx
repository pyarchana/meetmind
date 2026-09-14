const LOOK = {
  live: { label: 'Live', dot: 'bg-good', text: 'text-good', ring: 'border-good/40' },
  connecting: { label: 'Connecting', dot: 'bg-warn', text: 'text-warn', ring: 'border-warn/40' },
  reconnecting: { label: 'Reconnecting', dot: 'bg-warn', text: 'text-warn', ring: 'border-warn/40' },
  error: { label: 'Error', dot: 'bg-bad', text: 'text-bad', ring: 'border-bad/40' },
  offline: { label: 'Offline', dot: 'bg-muted', text: 'text-muted', ring: 'border-line' },
}

export default function StatusPill({ status }) {
  const look = LOOK[status] ?? LOOK.offline
  const pulsing = status === 'live' || status === 'reconnecting'

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${look.ring} ${look.text}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${look.dot} ${pulsing ? 'animate-pulse' : ''}`}
      />
      {look.label}
    </span>
  )
}
