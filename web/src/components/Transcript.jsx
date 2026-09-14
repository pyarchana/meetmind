import { useEffect, useRef } from 'react'

const ROLE = {
  user: 'border-l-2 border-cream/30 pl-3 text-cream',
  agent: 'border-l-2 border-good/50 pl-3 text-cream',
  system: 'text-muted italic',
  error: 'text-bad',
}

export default function Transcript({ lines, listening }) {
  const bottom = useRef(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [lines])

  return (
    <section className="flex min-h-0 w-full flex-col rounded-xl border border-line bg-panel lg:w-96">
      <header className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="text-sm font-medium tracking-wide text-cream">Transcript</h2>
        {listening && <span className="text-xs text-good">listening</span>}
      </header>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3">
        {lines.length === 0 ? (
          <p className="text-sm leading-relaxed text-muted">
            Start the session and talk. Everything said in the room lands here.
          </p>
        ) : (
          lines.map((line) => (
            <p key={line.id} className={`text-sm leading-relaxed ${ROLE[line.role]}`}>
              {line.text}
            </p>
          ))
        )}
        <div ref={bottom} />
      </div>
    </section>
  )
}
