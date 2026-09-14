function Panel({ title, count, accent, empty, children }) {
  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-line bg-panel">
      <header className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="text-sm font-medium tracking-wide text-cream">{title}</h2>
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${accent}`}>
          {count}
        </span>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {count === 0 ? (
          <p className="text-sm leading-relaxed text-muted">{empty}</p>
        ) : (
          <ul className="space-y-2">{children}</ul>
        )}
      </div>
    </section>
  )
}

function Row({ children, muted = false }) {
  return (
    <li
      className={`rounded-lg bg-raised px-3 py-2 text-sm leading-relaxed ${
        muted ? 'text-muted line-through' : 'text-cream'
      }`}
    >
      {children}
    </li>
  )
}

export default function Board({ meeting }) {
  const { decisions, questions, actions } = meeting

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <Panel
        title="Decisions"
        count={decisions.length}
        accent="bg-good/15 text-good"
        empty="Nothing settled yet. MeetMind records each decision as the group makes it."
      >
        {decisions.map((decision) => (
          <Row key={decision.id}>{decision.text}</Row>
        ))}
      </Panel>

      <Panel
        title="Open questions"
        count={questions.filter((q) => !q.answered).length}
        accent="bg-warn/15 text-warn"
        empty="No unresolved questions."
      >
        {questions.map((question) => (
          <Row key={question.id} muted={question.answered}>
            {question.text}
            {question.answered && (
              <span className="mt-1 block text-xs text-muted no-underline">
                {question.answer}
              </span>
            )}
          </Row>
        ))}
      </Panel>

      <Panel
        title="Action items"
        count={actions.length}
        accent="bg-cream/10 text-cream"
        empty="Nobody has committed to anything yet."
      >
        {actions.map((action) => (
          <Row key={action.id} muted={action.done}>
            <span className="mr-2 rounded bg-cream/10 px-1.5 py-0.5 text-xs text-muted">
              {action.owner}
            </span>
            {action.task}
          </Row>
        ))}
      </Panel>
    </div>
  )
}
