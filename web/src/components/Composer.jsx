import { useState } from 'react'

function Button({ onClick, disabled, tone = 'quiet', children }) {
  const tones = {
    primary: 'bg-cream text-ink hover:bg-cream/85',
    danger: 'border border-bad/40 text-bad hover:bg-bad/10',
    active: 'border border-good/40 text-good hover:bg-good/10',
    quiet: 'border border-line text-muted hover:border-cream/30 hover:text-cream',
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${tones[tone]}`}
    >
      {children}
    </button>
  )
}

export default function Composer({ connected, sharing, onToggleSession, onToggleScreen, onSend }) {
  const [draft, setDraft] = useState('')

  function submit(event) {
    event.preventDefault()
    const text = draft.trim()
    if (!text || !connected) return
    onSend(text)
    setDraft('')
  }

  return (
    <form onSubmit={submit} className="flex flex-wrap items-center gap-3">
      <Button onClick={onToggleSession} tone={connected ? 'danger' : 'primary'}>
        {connected ? 'End session' : 'Start session'}
      </Button>

      <Button onClick={onToggleScreen} disabled={!connected} tone={sharing ? 'active' : 'quiet'}>
        {sharing ? 'Sharing screen' : 'Share screen'}
      </Button>

      <input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        disabled={!connected}
        placeholder={connected ? 'Ask without speaking' : 'Start a session to type'}
        className="min-w-48 flex-1 rounded-lg border border-line bg-panel px-4 py-2 text-sm text-cream placeholder:text-muted focus:border-cream/30 focus:outline-none disabled:opacity-40"
      />

      <Button onClick={submit} disabled={!connected || !draft.trim()}>
        Send
      </Button>
    </form>
  )
}
