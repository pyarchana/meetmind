const INITIAL_BACKOFF_MS = 1000
const MAX_BACKOFF_MS = 30000

export const MEETING_KEYS = ['decisions', 'questions', 'actions']

export function nextBackoff(current) {
  return Math.min(current * 2, MAX_BACKOFF_MS)
}

/**
 * Fold a state delta into the board.
 *
 * The server reassigns a whole list per key, so a delta already carries the
 * complete value for anything it mentions. Only the three meeting keys are
 * copied across, to keep ADK's own session bookkeeping out of the UI.
 */
export function mergeMeetingState(current, delta) {
  const next = { ...current }
  for (const key of MEETING_KEYS) {
    if (Array.isArray(delta?.[key])) next[key] = delta[key]
  }
  return next
}

export function emptyMeetingState() {
  return { decisions: [], questions: [], actions: [] }
}

/**
 * WebSocket wrapper that reconnects on an unexpected drop.
 *
 * The ids stay fixed for the life of the page, so the backend hands back the
 * same ADK session and the meeting board survives a reconnect.
 */
export class Session {
  constructor({ userId, sessionId, onMessage, onStatus }) {
    this.userId = userId
    this.sessionId = sessionId
    this.onMessage = onMessage
    this.onStatus = onStatus

    this.socket = null
    this.backoff = INITIAL_BACKOFF_MS
    this.retry = null
    this.closedByUser = false
  }

  get url() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${scheme}://${location.host}/ws/${this.userId}/${this.sessionId}`
  }

  get isOpen() {
    return this.socket !== null && this.socket.readyState === 1
  }

  connect() {
    this.closedByUser = false
    this.onStatus('connecting')

    this.socket = new WebSocket(this.url)

    this.socket.onopen = () => {
      this.backoff = INITIAL_BACKOFF_MS
      this.onStatus('live')
    }

    this.socket.onmessage = (event) => {
      let message
      try {
        message = JSON.parse(event.data)
      } catch {
        return
      }
      this.onMessage(message)
    }

    this.socket.onclose = () => {
      this.socket = null
      if (this.closedByUser) {
        this.onStatus('offline')
        return
      }
      this.onStatus('reconnecting')
      this.retry = setTimeout(() => this.connect(), this.backoff)
      this.backoff = nextBackoff(this.backoff)
    }

    this.socket.onerror = () => this.onStatus('error')
  }

  send(payload) {
    if (!this.isOpen) return false
    this.socket.send(JSON.stringify(payload))
    return true
  }

  close() {
    this.closedByUser = true
    if (this.retry) clearTimeout(this.retry)
    this.retry = null
    if (this.socket) this.socket.close()
    this.socket = null
    this.onStatus('offline')
  }
}
