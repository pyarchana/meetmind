import { useCallback, useEffect, useRef, useState } from 'react'

import Board from './components/Board.jsx'
import Composer from './components/Composer.jsx'
import StatusPill from './components/StatusPill.jsx'
import Transcript from './components/Transcript.jsx'
import { Microphone, Player } from './lib/audio.js'
import { Session, emptyMeetingState, mergeMeetingState } from './lib/session.js'

const SCREEN_INTERVAL_MS = 5000
const SCREEN_WIDTH = 1280
const SCREEN_HEIGHT = 720
const SCREEN_QUALITY = 0.4

// Fixed for the life of the page so a reconnect lands on the same ADK
// session and the meeting board survives it.
const IDS = {
  user: `u-${Math.random().toString(36).slice(2, 8)}`,
  session: `s-${Math.random().toString(36).slice(2, 10)}`,
}

export default function App() {
  const [status, setStatus] = useState('offline')
  const [meeting, setMeeting] = useState(emptyMeetingState)
  const [lines, setLines] = useState([])
  const [listening, setListening] = useState(false)
  const [sharing, setSharing] = useState(false)

  const player = useRef(null)
  const mic = useRef(null)
  const session = useRef(null)
  const screenTimer = useRef(null)
  const screenStream = useRef(null)

  // Transcripts arrive as progressively longer versions of the same
  // sentence, so each role keeps one open line that gets rewritten until
  // the turn ends.
  const openLines = useRef({ user: null, agent: null })
  const nextLineId = useRef(0)

  const connected = status === 'live'

  const write = useCallback((role, text, { append = false } = {}) => {
    setLines((prev) => {
      const openId = openLines.current[role]
      if (openId !== null) {
        return prev.map((line) =>
          line.id === openId
            ? { ...line, text: append ? line.text + text : text }
            : line,
        )
      }
      const id = nextLineId.current++
      openLines.current[role] = id
      return [...prev, { id, role, text }]
    })
  }, [])

  const note = useCallback((role, text) => {
    setLines((prev) => [...prev, { id: nextLineId.current++, role, text }])
  }, [])

  const handleMessage = useCallback(
    (message) => {
      switch (message.type) {
        case 'audio':
          player.current?.play(message.data)
          break
        case 'transcript_user':
          openLines.current.agent = null
          write('user', message.data)
          break
        case 'transcript_agent':
          openLines.current.user = null
          write('agent', message.data)
          break
        case 'text':
          write('agent', message.data, { append: true })
          break
        case 'meeting_state':
          setMeeting((prev) => mergeMeetingState(prev, message.data))
          break
        case 'turn_complete':
          openLines.current = { user: null, agent: null }
          break
        case 'error':
          note('error', message.data)
          break
        default:
          break
      }
    },
    [write, note],
  )

  const stopScreen = useCallback(() => {
    if (screenTimer.current) clearInterval(screenTimer.current)
    screenTimer.current = null
    if (screenStream.current) {
      for (const track of screenStream.current.getTracks()) track.stop()
      screenStream.current = null
    }
    setSharing(false)
  }, [])

  const endSession = useCallback(() => {
    stopScreen()
    mic.current?.stop()
    mic.current = null
    session.current?.close()
    session.current = null
    player.current?.close()
    player.current = null
    setListening(false)
  }, [stopScreen])

  async function startSession() {
    player.current = new Player()

    session.current = new Session({
      userId: IDS.user,
      sessionId: IDS.session,
      onMessage: handleMessage,
      onStatus: setStatus,
    })
    session.current.connect()

    mic.current = new Microphone({
      onChunk: (data) => session.current?.send({ type: 'audio', data }),
      onSpeechStart: () => {
        setListening(true)
        player.current?.stop() // your voice wins, cut the agent off
        openLines.current.agent = null
      },
      onSpeechEnd: () => setListening(false),
    })

    try {
      await mic.current.start()
    } catch {
      note('error', 'Microphone access denied. Allow it, then start the session again.')
    }
  }

  async function toggleScreen() {
    if (sharing) {
      stopScreen()
      note('system', 'Screen sharing stopped.')
      return
    }

    let stream
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: true })
    } catch {
      note('system', 'Screen sharing cancelled.')
      return
    }

    screenStream.current = stream
    const track = stream.getVideoTracks()[0]
    const capture = new ImageCapture(track)
    track.onended = stopScreen
    setSharing(true)
    note('system', 'Screen sharing on. Ask what is on screen whenever you need it.')

    screenTimer.current = setInterval(async () => {
      if (!session.current?.isOpen) return
      try {
        const frame = await capture.grabFrame()
        const canvas = document.createElement('canvas')
        canvas.width = SCREEN_WIDTH
        canvas.height = SCREEN_HEIGHT
        canvas.getContext('2d').drawImage(frame, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        session.current.send({
          type: 'screen',
          data: canvas.toDataURL('image/jpeg', SCREEN_QUALITY).split(',')[1],
        })
      } catch {
        // one dropped frame is not worth telling anyone about
      }
    }, SCREEN_INTERVAL_MS)
  }

  function sendText(text) {
    player.current?.stop()
    openLines.current = { user: null, agent: null }
    note('user', text)
    session.current?.send({ type: 'text', data: text })
  }

  useEffect(() => endSession, [endSession])

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 p-4 lg:p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-medium tracking-tight text-cream">MeetMind</h1>
          <p className="text-xs text-muted">
            An AI in the room that keeps the notes so nobody else has to
          </p>
        </div>
        <StatusPill status={status} />
      </header>

      <main className="flex min-h-0 flex-1 flex-col gap-4 lg:flex-row">
        <Board meeting={meeting} />
        <Transcript lines={lines} listening={listening} />
      </main>

      <footer>
        <Composer
          connected={connected}
          sharing={sharing}
          onToggleSession={connected ? endSession : startSession}
          onToggleScreen={toggleScreen}
          onSend={sendText}
        />
      </footer>
    </div>
  )
}
