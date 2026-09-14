// Gemini sends 24kHz PCM back and expects 16kHz on the way up.
export const OUTPUT_SAMPLE_RATE = 24000
export const INPUT_SAMPLE_RATE = 16000

const MIC_BUFFER_SIZE = 4096
export const SPEECH_THRESHOLD = 0.012
export const SPEECH_FRAMES = 3
export const SILENCE_FRAMES = 25

// One mic callback covers this much audio. It sets the resolution of every
// client side timing, barge in included.
export const FRAME_MS = (MIC_BUFFER_SIZE / INPUT_SAMPLE_RATE) * 1000

/** base64 PCM16 to Float32 in [-1, 1). */
export function pcm16ToFloat32(base64) {
  const raw = atob(base64)
  const bytes = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)

  const pcm = new Int16Array(bytes.buffer)
  const samples = new Float32Array(pcm.length)
  for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768
  return samples
}

/** Float32 to base64 PCM16. Built one byte at a time because spreading a
 *  full buffer through String.fromCharCode overflows the argument limit. */
export function float32ToPcm16Base64(samples) {
  const pcm = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    pcm[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768)))
  }

  const bytes = new Uint8Array(pcm.buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

export function rms(samples) {
  let total = 0
  for (let i = 0; i < samples.length; i++) total += samples[i] * samples[i]
  return Math.sqrt(total / samples.length)
}

/**
 * Plays incoming chunks back to back.
 *
 * Each chunk is scheduled at a rolling cursor rather than "now". Starting
 * them all immediately overlaps them into noise, and starting them on a
 * timer leaves audible gaps.
 */
export class Player {
  constructor() {
    this.context = null
    this.nextStart = 0
    this.sources = []
  }

  ensureContext() {
    if (!this.context || this.context.state === 'closed') {
      this.context = new AudioContext()
      this.nextStart = this.context.currentTime
      this.sources = []
    }
    if (this.context.state === 'suspended') this.context.resume()
    return this.context
  }

  play(base64) {
    const context = this.ensureContext()
    const samples = pcm16ToFloat32(base64)

    const buffer = context.createBuffer(1, samples.length, OUTPUT_SAMPLE_RATE)
    buffer.copyToChannel(samples, 0)

    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)

    const startAt = Math.max(context.currentTime, this.nextStart)
    source.start(startAt)
    this.nextStart = startAt + buffer.duration

    this.sources.push(source)
    source.onended = () => {
      this.sources = this.sources.filter((s) => s !== source)
    }
  }

  /** Cut playback the instant somebody speaks. */
  stop() {
    for (const source of this.sources) {
      try {
        source.stop()
      } catch {
        // already ended
      }
    }
    this.sources = []
    if (this.context) this.nextStart = this.context.currentTime
  }

  close() {
    this.stop()
    if (this.context) this.context.close()
    this.context = null
  }
}

/**
 * Captures the microphone and reports speech boundaries.
 *
 * Audio streams continuously, including silence, because Gemini runs its own
 * voice activity detection server side and uses trailing silence to decide a
 * turn has ended. Gating the upload on local speech saves bandwidth but
 * breaks turn taking.
 */
export class Microphone {
  constructor({ onChunk, onSpeechStart, onSpeechEnd }) {
    this.onChunk = onChunk
    this.onSpeechStart = onSpeechStart
    this.onSpeechEnd = onSpeechEnd

    this.stream = null
    this.context = null
    this.processor = null
    this.speaking = false
    this.speechFrames = 0
    this.silenceFrames = 0
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    this.context = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE })

    const source = this.context.createMediaStreamSource(this.stream)
    this.processor = this.context.createScriptProcessor(MIC_BUFFER_SIZE, 1, 1)
    source.connect(this.processor)
    this.processor.connect(this.context.destination)

    this.processor.onaudioprocess = (event) => {
      const samples = event.inputBuffer.getChannelData(0)
      this.track(rms(samples))
      this.onChunk(float32ToPcm16Base64(samples))
    }
  }

  track(level) {
    if (level > SPEECH_THRESHOLD) {
      this.speechFrames += 1
      this.silenceFrames = 0
      if (this.speechFrames >= SPEECH_FRAMES && !this.speaking) {
        this.speaking = true
        this.onSpeechStart()
      }
      return
    }

    this.silenceFrames += 1
    this.speechFrames = 0
    if (this.silenceFrames >= SILENCE_FRAMES && this.speaking) {
      this.speaking = false
      this.onSpeechEnd()
    }
  }

  stop() {
    if (this.processor) {
      this.processor.onaudioprocess = null
      try {
        this.processor.disconnect()
      } catch {
        // already disconnected
      }
      this.processor = null
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop()
      this.stream = null
    }
    if (this.context) {
      this.context.close()
      this.context = null
    }
    this.speaking = false
  }
}
