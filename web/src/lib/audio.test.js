import { describe, expect, it } from 'vitest'

import {
  FRAME_MS,
  Microphone,
  SILENCE_FRAMES,
  SPEECH_FRAMES,
  float32ToPcm16Base64,
  pcm16ToFloat32,
  rms,
} from './audio.js'

describe('pcm conversion', () => {
  it('survives a round trip', () => {
    const original = Float32Array.from([0, 0.5, -0.5, 0.25, -0.25])
    const restored = pcm16ToFloat32(float32ToPcm16Base64(original))

    expect(restored).toHaveLength(original.length)
    for (let i = 0; i < original.length; i++) {
      expect(restored[i]).toBeCloseTo(original[i], 4)
    }
  })

  it('clamps rather than wrapping around', () => {
    const restored = pcm16ToFloat32(float32ToPcm16Base64(Float32Array.from([2, -2])))
    expect(restored[0]).toBeGreaterThan(0.99)
    expect(restored[1]).toBeCloseTo(-1, 4)
  })

  it('encodes buffers too large to spread through fromCharCode', () => {
    // 4096 samples is one mic frame, which overflows the argument limit
    // when passed via String.fromCharCode.apply.
    const frame = new Float32Array(4096).fill(0.1)
    expect(() => float32ToPcm16Base64(frame)).not.toThrow()
    expect(pcm16ToFloat32(float32ToPcm16Base64(frame))).toHaveLength(4096)
  })
})

describe('rms', () => {
  it('is zero for silence', () => {
    expect(rms(new Float32Array(128))).toBe(0)
  })

  it('rises above the speech threshold for a loud frame', () => {
    expect(rms(new Float32Array(128).fill(0.4))).toBeGreaterThan(0.012)
  })
})

describe('barge in', () => {
  function listener() {
    const events = []
    const mic = new Microphone({
      onChunk: () => {},
      onSpeechStart: () => events.push('start'),
      onSpeechEnd: () => events.push('end'),
    })
    return { mic, events }
  }

  it('waits for the speech run before cutting playback', () => {
    const { mic, events } = listener()
    for (let i = 0; i < SPEECH_FRAMES - 1; i++) mic.track(0.5)
    expect(events).toEqual([])

    mic.track(0.5)
    expect(events).toEqual(['start'])
  })

  it('fires once, not on every loud frame', () => {
    const { mic, events } = listener()
    for (let i = 0; i < 20; i++) mic.track(0.5)
    expect(events).toEqual(['start'])
  })

  it('a single loud frame is not speech', () => {
    const { mic, events } = listener()
    mic.track(0.5)
    mic.track(0)
    mic.track(0.5)
    expect(events).toEqual([])
  })

  it('needs a long pause before calling it the end of a turn', () => {
    const { mic, events } = listener()
    for (let i = 0; i < SPEECH_FRAMES; i++) mic.track(0.5)
    for (let i = 0; i < SILENCE_FRAMES - 1; i++) mic.track(0)
    expect(events).toEqual(['start'])

    mic.track(0)
    expect(events).toEqual(['start', 'end'])
  })

  it('pins what that costs in milliseconds', () => {
    // Local barge in cannot beat this, whatever the network does. Change the
    // mic buffer size and this number moves with it.
    expect(FRAME_MS).toBe(256)
    expect(SPEECH_FRAMES * FRAME_MS).toBe(768)
    expect(SILENCE_FRAMES * FRAME_MS).toBe(6400)
  })
})
