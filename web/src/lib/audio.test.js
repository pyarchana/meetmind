import { describe, expect, it } from 'vitest'

import { float32ToPcm16Base64, pcm16ToFloat32, rms } from './audio.js'

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
