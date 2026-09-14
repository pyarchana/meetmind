import { describe, expect, it } from 'vitest'

import { emptyMeetingState, mergeMeetingState, nextBackoff } from './session.js'

describe('nextBackoff', () => {
  it('doubles', () => {
    expect(nextBackoff(1000)).toBe(2000)
  })

  it('stops at thirty seconds', () => {
    expect(nextBackoff(30000)).toBe(30000)
    expect(nextBackoff(1_000_000)).toBe(30000)
  })
})

describe('mergeMeetingState', () => {
  it('replaces only the keys the delta mentions', () => {
    const current = {
      decisions: [{ id: 1, text: 'ship on friday' }],
      questions: [],
      actions: [],
    }
    const merged = mergeMeetingState(current, {
      actions: [{ id: 1, task: 'write the migration', owner: 'priya' }],
    })

    expect(merged.decisions).toEqual(current.decisions)
    expect(merged.actions).toHaveLength(1)
  })

  it('ignores adk bookkeeping keys', () => {
    const merged = mergeMeetingState(emptyMeetingState(), {
      'temp:invocation': 'abc',
      decisions: [{ id: 1, text: 'ship on friday' }],
    })

    expect(Object.keys(merged).sort()).toEqual(['actions', 'decisions', 'questions'])
  })

  it('survives an empty or missing delta', () => {
    expect(mergeMeetingState(emptyMeetingState(), undefined)).toEqual(emptyMeetingState())
  })
})
