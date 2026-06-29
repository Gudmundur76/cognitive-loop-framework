/**
 * tests/memory/ctcMemory.test.ts
 * Tests for CTCMemory — the CTC sidecar integration.
 * CTC_DISABLED=1 is set in vitest.config.ts, so the sidecar is always disabled.
 */
import { describe, it, expect, beforeEach } from 'vitest';

// ── Tests ─────────────────────────────────────────────────────────────────────
describe('CTCMemory (CTC_DISABLED=1)', () => {
  beforeEach(() => {
    // CTC_DISABLED=1 is set globally in vitest.config.ts
  });

  it('isEnabled returns false when CTC_DISABLED=1', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    expect(mem.isEnabled).toBe(false);
  });

  it('ingestCycle resolves without error when disabled', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    await expect(
      mem.ingestCycle({
        cycle_id: 'cycle-1',
        timestamp: new Date().toISOString(),
        task: 'test task',
        phases: { Observe: 'observed', Think: 'thought', Plan: 'planned', Act: 'acted', Verify: 'verified' },
        outcome: 'success',
      })
    ).resolves.toBeUndefined();
  });

  it('reconstruct returns a low-confidence "not available" result when disabled', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    const result = await mem.reconstruct('What happened last week?');
    expect(result.confidence).toBe('low');
    expect(result.answer).toMatch(/not available/i);
    expect(result.question).toBe('What happened last week?');
    expect(result.tool_calls_made).toBe(0);
    expect(result.rounds).toBe(0);
    expect(Array.isArray(result.supports)).toBe(true);
    expect(Array.isArray(result.evidence_texts)).toBe(true);
  });

  it('temporalQuery returns an empty events array when disabled', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    const result = await mem.temporalQuery('2024-01-01', '2024-01-31');
    expect(result.events).toHaveLength(0);
    expect(result.count).toBe(0);
    expect(result.start_date).toBe('2024-01-01');
    expect(result.end_date).toBe('2024-01-31');
  });

  it('getEventKeywords returns empty array when disabled', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    const result = await mem.getEventKeywords('event-123');
    expect(result).toEqual([]);
  });

  it('edgesByTag returns empty array when disabled', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    const result = await mem.edgesByTag('protein', 'binding');
    expect(result).toEqual([]);
  });

  it('reconstruct returns reasoning about why it is disabled', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    const result = await mem.reconstruct('test question');
    expect(result.reasoning).toBeTruthy();
  });

  it('singleton ctcMemory is exported', async () => {
    const { ctcMemory } = await import('../../src/memory/ctcMemory.js');
    expect(ctcMemory).toBeDefined();
    expect(ctcMemory.isEnabled).toBe(false);
  });

  it('CTCMemory accepts a custom dbPath', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory('/custom/path/ctc.db');
    expect(mem.isEnabled).toBe(false);
  });

  it('multiple ingestCycle calls do not throw', async () => {
    const { CTCMemory } = await import('../../src/memory/ctcMemory.js');
    const mem = new CTCMemory();
    const cycle = {
      cycle_id: 'cycle-x',
      timestamp: new Date().toISOString(),
      task: 'test task',
      phases: { Observe: 'obs', Think: 'think', Plan: 'plan', Act: 'act', Verify: 'verify' },
      outcome: 'success' as const,
    };
    await expect(Promise.all([
      mem.ingestCycle(cycle),
      mem.ingestCycle({ ...cycle, cycle_id: 'cycle-y' }),
      mem.ingestCycle({ ...cycle, cycle_id: 'cycle-z' }),
    ])).resolves.toBeDefined();
  });
});
