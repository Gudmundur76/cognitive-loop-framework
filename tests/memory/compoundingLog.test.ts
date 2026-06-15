/**
 * compoundingLog.test.ts
 *
 * Tests for the CompoundingLog — the structured memory log that powers
 * the self-improvement mechanics (self_improvement_mechanics.md).
 *
 * Ralph Wiggum loop: RED → GREEN → VALIDATE → COMPLETE
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { CompoundingLog } from '../../src/memory/compoundingLog.js';
import type { CompoundingEntry } from '../../src/memory/compoundingLog.js';

// ── Fixtures ──────────────────────────────────────────────────────────────────
function makeEntry(
  overrides: Partial<Omit<CompoundingEntry, 'timestamp'>> = {}
): Omit<CompoundingEntry, 'timestamp'> {
  return {
    trigger: 'test_failure',
    test: 'verifyClaim.should_handle_429',
    diagnosis: 'in-memory rate limiter resets on restart',
    patch_hash: 'a1b2c3d4',
    test_result: 'PASS',
    graph_delta: { added: ['RateLimiter.ts'], removed: [], modified: ['verifyClaim.ts'] },
    slm_confidence: 0.94,
    frontier_explored: 3,
    meta_review: 'APPROVED',
    ...overrides,
  };
}

/** Create a fresh CompoundingLog backed by a unique temp file for each test. */
function makeLog(): { log: CompoundingLog; logPath: string } {
  const logPath = path.join(
    os.tmpdir(),
    `compounding_log_test_${Date.now()}_${Math.random().toString(36).slice(2)}.jsonl`
  );
  return { log: new CompoundingLog(logPath), logPath };
}

// ── Append ────────────────────────────────────────────────────────────────────
describe('CompoundingLog.append', () => {
  it('appends an entry and stamps it with an ISO timestamp', () => {
    const { log } = makeLog();
    const entry = log.append(makeEntry());
    expect(entry.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    expect(entry.test).toBe('verifyClaim.should_handle_429');
  });

  it('writes a valid JSONL line to disk', () => {
    const { log, logPath } = makeLog();
    log.append(makeEntry());
    const content = fs.readFileSync(logPath, 'utf8');
    const parsed = JSON.parse(content.trim()) as CompoundingEntry;
    expect(parsed.trigger).toBe('test_failure');
    expect(parsed.slm_confidence).toBe(0.94);
  });

  it('appends multiple entries as separate lines', () => {
    const { log, logPath } = makeLog();
    log.append(makeEntry({ test: 'test-A' }));
    log.append(makeEntry({ test: 'test-B' }));
    const lines = fs.readFileSync(logPath, 'utf8').trim().split('\n');
    expect(lines).toHaveLength(2);
    const first = JSON.parse(lines[0]!) as CompoundingEntry;
    const second = JSON.parse(lines[1]!) as CompoundingEntry;
    expect(first.test).toBe('test-A');
    expect(second.test).toBe('test-B');
  });
});

// ── ReadAll ───────────────────────────────────────────────────────────────────
describe('CompoundingLog.readAll', () => {
  it('returns empty array when log does not exist', () => {
    const emptyLog = new CompoundingLog(
      path.join(os.tmpdir(), `nonexistent_${Date.now()}.jsonl`)
    );
    expect(emptyLog.readAll()).toEqual([]);
  });

  it('reads all entries in insertion order', () => {
    const { log } = makeLog();
    log.append(makeEntry({ test: 'first' }));
    log.append(makeEntry({ test: 'second' }));
    log.append(makeEntry({ test: 'third' }));
    const entries = log.readAll();
    expect(entries).toHaveLength(3);
    expect(entries[0]!.test).toBe('first');
    expect(entries[2]!.test).toBe('third');
  });

  it('skips malformed lines without throwing', () => {
    const { log, logPath } = makeLog();
    fs.writeFileSync(
      logPath,
      '{"valid":true,"trigger":"test_failure","test":"ok","diagnosis":"d","patch_hash":"","test_result":"PASS","graph_delta":{"added":[],"removed":[],"modified":[]},"slm_confidence":0.5,"frontier_explored":0,"meta_review":"APPROVED","timestamp":"2026-01-01T00:00:00Z"}\n{INVALID JSON}\n'
    );
    const entries = log.readAll();
    expect(entries).toHaveLength(1);
  });
});

// ── QuerySimilar ──────────────────────────────────────────────────────────────
describe('CompoundingLog.querySimilar', () => {
  it('returns empty array when log is empty', async () => {
    const { log } = makeLog();
    const results = await log.querySimilar('rate limiter');
    expect(results).toEqual([]);
  });

  it('returns results sorted by similarity descending', async () => {
    const { log } = makeLog();
    log.append(makeEntry({ test: 'rate-limiter-test', diagnosis: 'rate limiter resets on restart' }));
    log.append(makeEntry({ test: 'unrelated-test', diagnosis: 'database connection pool exhausted' }));
    const results = await log.querySimilar('rate limiter restart');
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.similarity).toBeGreaterThanOrEqual(results[results.length - 1]!.similarity);
  });

  it('respects topK limit', async () => {
    const { log } = makeLog();
    for (let i = 0; i < 10; i++) {
      log.append(makeEntry({ test: `test-${i}`, diagnosis: `diagnosis ${i}` }));
    }
    const results = await log.querySimilar('diagnosis', 3);
    expect(results.length).toBeLessThanOrEqual(3);
  });
});

// ── GetSuccessfulRepairs ──────────────────────────────────────────────────────
describe('CompoundingLog.getSuccessfulRepairs', () => {
  it('returns only PASS + APPROVED entries', () => {
    const { log } = makeLog();
    log.append(makeEntry({ test_result: 'PASS', meta_review: 'APPROVED' }));
    log.append(makeEntry({ test_result: 'FAIL', meta_review: 'REJECTED' }));
    log.append(makeEntry({ test_result: 'PASS', meta_review: 'PENDING' }));
    const repairs = log.getSuccessfulRepairs();
    expect(repairs).toHaveLength(1);
    expect(repairs[0]!.test_result).toBe('PASS');
    expect(repairs[0]!.meta_review).toBe('APPROVED');
  });

  it('filters by trigger type when provided', () => {
    const { log } = makeLog();
    log.append(makeEntry({ trigger: 'test_failure', test_result: 'PASS', meta_review: 'APPROVED' }));
    log.append(makeEntry({ trigger: 'verdict_event', test_result: 'PASS', meta_review: 'APPROVED' }));
    const repairs = log.getSuccessfulRepairs('test_failure');
    expect(repairs).toHaveLength(1);
    expect(repairs[0]!.trigger).toBe('test_failure');
  });
});

// ── GetFailedRepairs ──────────────────────────────────────────────────────────
describe('CompoundingLog.getFailedRepairs', () => {
  it('returns only FAIL entries', () => {
    const { log } = makeLog();
    log.append(makeEntry({ test_result: 'PASS' }));
    log.append(makeEntry({ test_result: 'FAIL' }));
    const failures = log.getFailedRepairs();
    expect(failures).toHaveLength(1);
    expect(failures[0]!.test_result).toBe('FAIL');
  });
});

// ── ScanContradictions ────────────────────────────────────────────────────────
describe('CompoundingLog.scanContradictions', () => {
  it('returns empty array when no contradictions exist', () => {
    const { log } = makeLog();
    log.append(makeEntry({ test: 'stable-test', test_result: 'PASS' }));
    log.append(makeEntry({ test: 'stable-test', test_result: 'PASS' }));
    expect(log.scanContradictions()).toEqual([]);
  });

  it('detects a test with both PASS and FAIL results', () => {
    const { log } = makeLog();
    log.append(makeEntry({ test: 'flaky-test', test_result: 'PASS' }));
    log.append(makeEntry({ test: 'flaky-test', test_result: 'FAIL' }));
    log.append(makeEntry({ test: 'flaky-test', test_result: 'PASS' }));
    const contradictions = log.scanContradictions();
    expect(contradictions).toHaveLength(1);
    expect(contradictions[0]!.test).toBe('flaky-test');
    expect(contradictions[0]!.passCount).toBe(2);
    expect(contradictions[0]!.failCount).toBe(1);
  });
});

// ── GetStats ──────────────────────────────────────────────────────────────────
describe('CompoundingLog.getStats', () => {
  it('returns zero stats for empty log', () => {
    const { log } = makeLog();
    const stats = log.getStats();
    expect(stats.total).toBe(0);
    expect(stats.passRate).toBe(0);
    expect(stats.avgSlmConfidence).toBe(0);
    expect(stats.pendingReviews).toBe(0);
  });

  it('calculates correct pass rate and average confidence', () => {
    const { log } = makeLog();
    log.append(makeEntry({ test_result: 'PASS', slm_confidence: 0.9, meta_review: 'APPROVED' }));
    log.append(makeEntry({ test_result: 'PASS', slm_confidence: 0.8, meta_review: 'APPROVED' }));
    log.append(makeEntry({ test_result: 'FAIL', slm_confidence: 0.5, meta_review: 'PENDING' }));
    const stats = log.getStats();
    expect(stats.total).toBe(3);
    expect(stats.passRate).toBeCloseTo(2 / 3, 3);
    expect(stats.avgSlmConfidence).toBeCloseTo(0.733, 2);
    expect(stats.pendingReviews).toBe(1);
  });
});
