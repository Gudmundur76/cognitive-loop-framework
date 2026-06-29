/**
 * tests/training/ttruthdeskBridge.test.ts
 * Tests for fetchVerifiedClaims(), countNewVerifiedClaims(), closeBridge()
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  fetchVerifiedClaims,
  countNewVerifiedClaims,
  closeBridge,
} from '../../src/training/ttruthdeskBridge.js';

// ── Mock mysql2/promise ───────────────────────────────────────────────────────
// Use vi.hoisted to ensure the mock functions are available before vi.mock is hoisted
const { mockExecute, mockEnd } = vi.hoisted(() => {
  const mockExecute = vi.fn();
  const mockEnd = vi.fn().mockResolvedValue(undefined);
  return { mockExecute, mockEnd };
});

vi.mock('mysql2/promise', () => {
  const mockPool = { execute: mockExecute, end: mockEnd };
  return {
    default: { createPool: vi.fn().mockReturnValue(mockPool) },
    createPool: vi.fn().mockReturnValue(mockPool),
  };
});

// ── Helpers ───────────────────────────────────────────────────────────────────
const SINCE = new Date('2024-01-01');

function makeRow(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    claimText: 'Darunavir IC50 0.003 nM',
    verdict: 'Supported',
    verdictRationale: 'Confirmed by assay',
    confidenceScore: 0.95,
    compositeTruthLabel: 'verified_faithful',
    verticalDomain: 'pharmacology',
    createdAt: new Date('2024-01-15'),
    evidenceUrl: 'https://www.rcsb.org/structure/1T3R',
    ...overrides,
  };
}

// ── fetchVerifiedClaims tests ─────────────────────────────────────────────────
describe('fetchVerifiedClaims()', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockEnd.mockResolvedValue(undefined);
    process.env['DATABASE_URL'] = 'mysql://localhost/test';
    await closeBridge(); // reset pool singleton
  });

  afterEach(async () => {
    await closeBridge();
    delete process.env['DATABASE_URL'];
  });

  it('returns empty array when DATABASE_URL is not set', async () => {
    delete process.env['DATABASE_URL'];
    await closeBridge();
    const result = await fetchVerifiedClaims(SINCE);
    expect(result).toEqual([]);
  });

  it('returns mapped ClaimRecord array on success', async () => {
    mockExecute.mockResolvedValueOnce([[makeRow()]]);
    const result = await fetchVerifiedClaims(SINCE);
    expect(result).toHaveLength(1);
    expect(result[0]!.id).toBe(1);
    expect(result[0]!.claimText).toBe('Darunavir IC50 0.003 nM');
    expect(result[0]!.verdict).toBe('Supported');
    expect(result[0]!.confidenceScore).toBe(0.95);
    expect(result[0]!.evidenceUrl).toBe('https://www.rcsb.org/structure/1T3R');
  });

  it('maps null fields correctly', async () => {
    mockExecute.mockResolvedValueOnce([[makeRow({
      verdictRationale: null,
      confidenceScore: null,
      compositeTruthLabel: null,
      verticalDomain: null,
      evidenceUrl: null,
    })]]);
    const result = await fetchVerifiedClaims(SINCE);
    expect(result[0]!.verdictRationale).toBeNull();
    expect(result[0]!.confidenceScore).toBeNull();
    expect(result[0]!.compositeTruthLabel).toBeNull();
    expect(result[0]!.verticalDomain).toBeNull();
    expect(result[0]!.evidenceUrl).toBeNull();
  });

  it('returns empty array when DB throws (non-fatal)', async () => {
    mockExecute.mockRejectedValueOnce(new Error('Connection refused'));
    const result = await fetchVerifiedClaims(SINCE);
    expect(result).toEqual([]);
  });

  it('passes the since date as a query parameter', async () => {
    mockExecute.mockResolvedValueOnce([[]]);
    const since = new Date('2024-06-01');
    await fetchVerifiedClaims(since);
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('WHERE'),
      [since]
    );
  });

  it('maps createdAt string to Date object', async () => {
    mockExecute.mockResolvedValueOnce([[makeRow({
      createdAt: '2024-03-15T10:00:00.000Z',
    })]]);
    const result = await fetchVerifiedClaims(SINCE);
    expect(result[0]!.createdAt).toBeInstanceOf(Date);
  });

  it('returns multiple rows correctly', async () => {
    const rows = Array.from({ length: 5 }, (_, i) =>
      makeRow({ id: i + 1, claimText: `Claim ${i}` })
    );
    mockExecute.mockResolvedValueOnce([rows]);
    const result = await fetchVerifiedClaims(SINCE);
    expect(result).toHaveLength(5);
  });
});

// ── countNewVerifiedClaims tests ──────────────────────────────────────────────
describe('countNewVerifiedClaims()', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockEnd.mockResolvedValue(undefined);
    process.env['DATABASE_URL'] = 'mysql://localhost/test';
    await closeBridge();
  });

  afterEach(async () => {
    await closeBridge();
    delete process.env['DATABASE_URL'];
  });

  it('returns 0 when DATABASE_URL is not set', async () => {
    delete process.env['DATABASE_URL'];
    await closeBridge();
    const result = await countNewVerifiedClaims(SINCE);
    expect(result).toBe(0);
  });

  it('returns the count from the DB', async () => {
    mockExecute.mockResolvedValueOnce([[{ cnt: 42 }]]);
    const result = await countNewVerifiedClaims(SINCE);
    expect(result).toBe(42);
  });

  it('returns 0 when DB throws', async () => {
    mockExecute.mockRejectedValueOnce(new Error('DB error'));
    const result = await countNewVerifiedClaims(SINCE);
    expect(result).toBe(0);
  });

  it('returns 0 when cnt is null', async () => {
    mockExecute.mockResolvedValueOnce([[{ cnt: null }]]);
    const result = await countNewVerifiedClaims(SINCE);
    expect(result).toBe(0);
  });

  it('passes the since date as a query parameter', async () => {
    mockExecute.mockResolvedValueOnce([[{ cnt: 10 }]]);
    const since = new Date('2024-05-01');
    await countNewVerifiedClaims(since);
    expect(mockExecute).toHaveBeenCalledWith(
      expect.stringContaining('COUNT(*)'),
      [since]
    );
  });
});

// ── closeBridge tests ─────────────────────────────────────────────────────────
describe('closeBridge()', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    mockEnd.mockResolvedValue(undefined);
    delete process.env['DATABASE_URL'];
    await closeBridge();
  });

  it('resolves without error when pool is not initialized', async () => {
    await expect(closeBridge()).resolves.toBeUndefined();
  });

  it('calls pool.end() when pool is initialized', async () => {
    process.env['DATABASE_URL'] = 'mysql://localhost/test';
    mockExecute.mockResolvedValueOnce([[{ cnt: 0 }]]);
    await countNewVerifiedClaims(new Date());
    await closeBridge();
    expect(mockEnd).toHaveBeenCalledTimes(1);
  });

  it('can be called multiple times without error', async () => {
    await expect(Promise.all([closeBridge(), closeBridge()])).resolves.toBeDefined();
  });
});
