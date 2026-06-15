import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { SIADatasetGenerator } from '../../src/training/siaDatasetGenerator.js';
import type { SIAPublicRecord, SIAGroundTruthRecord } from '../../src/training/siaDatasetGenerator.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function readJsonl<T>(filePath: string): T[] {
  return fs.readFileSync(filePath, 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line) as T);
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('SIADatasetGenerator', () => {
  let tmpDir: string;
  let generator: SIADatasetGenerator;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sia-test-'));
    generator = new SIADatasetGenerator({ publicCount: 20, privateCount: 10, seed: 42 });
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('generates the expected number of public and private records', () => {
    const { publicPath, privatePath } = generator.generate(tmpDir);
    const publicRecords = readJsonl<SIAPublicRecord>(publicPath);
    const privateRecords = readJsonl<SIAGroundTruthRecord>(privatePath);
    expect(publicRecords).toHaveLength(20);
    expect(privateRecords).toHaveLength(10);
  });

  it('public records have the required schema fields', () => {
    const { publicPath } = generator.generate(tmpDir);
    const records = readJsonl<SIAPublicRecord>(publicPath);
    for (const r of records) {
      expect(r).toHaveProperty('claim_id');
      expect(r).toHaveProperty('claim_text');
      expect(r).toHaveProperty('source_title');
      expect(r).toHaveProperty('source_abstract');
      expect(r).toHaveProperty('domain');
      expect(r.claim_id).toMatch(/^syn-\d{5}$/);
      expect(r.claim_text.length).toBeGreaterThan(10);
    }
  });

  it('ground truth records have the required schema fields', () => {
    const { privatePath } = generator.generate(tmpDir);
    const records = readJsonl<SIAGroundTruthRecord>(privatePath);
    const validStates = ['verified', 'contested', 'implied', 'beyond_evidence'];
    for (const r of records) {
      expect(r).toHaveProperty('claim_id');
      expect(r).toHaveProperty('citation_state');
      expect(r).toHaveProperty('confidence');
      expect(validStates).toContain(r.citation_state);
      expect(r.confidence).toBeGreaterThanOrEqual(0);
      expect(r.confidence).toBeLessThanOrEqual(1);
    }
  });

  it('output is deterministic for the same seed', () => {
    const g1 = new SIADatasetGenerator({ publicCount: 5, privateCount: 5, seed: 99 });
    const g2 = new SIADatasetGenerator({ publicCount: 5, privateCount: 5, seed: 99 });
    const dir1 = fs.mkdtempSync(path.join(os.tmpdir(), 'sia-det1-'));
    const dir2 = fs.mkdtempSync(path.join(os.tmpdir(), 'sia-det2-'));
    try {
      const { publicPath: p1 } = g1.generate(dir1);
      const { publicPath: p2 } = g2.generate(dir2);
      const r1 = readJsonl<SIAPublicRecord>(p1);
      const r2 = readJsonl<SIAPublicRecord>(p2);
      expect(r1).toEqual(r2);
    } finally {
      fs.rmSync(dir1, { recursive: true, force: true });
      fs.rmSync(dir2, { recursive: true, force: true });
    }
  });

  it('different seeds produce different outputs', () => {
    const g1 = new SIADatasetGenerator({ publicCount: 5, privateCount: 5, seed: 1 });
    const g2 = new SIADatasetGenerator({ publicCount: 5, privateCount: 5, seed: 2 });
    const dir1 = fs.mkdtempSync(path.join(os.tmpdir(), 'sia-seed1-'));
    const dir2 = fs.mkdtempSync(path.join(os.tmpdir(), 'sia-seed2-'));
    try {
      const { privatePath: p1 } = g1.generate(dir1);
      const { privatePath: p2 } = g2.generate(dir2);
      const r1 = readJsonl<SIAGroundTruthRecord>(p1);
      const r2 = readJsonl<SIAGroundTruthRecord>(p2);
      // At least some records should differ
      const allSame = r1.every((r, i) => r.citation_state === r2[i]?.citation_state);
      expect(allSame).toBe(false);
    } finally {
      fs.rmSync(dir1, { recursive: true, force: true });
      fs.rmSync(dir2, { recursive: true, force: true });
    }
  });

  it('distribution() returns counts for all four citation states', () => {
    const gen = new SIADatasetGenerator({ publicCount: 20, privateCount: 100, seed: 42 });
    const dist = gen.distribution();
    const validStates = ['verified', 'contested', 'implied', 'beyond_evidence'];
    const total = Object.values(dist).reduce((a, b) => a + b, 0);
    // All states should appear
    for (const state of validStates) {
      expect(dist[state]).toBeGreaterThan(0);
    }
    // Total should equal privateCount
    expect(total).toBe(100);
  });

  it('creates output directories if they do not exist', () => {
    const nestedDir = path.join(tmpDir, 'deeply', 'nested', 'output');
    expect(fs.existsSync(nestedDir)).toBe(false);
    generator.generate(nestedDir);
    expect(fs.existsSync(path.join(nestedDir, 'data', 'public'))).toBe(true);
    expect(fs.existsSync(path.join(nestedDir, 'data', 'private'))).toBe(true);
  });

  it('public and private claim_ids do not overlap', () => {
    const { publicPath, privatePath } = generator.generate(tmpDir);
    const publicIds = new Set(readJsonl<SIAPublicRecord>(publicPath).map(r => r.claim_id));
    const privateIds = readJsonl<SIAGroundTruthRecord>(privatePath).map(r => r.claim_id);
    for (const id of privateIds) {
      expect(publicIds.has(id)).toBe(false);
    }
  });
});
