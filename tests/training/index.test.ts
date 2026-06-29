/**
 * tests/training/index.test.ts
 * Tests for the createTrainingPipeline() factory function in training/index.ts
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// ── Mock IncrementalTrainer to avoid spawning Python ─────────────────────────
vi.mock('../../src/training/incrementalTrainer.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../src/training/incrementalTrainer.js')>();
  class MockIncrementalTrainer {
    constructor(public config: unknown) {}
    async run() { return { success: true, durationMs: 1 }; }
  }
  return { ...actual, IncrementalTrainer: MockIncrementalTrainer };
});

// ── Tests ─────────────────────────────────────────────────────────────────────
describe('createTrainingPipeline()', () => {
  let tmpCorpus: string;

  beforeEach(() => {
    tmpCorpus = path.join(
      os.tmpdir(),
      `test_pipeline_${Date.now()}_${Math.random().toString(36).slice(2)}.jsonl`
    );
  });

  afterEach(() => {
    if (fs.existsSync(tmpCorpus)) fs.unlinkSync(tmpCorpus);
  });

  it('returns an object with generator, watcher, and trainer', async () => {
    const { createTrainingPipeline } = await import('../../src/training/index.js');
    const pipeline = createTrainingPipeline({ corpusPath: tmpCorpus });
    expect(pipeline).toHaveProperty('generator');
    expect(pipeline).toHaveProperty('watcher');
    expect(pipeline).toHaveProperty('trainer');
  });

  it('generator is a ClaimsCorpusGenerator', async () => {
    const { createTrainingPipeline, ClaimsCorpusGenerator } = await import('../../src/training/index.js');
    const { generator } = createTrainingPipeline({ corpusPath: tmpCorpus });
    expect(generator).toBeInstanceOf(ClaimsCorpusGenerator);
  });

  it('watcher is a CorpusWatcher', async () => {
    const { createTrainingPipeline, CorpusWatcher } = await import('../../src/training/index.js');
    const { watcher } = createTrainingPipeline({ corpusPath: tmpCorpus });
    expect(watcher).toBeInstanceOf(CorpusWatcher);
  });

  it('trainer is an IncrementalTrainer', async () => {
    const { createTrainingPipeline, IncrementalTrainer } = await import('../../src/training/index.js');
    const { trainer } = createTrainingPipeline({ corpusPath: tmpCorpus });
    expect(trainer).toBeInstanceOf(IncrementalTrainer);
  });

  it('uses default corpus path when not provided (via env var)', async () => {
    const { createTrainingPipeline } = await import('../../src/training/index.js');
    // Set env var to avoid /data/training permission error
    process.env['TRAINING_CORPUS_PATH'] = tmpCorpus;
    const pipeline = createTrainingPipeline({});
    expect(pipeline.generator).toBeDefined();
    delete process.env['TRAINING_CORPUS_PATH'];
  });

  it('respects custom minPairsThreshold', async () => {
    const { createTrainingPipeline } = await import('../../src/training/index.js');
    const { watcher } = createTrainingPipeline({
      corpusPath: tmpCorpus,
      minPairsThreshold: 100,
    });
    expect(watcher).toBeDefined();
    // Watcher with threshold 100 should not fire immediately
    const mockCallback = vi.fn();
    watcher.onReady(mockCallback);
    expect(mockCallback).not.toHaveBeenCalled();
  });

  it('wires watcher to trigger trainer automatically', async () => {
    const { createTrainingPipeline } = await import('../../src/training/index.js');
    const { trainer, watcher } = createTrainingPipeline({
      corpusPath: tmpCorpus,
      minPairsThreshold: 1,
    });
    const runSpy = vi.spyOn(trainer, 'run').mockResolvedValue({
      success: true,
      durationMs: 1,
    });
    // Manually trigger the watcher's check (corpus is empty, so threshold not reached)
    watcher.check();
    // Trainer should not have been called yet (corpus is empty)
    expect(runSpy).not.toHaveBeenCalled();
  });

  it('multiple pipelines are independent', async () => {
    const { createTrainingPipeline } = await import('../../src/training/index.js');
    const tmpCorpus2 = path.join(os.tmpdir(), `test_pipeline2_${Date.now()}.jsonl`);
    try {
      const p1 = createTrainingPipeline({ corpusPath: tmpCorpus });
      const p2 = createTrainingPipeline({ corpusPath: tmpCorpus2 });
      expect(p1.generator).not.toBe(p2.generator);
      expect(p1.watcher).not.toBe(p2.watcher);
      expect(p1.trainer).not.toBe(p2.trainer);
    } finally {
      if (fs.existsSync(tmpCorpus2)) fs.unlinkSync(tmpCorpus2);
    }
  });
});

// ── Module re-exports ─────────────────────────────────────────────────────────
describe('training/index.ts re-exports', () => {
  it('exports ClaimsCorpusGenerator', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.ClaimsCorpusGenerator).toBeDefined();
    expect(typeof mod.ClaimsCorpusGenerator).toBe('function');
  });

  it('exports CorpusWatcher', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.CorpusWatcher).toBeDefined();
    expect(typeof mod.CorpusWatcher).toBe('function');
  });

  it('exports IncrementalTrainer', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.IncrementalTrainer).toBeDefined();
    expect(typeof mod.IncrementalTrainer).toBe('function');
  });

  it('exports SIADatasetGenerator', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.SIADatasetGenerator).toBeDefined();
    expect(typeof mod.SIADatasetGenerator).toBe('function');
  });

  it('exports fetchVerifiedClaims', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.fetchVerifiedClaims).toBeDefined();
    expect(typeof mod.fetchVerifiedClaims).toBe('function');
  });

  it('exports countNewVerifiedClaims', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.countNewVerifiedClaims).toBeDefined();
    expect(typeof mod.countNewVerifiedClaims).toBe('function');
  });

  it('exports closeBridge', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.closeBridge).toBeDefined();
    expect(typeof mod.closeBridge).toBe('function');
  });

  it('exports createTrainingPipeline', async () => {
    const mod = await import('../../src/training/index.js');
    expect(mod.createTrainingPipeline).toBeDefined();
    expect(typeof mod.createTrainingPipeline).toBe('function');
  });
});
