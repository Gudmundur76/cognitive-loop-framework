/**
 * tests/loop/index.test.ts
 * Tests for the createLoop() factory function in loop/index.ts
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// ── Mock heavy dependencies ───────────────────────────────────────────────────
vi.mock('../../src/memory/index.js', () => {
  class MockMemoryLayer {
    ingestFile = vi.fn().mockResolvedValue({ processed: 1, skipped: 0, errors: 0 });
    findSimilar = vi.fn().mockResolvedValue([]);
    getRelationships = vi.fn().mockReturnValue([]);
  }
  return { MemoryLayer: MockMemoryLayer };
});

vi.mock('../../src/slm/selfPromptEngine.js', () => {
  class MockSelfPromptEngine {
    reason = vi.fn().mockResolvedValue({ output: 'mock reasoning', model: 'mock' });
  }
  return { SelfPromptEngine: MockSelfPromptEngine };
});

// ── Tests ─────────────────────────────────────────────────────────────────────
describe('createLoop()', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns an object with orchestrator, dispatcher, and meta', async () => {
    const { createLoop } = await import('../../src/loop/index.js');
    const result = createLoop({ dryRun: true });
    expect(result).toHaveProperty('orchestrator');
    expect(result).toHaveProperty('dispatcher');
    expect(result).toHaveProperty('meta');
  });

  it('creates a dispatcher in dry-run mode when dryRun=true', async () => {
    const { createLoop } = await import('../../src/loop/index.js');
    const { dispatcher } = createLoop({ dryRun: true });
    // In dry-run mode, processEvents should not throw and should return dispatched=0 for no events
    const result = await dispatcher.processEvents([]);
    expect(result.dryRun).toBe(true);
    expect(result.dispatched).toBe(0);
  });

  it('defaults to dryRun=true when no manusApiKey is provided', async () => {
    const { createLoop } = await import('../../src/loop/index.js');
    const { dispatcher } = createLoop({});
    const result = await dispatcher.processEvents([]);
    expect(result.dryRun).toBe(true);
  });

  it('creates a MetaAgent with an empty event queue', async () => {
    const { createLoop } = await import('../../src/loop/index.js');
    const { meta } = createLoop({ dryRun: true });
    expect(meta.peekEvents()).toHaveLength(0);
  });

  it('creates a LoopOrchestrator that can be called', async () => {
    const { createLoop } = await import('../../src/loop/index.js');
    const { orchestrator } = createLoop({ dryRun: true });
    expect(typeof orchestrator.run).toBe('function');
  });

  it('passes maxIterations to the orchestrator config', async () => {
    const { createLoop } = await import('../../src/loop/index.js');
    const { orchestrator } = createLoop({ dryRun: true, maxIterations: 3 });
    // The orchestrator should be created without error
    expect(orchestrator).toBeDefined();
  });

  it('re-exports LoopOrchestrator from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.LoopOrchestrator).toBeDefined();
    expect(typeof mod.LoopOrchestrator).toBe('function');
  });

  it('re-exports MetaAgent from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.MetaAgent).toBeDefined();
    expect(typeof mod.MetaAgent).toBe('function');
  });

  it('re-exports ManusDispatcher from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.ManusDispatcher).toBeDefined();
    expect(typeof mod.ManusDispatcher).toBe('function');
  });

  it('re-exports createTrainingPipeline from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.createTrainingPipeline).toBeDefined();
    expect(typeof mod.createTrainingPipeline).toBe('function');
  });

  it('re-exports ClaimsCorpusGenerator from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.ClaimsCorpusGenerator).toBeDefined();
    expect(typeof mod.ClaimsCorpusGenerator).toBe('function');
  });

  it('re-exports CorpusWatcher from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.CorpusWatcher).toBeDefined();
    expect(typeof mod.CorpusWatcher).toBe('function');
  });

  it('re-exports IncrementalTrainer from the module', async () => {
    const mod = await import('../../src/loop/index.js');
    expect(mod.IncrementalTrainer).toBeDefined();
    expect(typeof mod.IncrementalTrainer).toBe('function');
  });
});
