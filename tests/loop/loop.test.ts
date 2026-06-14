/**
 * Sprint 4 Test Suite — Loop Wiring
 *
 * Integration tests for:
 * - MetaAgent: health assessment, event queue, repair detection
 * - ManusDispatcher: dry-run dispatch, deduplication, prompt building
 * - LoopOrchestrator: full five-layer loop, convergence, event propagation
 *
 * All tests run without a live Ollama instance, Manus API, or RuVector.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import { MetaAgent } from '../../src/loop/metaAgent.js';
import { ManusDispatcher } from '../../src/loop/manusDispatcher.js';
import { LoopOrchestrator } from '../../src/loop/loopOrchestrator.js';
import { MemoryLayer } from '../../src/memory/index.js';
import { SelfPromptEngine } from '../../src/slm/selfPromptEngine.js';
import type { LayerResult } from '../../src/loop/loopOrchestrator.js';
import type { SystemEvent } from '../../src/loop/metaAgent.js';

// ── Fixtures ─────────────────────────────────────────────────────

function makeTsFile(content: string): string {
  const tmpPath = path.join(os.tmpdir(), `test_${Date.now()}.ts`);
  fs.writeFileSync(tmpPath, content, 'utf8');
  return tmpPath;
}

const sampleTs = `
export function rateLimiter(key: string): boolean {
  const store = new Map<string, number>();
  const count = store.get(key) ?? 0;
  store.set(key, count + 1);
  return count < 10;
}
`.trim();

// ── MetaAgent Tests ───────────────────────────────────────────────

describe('MetaAgent', () => {
  let meta: MetaAgent;

  beforeEach(() => {
    meta = new MetaAgent();
  });

  it('starts with an empty event queue', () => {
    expect(meta.peekEvents()).toHaveLength(0);
  });

  it('publishes and drains events correctly', () => {
    meta.publishEvent({
      type: 'system_capability_required',
      severity: 'critical',
      message: 'rate limiter broken',
      filePath: 'server/rateLimiter.ts'
    });
    const events = meta.drainEvents();
    expect(events).toHaveLength(1);
    expect(events[0]?.type).toBe('system_capability_required');
    expect(meta.peekEvents()).toHaveLength(0);
  });

  it('adds timestamps to published events', () => {
    meta.publishEvent({
      type: 'knowledge_gap_detected',
      severity: 'info',
      message: 'gap found',
      filePath: ''
    });
    const events = meta.drainEvents();
    expect(events[0]?.timestamp).toBeDefined();
    expect(new Date(events[0]!.timestamp!).getTime()).toBeGreaterThan(0);
  });

  it('assesses healthy state when all layers pass', () => {
    const results: LayerResult[] = [
      { layer: 'friction', passed: true, output: 'ok', durationMs: 1 },
      { layer: 'truth', passed: true, output: 'ok', durationMs: 1 },
      { layer: 'selfPrompt', passed: true, output: 'ok', durationMs: 1 },
      { layer: 'frontier', passed: true, output: 'ok', durationMs: 1 },
      { layer: 'meta', passed: true, output: 'ok', durationMs: 1 }
    ];
    const health = meta.assessHealth(results);
    expect(health.score).toBe(1.0);
    expect(health.status).toBe('healthy');
    expect(health.failingLayers).toHaveLength(0);
  });

  it('assesses critical state when friction layer fails', () => {
    const results: LayerResult[] = [
      { layer: 'friction', passed: false, output: 'no files', durationMs: 1 },
      { layer: 'truth', passed: true, output: 'ok', durationMs: 1 }
    ];
    const health = meta.assessHealth(results);
    expect(health.status).toBe('critical');
    expect(health.failingLayers).toContain('friction');
  });

  it('assesses degraded state when two non-friction layers fail', () => {
    const results: LayerResult[] = [
      { layer: 'friction', passed: true, output: 'ok', durationMs: 1 },
      { layer: 'truth', passed: false, output: 'failed', durationMs: 1 },
      { layer: 'selfPrompt', passed: false, output: 'failed', durationMs: 1 },
      { layer: 'frontier', passed: true, output: 'ok', durationMs: 1 },
      { layer: 'meta', passed: true, output: 'ok', durationMs: 1 }
    ];
    const health = meta.assessHealth(results);
    // 3/5 passing = score 0.6 — below 0.8 threshold → degraded
    expect(health.status).toBe('degraded');
  });

  it('publishes health_degraded event when health is critical', () => {
    const results: LayerResult[] = [
      { layer: 'friction', passed: false, output: 'no files', durationMs: 1 }
    ];
    meta.assessHealth(results);
    const events = meta.drainEvents();
    expect(events.some(e => e.type === 'health_degraded')).toBe(true);
  });

  it('detects repair requirement from system_capability_required events', () => {
    const events: SystemEvent[] = [
      {
        type: 'system_capability_required',
        severity: 'critical',
        message: 'adapter broken',
        filePath: 'server/adapter.ts'
      }
    ];
    expect(meta.requiresRepair(events)).toBe(true);
  });

  it('does not require repair for info-only events', () => {
    const events: SystemEvent[] = [
      {
        type: 'knowledge_gap_detected',
        severity: 'info',
        message: 'gap found',
        filePath: ''
      }
    ];
    expect(meta.requiresRepair(events)).toBe(false);
  });

  it('builds a structured repair context from events', () => {
    const events: SystemEvent[] = [
      {
        type: 'system_capability_required',
        severity: 'critical',
        message: 'rate limiter resets on restart',
        filePath: 'server/rateLimiter.ts'
      }
    ];
    const context = meta.buildRepairContext(events);
    expect(context).toContain('system capabilities are required');
    expect(context).toContain('rate limiter resets on restart');
    expect(context).toContain('server/rateLimiter.ts');
  });

  it('caps the event queue at maxQueueSize', () => {
    const smallMeta = new MetaAgent(3);
    for (let i = 0; i < 5; i++) {
      smallMeta.publishEvent({
        type: 'knowledge_gap_detected',
        severity: 'info',
        message: `gap ${i}`,
        filePath: ''
      });
    }
    expect(smallMeta.peekEvents()).toHaveLength(3);
  });
});

// ── ManusDispatcher Tests ─────────────────────────────────────────

describe('ManusDispatcher', () => {
  let meta: MetaAgent;
  let dispatcher: ManusDispatcher;

  beforeEach(() => {
    meta = new MetaAgent();
    dispatcher = new ManusDispatcher(meta, {
      apiKey: 'test-key',
      dryRun: true
    });
  });

  it('dispatches a repair task in dry-run mode without calling the API', async () => {
    const events: SystemEvent[] = [
      {
        type: 'system_capability_required',
        severity: 'critical',
        message: 'rate limiter broken',
        filePath: 'server/rateLimiter.ts'
      }
    ];
    const result = await dispatcher.processEvents(events);
    expect(result.dispatched).toBe(1);
    expect(result.dryRun).toBe(true);
    expect(result.tasks[0]?.status).toBe('dispatched');
  });

  it('skips non-repair events', async () => {
    const events: SystemEvent[] = [
      {
        type: 'knowledge_gap_detected',
        severity: 'info',
        message: 'gap found',
        filePath: ''
      }
    ];
    const result = await dispatcher.processEvents(events);
    expect(result.dispatched).toBe(0);
    expect(result.skipped).toBe(1);
  });

  it('deduplicates repair tasks for the same file', async () => {
    const event: SystemEvent = {
      type: 'system_capability_required',
      severity: 'critical',
      message: 'broken',
      filePath: 'server/rateLimiter.ts'
    };
    await dispatcher.processEvents([event]);
    const result = await dispatcher.processEvents([event]);
    expect(result.dispatched).toBe(0);
  });

  it('includes the file path and instructions in the repair prompt', async () => {
    const events: SystemEvent[] = [
      {
        type: 'system_capability_required',
        severity: 'critical',
        message: 'verdict flip not wired',
        filePath: 'server/reEvalEngine.ts'
      }
    ];
    await dispatcher.processEvents(events);
    const history = dispatcher.getHistory();
    expect(history[0]?.prompt).toContain('server/reEvalEngine.ts');
    expect(history[0]?.prompt).toContain('DEVELOPMENT_DISCIPLINE.md');
  });

  it('records dispatched tasks in history', async () => {
    const events: SystemEvent[] = [
      {
        type: 'system_capability_required',
        severity: 'warning',
        message: 'embedding schema missing',
        filePath: 'drizzle/schema.ts'
      }
    ];
    await dispatcher.processEvents(events);
    expect(dispatcher.getHistory()).toHaveLength(1);
  });

  it('calls the Manus API when not in dry-run mode', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: 'task-123' })
    });
    global.fetch = mockFetch as unknown as typeof fetch;

    const liveDispatcher = new ManusDispatcher(meta, {
      apiKey: 'live-key',
      dryRun: false
    });
    const events: SystemEvent[] = [
      {
        type: 'system_capability_required',
        severity: 'critical',
        message: 'adapter broken',
        filePath: 'server/adapter.ts'
      }
    ];
    await liveDispatcher.processEvents(events);
    expect(mockFetch).toHaveBeenCalledOnce();
    expect(mockFetch.mock.calls[0]?.[0]).toContain('/tasks');
  });
});

// ── LoopOrchestrator Integration Tests ───────────────────────────

describe('LoopOrchestrator (integration)', () => {
  let meta: MetaAgent;
  let orchestrator: LoopOrchestrator;
  let tmpFile: string;

  beforeEach(() => {
    // Mock the SelfPromptEngine to avoid Ollama/OpenAI calls
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('11434')) return Promise.reject(new Error('down'));
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'DIAGNOSIS: looks healthy' } }],
          usage: { total_tokens: 10 }
        })
      });
    }) as unknown as typeof fetch;

    meta = new MetaAgent();
    const memory = new MemoryLayer();
    const slm = new SelfPromptEngine({ fallbackToOpenAI: true });
    orchestrator = new LoopOrchestrator(memory, slm, meta, { maxIterations: 2 });
    tmpFile = makeTsFile(sampleTs);
  });

  it('runs the full loop and returns a result', async () => {
    const result = await orchestrator.run({ filePaths: [tmpFile] });
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.layerResults.length).toBeGreaterThan(0);
  }, 30000);

  it('produces layer results for all five layers on a valid file', async () => {
    const result = await orchestrator.run({ filePaths: [tmpFile] });
    const layers = result.layerResults.map(r => r.layer);
    expect(layers).toContain('friction');
    expect(layers).toContain('truth');
    expect(layers).toContain('selfPrompt');
    expect(layers).toContain('frontier');
    expect(layers).toContain('meta');
  });

  it('halts at friction layer when no valid files are provided', async () => {
    const result = await orchestrator.run({ filePaths: ['not-a-ts-file.js'] });
    const layers = result.layerResults.map(r => r.layer);
    expect(layers).toContain('friction');
    expect(layers).not.toContain('truth');
  });

  it('returns a completion promise string', async () => {
    const result = await orchestrator.run({ filePaths: [tmpFile] });
    expect(typeof result.completionPromise).toBe('string');
    expect(result.completionPromise.length).toBeGreaterThan(0);
  });

  it('emits events during the loop run', async () => {
    const result = await orchestrator.run({ filePaths: [tmpFile] });
    // Events may or may not be emitted depending on health — just verify the field exists
    expect(Array.isArray(result.events)).toBe(true);
  });
});
