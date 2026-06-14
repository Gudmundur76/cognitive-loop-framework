/**
 * Sprint 3 Test Suite — SLM Layer
 *
 * Tests the CorpusGenerator and SelfPromptEngine.
 * All tests run without a live Ollama instance or GPU.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { CorpusGenerator, TrainingPair } from '../../src/slm/corpusGenerator.js';
import { SelfPromptEngine } from '../../src/slm/selfPromptEngine.js';
import type { CodeNode, CodeEdge } from '../../src/indexer/extractor.js';

// ── Fixtures ─────────────────────────────────────────────────────

const sampleNodes: CodeNode[] = [
  {
    id: 'server/rateLimiter.ts:checkLimit',
    name: 'checkLimit',
    type: 'function',
    filePath: 'server/rateLimiter.ts',
    startLine: 12,
    endLine: 28,
    code: `export function checkLimit(key: string, max: number): boolean {
  const bucket = store.get(key);
  if (!bucket) { store.set(key, { count: 1, reset: Date.now() + 60000 }); return true; }
  if (bucket.count >= max) return false;
  bucket.count++;
  return true;
}`
  },
  {
    id: 'server/dreamEngine.ts:runDreamSession',
    name: 'runDreamSession',
    type: 'function',
    filePath: 'server/dreamEngine.ts',
    startLine: 45,
    endLine: 72,
    code: `export async function runDreamSession(context: string): Promise<string> {
  const hypothesis = await llm.complete(context);
  // TODO: add confidence gate
  await ingestPipeline.push(hypothesis);
  return hypothesis;
}`
  },
  {
    id: 'server/reEvalEngine.ts:ReEvalEngine',
    name: 'ReEvalEngine',
    type: 'class',
    filePath: 'server/reEvalEngine.ts',
    startLine: 1,
    endLine: 120,
    code: `export class ReEvalEngine {
  async run(claimId: string): Promise<void> {
    const claim = await db.getClaim(claimId);
    const verdict = await verifier.verify(claim);
    await db.updateClaim(claimId, verdict);
  }
}`
  }
];

const sampleEdges: CodeEdge[] = [
  {
    sourceId: 'server/dreamEngine.ts:runDreamSession',
    targetId: 'server/reEvalEngine.ts:ReEvalEngine',
    type: 'calls'
  },
  {
    sourceId: 'server/dreamEngine.ts:runDreamSession',
    targetId: 'server/rateLimiter.ts:checkLimit',
    type: 'calls'
  }
];

// ── CorpusGenerator Tests ─────────────────────────────────────────

describe('CorpusGenerator', () => {
  let generator: CorpusGenerator;

  beforeEach(() => {
    generator = new CorpusGenerator();
  });

  it('generates training pairs for all nodes', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    expect(pairs.length).toBeGreaterThan(0);
  });

  it('generates locate pairs for every node', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const locatePairs = pairs.filter(p => p.type === 'locate');
    expect(locatePairs.length).toBe(sampleNodes.length);
  });

  it('locate pair output contains file path and line numbers', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const locate = pairs.find(
      p => p.type === 'locate' && p.sourceNodeId.includes('checkLimit')
    );
    expect(locate?.output).toContain('server/rateLimiter.ts');
    expect(locate?.output).toContain('12');
    expect(locate?.output).toContain('28');
  });

  it('generates relate pairs only for nodes with edges', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const relatePairs = pairs.filter(p => p.type === 'relate');
    // Only dreamEngine has outgoing edges
    expect(relatePairs.length).toBe(1);
    expect(relatePairs[0]?.sourceNodeId).toContain('runDreamSession');
  });

  it('relate pair output mentions the correct number of dependencies', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const relate = pairs.find(p => p.type === 'relate');
    expect(relate?.output).toContain('2 outgoing relationship');
  });

  it('generates diagnose pairs for nodes with risky patterns', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const diagnosePairs = pairs.filter(p => p.type === 'diagnose');
    // dreamEngine has a TODO comment and uses ingestPipeline directly
    expect(diagnosePairs.length).toBeGreaterThan(0);
  });

  it('pairs are sorted by type', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const types = pairs.map(p => p.type);
    const sorted = [...types].sort();
    expect(types).toEqual(sorted);
  });

  it('writeJsonl produces valid JSONL output', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const tmpDir = os.tmpdir();
    const outPath = path.join(tmpDir, 'test_corpus.jsonl');
    const stats = generator.writeJsonl(pairs, outPath);

    expect(stats.totalPairs).toBe(pairs.length);
    expect(fs.existsSync(outPath)).toBe(true);

    const lines = fs.readFileSync(outPath, 'utf8').trim().split('\n');
    expect(lines.length).toBe(pairs.length);

    for (const line of lines) {
      const parsed = JSON.parse(line);
      expect(parsed).toHaveProperty('instruction');
      expect(parsed).toHaveProperty('output');
    }

    fs.unlinkSync(outPath);
  });

  it('stats byType counts match actual pair counts', () => {
    const pairs = generator.generate(sampleNodes, sampleEdges);
    const tmpDir = os.tmpdir();
    const outPath = path.join(tmpDir, 'test_stats.jsonl');
    const stats = generator.writeJsonl(pairs, outPath);

    const manualCount = pairs.reduce((acc, p) => {
      acc[p.type] = (acc[p.type] ?? 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    for (const [type, count] of Object.entries(manualCount)) {
      expect(stats.byType[type as TrainingPair['type']]).toBe(count);
    }

    fs.unlinkSync(outPath);
  });
});

// ── SelfPromptEngine Tests ────────────────────────────────────────

describe('SelfPromptEngine', () => {
  it('falls back to OpenAI when Ollama is unavailable', async () => {
    const mockResponse = {
      choices: [{ message: { content: 'DIAGNOSIS: rate limit resets on restart' } }],
      usage: { total_tokens: 42 }
    };

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('11434')) {
        return Promise.reject(new Error('connection refused'));
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });
    }) as unknown as typeof fetch;

    const engine = new SelfPromptEngine({ fallbackToOpenAI: true });
    const result = await engine.reason({
      mode: 'diagnose',
      context: 'const store = new Map()',
      failureDescription: 'Rate limit resets on server restart'
    });

    expect(result.output).toContain('DIAGNOSIS');
    expect(result.model).toBe('openai-fallback');
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('uses Ollama when it is available and model is loaded', async () => {
    const ollamaTagsResponse = {
      models: [{ name: 'codebase-slm:latest' }]
    };
    const ollamaGenerateResponse = {
      response: 'FIX: replace Map with database table',
      eval_count: 18
    };

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/tags')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(ollamaTagsResponse)
        });
      }
      if (url.includes('/api/generate')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(ollamaGenerateResponse)
        });
      }
      return Promise.reject(new Error('unexpected url'));
    }) as unknown as typeof fetch;

    const engine = new SelfPromptEngine({
      ollamaUrl: 'http://localhost:11434',
      ollamaModel: 'codebase-slm'
    });
    const result = await engine.reason({
      mode: 'repair',
      context: 'const store = new Map()'
    });

    expect(result.output).toContain('FIX');
    expect(result.model).toBe('codebase-slm');
    expect(result.tokenCount).toBe(18);
  });

  it('throws when Ollama is down and fallback is disabled', async () => {
    global.fetch = vi.fn().mockRejectedValue(
      new Error('connection refused')
    ) as unknown as typeof fetch;

    const engine = new SelfPromptEngine({ fallbackToOpenAI: false });
    await expect(
      engine.reason({ mode: 'explain', context: 'some code' })
    ).rejects.toThrow('Ollama unavailable');
  });

  it('supports all five reasoning modes', async () => {
    const modes = ['diagnose', 'repair', 'explain', 'dream', 'relate'] as const;

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('11434')) return Promise.reject(new Error('down'));
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'ok' } }],
          usage: { total_tokens: 5 }
        })
      });
    }) as unknown as typeof fetch;

    const engine = new SelfPromptEngine({ fallbackToOpenAI: true });
    for (const mode of modes) {
      const result = await engine.reason({ mode, context: 'test context' });
      expect(result.mode).toBe(mode);
    }
  });
});
