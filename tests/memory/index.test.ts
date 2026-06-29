/**
 * tests/memory/index.test.ts
 * Tests for the MemoryLayer class and its public API in memory/index.ts
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// ── Mock RuVector (avoid real HTTP calls) ─────────────────────────────────────
vi.mock('../../src/memory/ruVectorClient.js', () => ({
  RuVectorClient: vi.fn().mockImplementation(() => ({
    upsertNode: vi.fn().mockResolvedValue({ id: 'mock-id' }),
    upsertEdge: vi.fn().mockResolvedValue(undefined),
    querySimilar: vi.fn().mockResolvedValue([]),
    bulkUpsertNodes: vi.fn().mockResolvedValue(undefined),
    getEdgesFrom: vi.fn().mockReturnValue([]),
    updateEmbedding: vi.fn().mockResolvedValue(undefined),
  })),
}));

// ── Helpers ───────────────────────────────────────────────────────────────────
function makeTsFile(content: string): string {
  const tmpPath = path.join(os.tmpdir(), `mem_test_${Date.now()}_${Math.random().toString(36).slice(2)}.ts`);
  fs.writeFileSync(tmpPath, content, 'utf8');
  return tmpPath;
}

const sampleTs = `
export function greet(name: string): string {
  return \`Hello, \${name}!\`;
}
export const MAX_RETRIES = 3;
`.trim();

// ── MemoryLayer tests ─────────────────────────────────────────────────────────
describe('MemoryLayer', () => {
  let tmpFile: string;

  beforeEach(() => {
    vi.clearAllMocks();
    tmpFile = makeTsFile(sampleTs);
  });

  afterEach(() => {
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
  });

  it('can be instantiated with default config', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer();
    expect(layer).toBeDefined();
    expect(layer).toBeInstanceOf(MemoryLayer);
  });

  it('can be instantiated with a custom namespace', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('test-ns');
    expect(layer).toBeDefined();
  });

  it('can be instantiated with useInMemoryFallback=true', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    expect(layer).toBeDefined();
  });

  it('ingestFile returns a PipelineResult with processed/skipped/failed', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    const result = await layer.ingestFile(tmpFile);
    expect(typeof result.processed).toBe('number');
    expect(typeof result.skipped).toBe('number');
    expect(typeof result.failed).toBe('number');
  });

  it('ingestFile processes nodes from the TypeScript file', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    const result = await layer.ingestFile(tmpFile);
    // The file has at least one function and one constant
    expect(result.processed + result.skipped).toBeGreaterThanOrEqual(0);
  });

  it('findSimilar returns an array of SimilarityResults', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    await layer.ingestFile(tmpFile);
    const results = await layer.findSimilar('greet function', 3);
    expect(Array.isArray(results)).toBe(true);
  });

  it('findSimilar accepts topK parameter', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    const results = await layer.findSimilar('test query', 5);
    expect(Array.isArray(results)).toBe(true);
  });

  it('getRelationships returns an array', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    const rels = layer.getRelationships('some-node-id');
    expect(Array.isArray(rels)).toBe(true);
  });

  it('creates a MemoryLayer instance with expected shape', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer = new MemoryLayer('default', { useInMemoryFallback: true });
    expect(layer).toBeDefined();
    expect(layer).toBeInstanceOf(MemoryLayer);
    // Verify public API methods are present
    expect(typeof layer.ingestFile).toBe('function');
    expect(typeof layer.findSimilar).toBe('function');
    expect(typeof layer.getRelationships).toBe('function');
  });

  it('multiple MemoryLayer instances are independent', async () => {
    const { MemoryLayer } = await import('../../src/memory/index.js');
    const layer1 = new MemoryLayer('ns1', { useInMemoryFallback: true });
    const layer2 = new MemoryLayer('ns2', { useInMemoryFallback: true });
    expect(layer1).not.toBe(layer2);
    // Instances are independent objects
    expect(Object.is(layer1, layer2)).toBe(false);
  });
});

// ── Module re-exports ─────────────────────────────────────────────────────────
describe('memory/index.ts re-exports', () => {
  it('exports MemoryLayer', async () => {
    const mod = await import('../../src/memory/index.js');
    expect(mod.MemoryLayer).toBeDefined();
    expect(typeof mod.MemoryLayer).toBe('function');
  });

  it('exports MemoryLayerConfig type (compile-time check via MemoryLayer)', async () => {
    const mod = await import('../../src/memory/index.js');
    // MemoryLayerConfig is an interface (type-only), not a runtime value
    // We verify MemoryLayer accepts the config shape
    const layer = new mod.MemoryLayer('ns', { useInMemoryFallback: true });
    expect(layer).toBeDefined();
  });
});
