/**
 * RuVectorClient tests
 *
 * All tests use in-memory GraphDatabase (no storagePath) so they are
 * fully isolated and leave no disk artefacts.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { RuVectorClient } from '../../src/memory/ruVectorClient.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeClient(): RuVectorClient {
  return new RuVectorClient({ dimensions: 128 });
}

function makeEmbedding(seed: number, dims = 128): Float32Array {
  const raw = new Float32Array(dims);
  for (let i = 0; i < dims; i++) {
    raw[i] = Math.sin(seed + i);
  }
  const mag = Math.sqrt(raw.reduce((s, v) => s + v * v, 0)) || 1;
  return raw.map(v => v / mag);
}

// ── Node operations ───────────────────────────────────────────────────────────

describe('RuVectorClient — node operations', () => {
  let client: RuVectorClient;

  beforeEach(() => {
    client = makeClient();
  });

  it('upsertNode returns a non-empty string ID', async () => {
    const id = await client.upsertNode({
      id: 'node-1',
      labels: ['Function'],
      embedding: makeEmbedding(1),
      properties: { name: 'myFunc', filePath: 'src/a.ts' },
    });
    expect(typeof id).toBe('string');
    expect(id.length).toBeGreaterThan(0);
  });

  it('batchUpsertNodes returns one ID per input node', async () => {
    const inputs = [
      { id: 'n1', labels: ['Class'], embedding: makeEmbedding(1), properties: { name: 'A' } },
      { id: 'n2', labels: ['Class'], embedding: makeEmbedding(2), properties: { name: 'B' } },
      { id: 'n3', labels: ['Function'], embedding: makeEmbedding(3), properties: { name: 'C' } },
    ];
    const ids = await client.batchUpsertNodes(inputs);
    expect(ids).toHaveLength(3);
    for (const id of ids) {
      expect(typeof id).toBe('string');
    }
  });

  it('batchUpsertNodes with empty array returns empty array', async () => {
    const ids = await client.batchUpsertNodes([]);
    expect(ids).toEqual([]);
  });
});

// ── Edge operations ───────────────────────────────────────────────────────────

describe('RuVectorClient — edge operations', () => {
  let client: RuVectorClient;

  beforeEach(async () => {
    client = makeClient();
    // Pre-create nodes so edges have valid endpoints
    await client.batchUpsertNodes([
      { id: 'src', labels: ['Function'], embedding: makeEmbedding(10), properties: { name: 'caller' } },
      { id: 'tgt', labels: ['Function'], embedding: makeEmbedding(11), properties: { name: 'callee' } },
    ]);
  });

  it('createEdge returns a non-empty string ID', async () => {
    const id = await client.createEdge({
      from: 'src',
      to: 'tgt',
      description: 'calls',
      embedding: makeEmbedding(20),
      confidence: 0.9,
    });
    expect(typeof id).toBe('string');
    expect(id.length).toBeGreaterThan(0);
  });

  it('createEdge accepts optional metadata', async () => {
    const id = await client.createEdge({
      from: 'src',
      to: 'tgt',
      description: 'imports',
      embedding: makeEmbedding(21),
      metadata: { relationType: 'import' },
    });
    expect(typeof id).toBe('string');
  });
});

// ── Batch insert ──────────────────────────────────────────────────────────────

describe('RuVectorClient — batchInsert', () => {
  let client: RuVectorClient;

  beforeEach(() => {
    client = makeClient();
  });

  it('inserts nodes and edges atomically', async () => {
    const result = await client.batchInsert(
      [
        { id: 'a', labels: ['Class'], embedding: makeEmbedding(1), properties: { name: 'A' } },
        { id: 'b', labels: ['Class'], embedding: makeEmbedding(2), properties: { name: 'B' } },
      ],
      [
        {
          from: 'a',
          to: 'b',
          description: 'extends',
          embedding: makeEmbedding(3),
          confidence: 1.0,
        },
      ]
    );
    expect(result.nodeIds).toHaveLength(2);
    expect(result.edgeIds).toHaveLength(1);
  });

  it('batchInsert with no edges returns empty edgeIds', async () => {
    const result = await client.batchInsert(
      [{ id: 'x', labels: ['Variable'], embedding: makeEmbedding(5), properties: { name: 'x' } }],
      []
    );
    expect(result.nodeIds).toHaveLength(1);
    expect(result.edgeIds).toHaveLength(0);
  });
});

// ── Hyperedge operations ──────────────────────────────────────────────────────

describe('RuVectorClient — hyperedge operations', () => {
  let client: RuVectorClient;

  beforeEach(async () => {
    client = makeClient();
    await client.batchUpsertNodes([
      { id: 'h1', labels: ['Function'], embedding: makeEmbedding(30), properties: { name: 'f1' } },
      { id: 'h2', labels: ['Function'], embedding: makeEmbedding(31), properties: { name: 'f2' } },
      { id: 'h3', labels: ['Function'], embedding: makeEmbedding(32), properties: { name: 'f3' } },
    ]);
  });

  it('createHyperedge returns a non-empty string ID', async () => {
    const id = await client.createHyperedge({
      nodes: ['h1', 'h2', 'h3'],
      description: 'co-modified in commit abc123',
      embedding: makeEmbedding(40),
      confidence: 0.8,
    });
    expect(typeof id).toBe('string');
    expect(id.length).toBeGreaterThan(0);
  });

  it('searchHyperedges returns an array', async () => {
    await client.createHyperedge({
      nodes: ['h1', 'h2', 'h3'],
      description: 'co-modified',
      embedding: makeEmbedding(40),
    });

    const results = await client.searchHyperedges(makeEmbedding(40), 5);
    expect(Array.isArray(results)).toBe(true);
    if (results.length > 0) {
      expect(typeof results[0]!.id).toBe('string');
      expect(typeof results[0]!.score).toBe('number');
    }
  });
});

// ── Graph stats ───────────────────────────────────────────────────────────────

describe('RuVectorClient — stats', () => {
  it('stats returns an object with node_count', async () => {
    const client = makeClient();
    await client.upsertNode({
      id: 'stat-node',
      labels: ['Test'],
      embedding: makeEmbedding(99),
      properties: { name: 'test' },
    });
    const stats = await client.stats();
    expect(stats).toBeDefined();
    expect(typeof stats.totalNodes).toBe('number');
    expect(stats.totalNodes).toBeGreaterThanOrEqual(1);
  });
});

// ── Static open ───────────────────────────────────────────────────────────────

describe('RuVectorClient.open', () => {
  it('creates a client instance without throwing', () => {
    // In-memory open (no actual file path needed for basic instantiation)
    const client = RuVectorClient.open('/tmp/test-ruvector-open', 128);
    expect(client).toBeInstanceOf(RuVectorClient);
  });
});
