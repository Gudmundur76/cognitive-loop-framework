/**
 * Sprint 2 Test Suite — Memory Layer
 *
 * Tests for:
 *   - RuVectorStore (in-memory fallback mode)
 *   - EmbeddingPipeline (mock embeddings mode)
 *   - MemoryLayer integration (ingest → store → query)
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { RuVectorStore } from '../../src/memory/ruvectorStore.js';
import { EmbeddingPipeline } from '../../src/memory/embeddingPipeline.js';
import { MemoryLayer } from '../../src/memory/index.js';
import type { CodeNode, CodeEdge } from '../../src/indexer/extractor.js';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Fixtures ──────────────────────────────────────────────────────

const mockNode: CodeNode = {
  id: 'src/auth.ts:verifyToken',
  type: 'function',
  name: 'verifyToken',
  startLine: 10,
  endLine: 25,
  filePath: 'src/auth.ts',
  code: 'function verifyToken(token: string): boolean { return true; }'
};

const mockEdge: CodeEdge = {
  sourceId: 'src/auth.ts:verifyToken',
  targetId: 'src/db.ts:queryUser',
  type: 'calls'
};

// ── RuVectorStore Tests ───────────────────────────────────────────

describe('RuVectorStore (in-memory fallback)', () => {
  let store: RuVectorStore;

  beforeEach(() => {
    store = new RuVectorStore('test-namespace', true);
  });

  it('should upsert and retrieve a vector record', async () => {
    await store.upsertVector({
      id: 'node-1',
      vector: [0.1, 0.2, 0.3],
      metadata: { name: 'testFn', type: 'function' }
    });
    const results = await store.querySimilar([0.1, 0.2, 0.3], 1);
    expect(results).toHaveLength(1);
    expect(results[0]?.id).toBe('node-1');
  });

  it('should return similarity score of 1.0 for identical vectors', async () => {
    const vector = [0.5, 0.5, 0.5, 0.5];
    await store.upsertVector({ id: 'exact', vector, metadata: {} });
    const results = await store.querySimilar(vector, 1);
    expect(results[0]?.score).toBeCloseTo(1.0, 5);
  });

  it('should skip nodes with empty vectors in similarity search', async () => {
    await store.upsertVector({ id: 'empty', vector: [], metadata: {} });
    const results = await store.querySimilar([0.1, 0.2], 5);
    expect(results.find(r => r.id === 'empty')).toBeUndefined();
  });

  it('should upsert and retrieve graph edges', async () => {
    await store.upsertEdge(mockEdge);
    const edges = store.getEdgesFrom(mockEdge.sourceId);
    expect(edges).toHaveLength(1);
    expect(edges[0]?.targetId).toBe(mockEdge.targetId);
  });

  it('should not duplicate edges on repeated upsert', async () => {
    await store.upsertEdge(mockEdge);
    await store.upsertEdge(mockEdge);
    const edges = store.getEdgesFrom(mockEdge.sourceId);
    expect(edges).toHaveLength(1);
  });

  it('should bulk upsert nodes with pending embedding status', async () => {
    await store.bulkUpsertNodes([mockNode]);
    const results = await store.querySimilar([0.1], 5);
    // Pending nodes have empty vectors, so should not appear in results
    expect(results.find(r => r.id === mockNode.id)).toBeUndefined();
  });

  it('should mark embedding as complete and make node queryable', async () => {
    await store.bulkUpsertNodes([mockNode]);
    const vector = [0.3, 0.4, 0.5];
    await store.markEmbeddingComplete(mockNode.id, vector);
    const results = await store.querySimilar(vector, 1);
    expect(results[0]?.id).toBe(mockNode.id);
    expect(results[0]?.metadata.embeddingStatus).toBe('complete');
  });
});

// ── EmbeddingPipeline Tests ───────────────────────────────────────

describe('EmbeddingPipeline (mock embeddings)', () => {
  let store: RuVectorStore;
  let pipeline: EmbeddingPipeline;

  beforeEach(() => {
    store = new RuVectorStore('test-pipeline', true);
    pipeline = new EmbeddingPipeline(store, { useMockEmbeddings: true });
  });

  it('should process all nodes and return correct processed count', async () => {
    await store.bulkUpsertNodes([mockNode]);
    const result = await pipeline.run([mockNode]);
    expect(result.processed).toBe(1);
    expect(result.failed).toBe(0);
  });

  it('should generate non-zero mock embeddings', async () => {
    await store.bulkUpsertNodes([mockNode]);
    await pipeline.run([mockNode]);
    const results = await store.querySimilar([0.1, 0.2, 0.3], 5);
    // After embedding, the node should appear in similarity results
    expect(results.length).toBeGreaterThan(0);
  });

  it('should handle empty node list gracefully', async () => {
    const result = await pipeline.run([]);
    expect(result.processed).toBe(0);
    expect(result.failed).toBe(0);
    expect(result.skipped).toBe(0);
  });
});

// ── MemoryLayer Integration Tests ────────────────────────────────

describe('MemoryLayer (integration)', () => {
  let memory: MemoryLayer;

  beforeEach(() => {
    memory = new MemoryLayer('test-memory', {
      useInMemoryFallback: true,
      embedding: { useMockEmbeddings: true }
    });
  });

  it('should ingest a real TypeScript file and return pipeline result', async () => {
    const filePath = path.resolve(
      __dirname,
      '../../src/indexer/extractor.ts'
    );
    const result = await memory.ingestFile(filePath);
    expect(result.processed).toBeGreaterThan(0);
    expect(result.failed).toBe(0);
  });

  it('should find similar nodes after ingestion', async () => {
    const filePath = path.resolve(
      __dirname,
      '../../src/indexer/extractor.ts'
    );
    await memory.ingestFile(filePath);
    const results = await memory.findSimilar('function that parses files', 3);
    expect(Array.isArray(results)).toBe(true);
  });
});
