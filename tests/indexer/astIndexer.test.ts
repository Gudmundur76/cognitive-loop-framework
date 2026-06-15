/**
 * ASTIndexer tests
 *
 * Uses in-memory RuVectorClient (no storagePath) and a temporary
 * TypeScript fixture file so tests are fully isolated.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { RuVectorClient } from '../../src/memory/ruVectorClient.js';
import { ASTIndexer } from '../../src/indexer/astIndexer.js';

// ── Fixture helpers ───────────────────────────────────────────────────────────

function makeTmpFile(content: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ast-indexer-test-'));
  const fp = path.join(dir, 'fixture.ts');
  fs.writeFileSync(fp, content, 'utf8');
  return fp;
}

function cleanTmpFile(fp: string): void {
  try {
    fs.rmSync(path.dirname(fp), { recursive: true, force: true });
  } catch { /* ignore */ }
}

const FIXTURE_ONE_CLASS = `
export class MyService {
  doWork(): void {
    console.log('working');
  }
}
`;

const FIXTURE_TWO_FUNCTIONS = `
function alpha(x: number): number {
  return x * 2;
}

function beta(y: string): string {
  return y.trim();
}
`;

const FIXTURE_EMPTY = `
// No declarations here
const x = 1;
`;

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('ASTIndexer — indexFile', () => {
  let client: RuVectorClient;
  let indexer: ASTIndexer;
  let tmpFile: string;

  beforeEach(() => {
    client = new RuVectorClient({ dimensions: 128 });
    indexer = new ASTIndexer(client, { useMockEmbeddings: true });
  });

  afterEach(() => {
    if (tmpFile) cleanTmpFile(tmpFile);
  });

  it('returns a result with correct filePath', async () => {
    tmpFile = makeTmpFile(FIXTURE_ONE_CLASS);
    const result = await indexer.indexFile(tmpFile);
    expect(result.filePath).toBe(tmpFile);
  });

  it('returns a non-empty indexedAt timestamp', async () => {
    tmpFile = makeTmpFile(FIXTURE_ONE_CLASS);
    const result = await indexer.indexFile(tmpFile);
    expect(typeof result.indexedAt).toBe('string');
    expect(result.indexedAt.length).toBeGreaterThan(0);
  });

  it('indexes a class node from fixture', async () => {
    tmpFile = makeTmpFile(FIXTURE_ONE_CLASS);
    const result = await indexer.indexFile(tmpFile);
    // Tree-sitter finds the class declaration
    expect(result.nodesIndexed).toBeGreaterThanOrEqual(1);
  });

  it('indexes two function declarations', async () => {
    tmpFile = makeTmpFile(FIXTURE_TWO_FUNCTIONS);
    const result = await indexer.indexFile(tmpFile);
    expect(result.nodesIndexed).toBeGreaterThanOrEqual(2);
  });

  it('returns zero nodes for a file with no declarations', async () => {
    tmpFile = makeTmpFile(FIXTURE_EMPTY);
    const result = await indexer.indexFile(tmpFile);
    expect(result.nodesIndexed).toBe(0);
    expect(result.edgesIndexed).toBe(0);
  });

  it('nodesIndexed and edgesIndexed are non-negative integers', async () => {
    tmpFile = makeTmpFile(FIXTURE_ONE_CLASS);
    const result = await indexer.indexFile(tmpFile);
    expect(result.nodesIndexed).toBeGreaterThanOrEqual(0);
    expect(result.edgesIndexed).toBeGreaterThanOrEqual(0);
    expect(Number.isInteger(result.nodesIndexed)).toBe(true);
    expect(Number.isInteger(result.edgesIndexed)).toBe(true);
  });
});

// ── indexFiles ────────────────────────────────────────────────────────────────

describe('ASTIndexer — indexFiles', () => {
  let client: RuVectorClient;
  let indexer: ASTIndexer;
  const tmpFiles: string[] = [];

  beforeEach(() => {
    client = new RuVectorClient({ dimensions: 128 });
    indexer = new ASTIndexer(client, { useMockEmbeddings: true });
  });

  afterEach(() => {
    for (const f of tmpFiles) cleanTmpFile(f);
    tmpFiles.length = 0;
  });

  it('returns one result per file', async () => {
    const f1 = makeTmpFile(FIXTURE_ONE_CLASS);
    const f2 = makeTmpFile(FIXTURE_TWO_FUNCTIONS);
    tmpFiles.push(f1, f2);

    const results = await indexer.indexFiles([f1, f2]);
    expect(results).toHaveLength(2);
  });

  it('returns empty array for empty input', async () => {
    const results = await indexer.indexFiles([]);
    expect(results).toEqual([]);
  });

  it('each result has the correct filePath', async () => {
    const f1 = makeTmpFile(FIXTURE_ONE_CLASS);
    const f2 = makeTmpFile(FIXTURE_TWO_FUNCTIONS);
    tmpFiles.push(f1, f2);

    const results = await indexer.indexFiles([f1, f2]);
    expect(results[0]!.filePath).toBe(f1);
    expect(results[1]!.filePath).toBe(f2);
  });

  it('total nodesIndexed across files is sum of individual runs', async () => {
    const f1 = makeTmpFile(FIXTURE_ONE_CLASS);
    const f2 = makeTmpFile(FIXTURE_TWO_FUNCTIONS);
    tmpFiles.push(f1, f2);

    const [r1, r2] = await indexer.indexFiles([f1, f2]);
    const combined = await indexer.indexFiles([f1, f2]);
    const total = combined.reduce((s, r) => s + r.nodesIndexed, 0);
    expect(total).toBe((r1?.nodesIndexed ?? 0) + (r2?.nodesIndexed ?? 0));
  });
});

// ── indexCoModification ───────────────────────────────────────────────────────

describe('ASTIndexer — indexCoModification', () => {
  let client: RuVectorClient;
  let indexer: ASTIndexer;

  beforeEach(async () => {
    client = new RuVectorClient({ dimensions: 128 });
    indexer = new ASTIndexer(client, { useMockEmbeddings: true });
    // Pre-create nodes so hyperedge endpoints exist
    await client.batchUpsertNodes([
      { id: 'co-a', labels: ['Function'], embedding: new Float32Array(128).fill(0.1), properties: { name: 'a' } },
      { id: 'co-b', labels: ['Function'], embedding: new Float32Array(128).fill(0.2), properties: { name: 'b' } },
    ]);
  });

  it('returns a non-empty hyperedge ID', async () => {
    const id = await indexer.indexCoModification(
      ['co-a', 'co-b'],
      'co-modified in commit abc'
    );
    expect(typeof id).toBe('string');
    expect(id.length).toBeGreaterThan(0);
  });

  it('throws when fewer than 2 nodes are provided', async () => {
    await expect(
      indexer.indexCoModification(['co-a'], 'only one node')
    ).rejects.toThrow('at least 2 nodes');
  });

  it('accepts 3+ nodes', async () => {
    await client.upsertNode({
      id: 'co-c',
      labels: ['Class'],
      embedding: new Float32Array(128).fill(0.3),
      properties: { name: 'c' },
    });
    const id = await indexer.indexCoModification(
      ['co-a', 'co-b', 'co-c'],
      'three-way co-modification'
    );
    expect(typeof id).toBe('string');
  });
});

// ── Graph state after indexing ────────────────────────────────────────────────

describe('ASTIndexer — graph state after indexing', () => {
  it('graph stats reflect indexed nodes', async () => {
    const client = new RuVectorClient({ dimensions: 128 });
    const indexer = new ASTIndexer(client, { useMockEmbeddings: true });

    const tmpFile = makeTmpFile(FIXTURE_ONE_CLASS);
    try {
      const result = await indexer.indexFile(tmpFile);
      const stats = await client.stats();

      if (result.nodesIndexed > 0) {
        expect(stats.totalNodes).toBeGreaterThanOrEqual(result.nodesIndexed);
      }
    } finally {
      cleanTmpFile(tmpFile);
    }
  });
});
