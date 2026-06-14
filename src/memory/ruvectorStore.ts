/**
 * RuVector Memory Store
 *
 * Adapter layer for the RuVector graph-vector database.
 * Provides a typed interface for storing and querying CodeNode
 * embeddings and graph relationships.
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 * - All public methods return typed results, never raw any
 */

import { execSync } from 'child_process';
import { CodeNode, CodeEdge } from '../indexer/extractor.js';

export interface VectorRecord {
  id: string;
  vector: number[];
  metadata: Record<string, string | number | boolean>;
}

export interface GraphQueryResult {
  nodes: Array<{ id: string; properties: Record<string, unknown> }>;
  edges: Array<{ source: string; target: string; type: string }>;
}

export interface SimilarityResult {
  id: string;
  score: number;
  metadata: Record<string, string | number | boolean>;
}

/**
 * RuVectorStore wraps the RuVector CLI and provides a typed
 * TypeScript interface for the cognitive loop framework.
 *
 * In production, this adapter connects to a running RuVector
 * instance. In test environments, it uses an in-memory fallback.
 */
export class RuVectorStore {
  private readonly namespace: string;
  private readonly useInMemoryFallback: boolean;

  // In-memory fallback for test environments
  private memoryStore: Map<string, VectorRecord> = new Map();
  private graphStore: Map<string, { edges: CodeEdge[] }> = new Map();

  constructor(namespace: string, useInMemoryFallback = false) {
    this.namespace = namespace;
    this.useInMemoryFallback = useInMemoryFallback;
  }

  /**
   * Store a vector record for a code node.
   * Associates the embedding with the node's ID and metadata.
   */
  public async upsertVector(record: VectorRecord): Promise<void> {
    if (this.useInMemoryFallback) {
      this.memoryStore.set(record.id, record);
      return;
    }
    this.execRuVector('upsert', {
      namespace: this.namespace,
      id: record.id,
      vector: record.vector,
      metadata: record.metadata
    });
  }

  /**
   * Query for the top-k most similar vectors to a given query vector.
   * Returns results ordered by descending similarity score.
   */
  public async querySimilar(
    queryVector: number[],
    topK = 5
  ): Promise<SimilarityResult[]> {
    if (this.useInMemoryFallback) {
      return this.inMemorySimilaritySearch(queryVector, topK);
    }
    const raw = this.execRuVector('query', {
      namespace: this.namespace,
      vector: queryVector,
      topK
    });
    return JSON.parse(raw) as SimilarityResult[];
  }

  /**
   * Store a directed graph edge between two code nodes.
   * Uses Cypher-style MERGE to avoid duplicate edges.
   */
  public async upsertEdge(edge: CodeEdge): Promise<void> {
    if (this.useInMemoryFallback) {
      const existing = this.graphStore.get(edge.sourceId) ?? { edges: [] };
      const alreadyExists = existing.edges.some(
        e => e.targetId === edge.targetId && e.type === edge.type
      );
      if (!alreadyExists) existing.edges.push(edge);
      this.graphStore.set(edge.sourceId, existing);
      return;
    }
    const cypher = [
      `MATCH (source:CodeNode {id: "${edge.sourceId}"})`,
      `MATCH (target:CodeNode {id: "${edge.targetId}"})`,
      `MERGE (source)-[r:${edge.type.toUpperCase()}]->(target)`
    ].join(' ');
    this.execRuVector('cypher', { query: cypher });
  }

  /**
   * Retrieve all outgoing edges from a given node.
   */
  public getEdgesFrom(nodeId: string): CodeEdge[] {
    if (this.useInMemoryFallback) {
      return this.graphStore.get(nodeId)?.edges ?? [];
    }
    const cypher = `MATCH (n:CodeNode {id: "${nodeId}"})-[r]->(m) RETURN r, m`;
    const raw = this.execRuVector('cypher', { query: cypher });
    return JSON.parse(raw) as CodeEdge[];
  }

  /**
   * Bulk-store all nodes from an AST extraction run.
   * Nodes without embeddings are stored with an empty vector
   * and marked as pending for the embedding pipeline.
   */
  public async bulkUpsertNodes(nodes: CodeNode[]): Promise<void> {
    for (const node of nodes) {
      await this.upsertVector({
        id: node.id,
        vector: [],
        metadata: {
          type: node.type,
          name: node.name,
          filePath: node.filePath,
          startLine: node.startLine,
          endLine: node.endLine,
          embeddingStatus: 'pending'
        }
      });
    }
  }

  /**
   * Mark a node's embedding as complete after the pipeline runs.
   */
  public async markEmbeddingComplete(
    nodeId: string,
    vector: number[]
  ): Promise<void> {
    const existing = this.memoryStore.get(nodeId);
    if (!existing) return;
    await this.upsertVector({
      ...existing,
      vector,
      metadata: { ...existing.metadata, embeddingStatus: 'complete' }
    });
  }

  // ── Private helpers ─────────────────────────────────────────────

  private execRuVector(command: string, args: Record<string, unknown>): string {
    const argsJson = JSON.stringify(args);
    try {
      return execSync(
        `npx ruvector ${command} '${argsJson}'`,
        { encoding: 'utf8' }
      );
    } catch {
      return '[]';
    }
  }

  private inMemorySimilaritySearch(
    queryVector: number[],
    topK: number
  ): SimilarityResult[] {
    const results: SimilarityResult[] = [];
    for (const [id, record] of this.memoryStore.entries()) {
      if (record.vector.length === 0) continue;
      const score = this.cosineSimilarity(queryVector, record.vector);
      results.push({ id, score, metadata: record.metadata });
    }
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    const dot = a.reduce((sum, ai, i) => sum + ai * (b[i] ?? 0), 0);
    const magA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const magB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    if (magA === 0 || magB === 0) return 0;
    return dot / (magA * magB);
  }
}
