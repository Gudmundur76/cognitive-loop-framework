/**
 * RuVectorClient
 *
 * Typed wrapper around the @ruvector/graph-node native binding.
 * Replaces the old CLI-exec stub in ruvectorStore.ts with direct
 * in-process calls to the Rust-backed GraphDatabase.
 *
 * Responsibilities:
 * - Manage a single GraphDatabase instance per client
 * - Encode/decode string properties to/from the Record<string,string> format
 *   that the native binding requires (all values must be strings)
 * - Provide a stable TypeScript interface that the rest of the framework
 *   depends on, so the underlying DB can be swapped without touching callers
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import {
  GraphDatabase,
  JsDistanceMetric,
  JsNode,
  JsEdge,
  JsHyperedge,
  JsGraphStats,
  JsNodeResult,
  JsEdgeResult,
} from '@ruvector/graph-node';

// ── Public types ──────────────────────────────────────────────────────────────

export interface RuVectorNodeInput {
  id: string;
  labels: string[];
  embedding: Float32Array;
  properties: Record<string, string>;
}

export interface RuVectorEdgeInput {
  from: string;
  to: string;
  description: string;
  embedding: Float32Array;
  confidence?: number;
  metadata?: Record<string, string>;
}

export interface RuVectorHyperedgeInput {
  nodes: string[];
  description: string;
  embedding: Float32Array;
  confidence?: number;
  metadata?: Record<string, string>;
}

export interface RuVectorQueryResult {
  nodes: JsNodeResult[];
  edges: JsEdgeResult[];
}

export interface RuVectorSimilarityResult {
  id: string;
  score: number;
}

export interface RuVectorClientConfig {
  /** Vector dimensions — must match embedding model output. Default: 128 */
  dimensions?: number;
  /** Persist graph to disk at this path. Omit for in-memory only. */
  storagePath?: string;
}

// ── RuVectorClient ────────────────────────────────────────────────────────────

export class RuVectorClient {
  private readonly db: GraphDatabase;
  private readonly dimensions: number;

  constructor(config: RuVectorClientConfig = {}) {
    this.dimensions = config.dimensions ?? 128;
    this.db = new GraphDatabase({
      dimensions: this.dimensions,
      distanceMetric: JsDistanceMetric.Cosine,
      ...(config.storagePath ? { storagePath: config.storagePath } : {}),
    });
  }

  /**
   * Open an existing persisted graph database from disk.
   */
  static open(storagePath: string, dimensions = 128): RuVectorClient {
    const client = new RuVectorClient({ storagePath, dimensions });
    return client;
  }

  // ── Node operations ────────────────────────────────────────────────────────

  /**
   * Upsert a node into the graph.
   * If a node with the same id already exists it will be overwritten.
   */
  async upsertNode(input: RuVectorNodeInput): Promise<string> {
    const node: JsNode = {
      id: input.id,
      embedding: input.embedding,
      labels: input.labels,
      properties: input.properties,
    };
    return this.db.createNode(node);
  }

  /**
   * Batch-upsert multiple nodes in a single transaction.
   * Returns the list of node IDs that were inserted.
   */
  async batchUpsertNodes(inputs: RuVectorNodeInput[]): Promise<string[]> {
    if (inputs.length === 0) return [];
    const nodes: JsNode[] = inputs.map(input => ({
      id: input.id,
      embedding: input.embedding,
      labels: input.labels,
      properties: input.properties,
    }));
    const result = await this.db.batchInsert({ nodes, edges: [] });
    return result.nodeIds;
  }

  // ── Edge operations ────────────────────────────────────────────────────────

  /**
   * Create a directed edge between two existing nodes.
   * Returns the generated edge ID.
   */
  async createEdge(input: RuVectorEdgeInput): Promise<string> {
    const edge: JsEdge = {
      from: input.from,
      to: input.to,
      description: input.description,
      embedding: input.embedding,
      confidence: input.confidence ?? 1.0,
      metadata: input.metadata,
    };
    return this.db.createEdge(edge);
  }

  /**
   * Batch-insert nodes and edges in a single atomic transaction.
   * Use this when ingesting an AST file to avoid partial writes.
   */
  async batchInsert(
    nodes: RuVectorNodeInput[],
    edges: RuVectorEdgeInput[]
  ): Promise<{ nodeIds: string[]; edgeIds: string[] }> {
    const jsNodes: JsNode[] = nodes.map(n => ({
      id: n.id,
      embedding: n.embedding,
      labels: n.labels,
      properties: n.properties,
    }));
    const jsEdges: JsEdge[] = edges.map(e => ({
      from: e.from,
      to: e.to,
      description: e.description,
      embedding: e.embedding,
      confidence: e.confidence ?? 1.0,
      metadata: e.metadata,
    }));
    return this.db.batchInsert({ nodes: jsNodes, edges: jsEdges });
  }

  // ── Hyperedge operations ───────────────────────────────────────────────────

  /**
   * Create a hyperedge connecting 3+ nodes (e.g., "co-modified in same PR").
   */
  async createHyperedge(input: RuVectorHyperedgeInput): Promise<string> {
    const he: JsHyperedge = {
      nodes: input.nodes,
      description: input.description,
      embedding: input.embedding,
      confidence: input.confidence ?? 1.0,
      metadata: input.metadata,
    };
    return this.db.createHyperedge(he);
  }

  /**
   * Search for hyperedges semantically similar to the given embedding.
   * Returns the top-k results ordered by descending similarity score.
   */
  async searchHyperedges(
    embedding: Float32Array,
    topK = 5
  ): Promise<RuVectorSimilarityResult[]> {
    const raw = await this.db.searchHyperedges({ embedding, k: topK });
    return raw.map(r => ({ id: r.id, score: r.score }));
  }

  // ── Query operations ───────────────────────────────────────────────────────

  /**
   * Execute a Cypher-like query against the graph.
   * Returns all matching nodes and edges.
   */
  async query(cypher: string): Promise<RuVectorQueryResult> {
    const result = await this.db.query(cypher);
    return { nodes: result.nodes, edges: result.edges };
  }

  /**
   * Get all nodes reachable within k hops from a starting node.
   * Used by the Frontier Layer to find blast-radius of a change.
   */
  async kHopNeighbors(nodeId: string, k: number): Promise<string[]> {
    return this.db.kHopNeighbors(nodeId, k);
  }

  // ── Stats ──────────────────────────────────────────────────────────────────

  async stats(): Promise<JsGraphStats> {
    return this.db.stats();
  }

  /**
   * Subscribe to graph change events.
   * Callback receives raw change objects from the native layer.
   */
  subscribe(callback: (...args: unknown[]) => void): void {
    this.db.subscribe(callback);
  }
}
