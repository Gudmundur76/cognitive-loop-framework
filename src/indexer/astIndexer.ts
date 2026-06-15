/**
 * ASTIndexer
 *
 * Bridges the Tree-sitter ASTExtractor and the RuVectorClient.
 * Converts CodeNode / CodeEdge output into RuVector JsNode / JsEdge
 * records and persists them in a single batchInsert transaction.
 *
 * Embedding strategy (deterministic, no external API required):
 * - 128-dim vector derived from the node's text representation
 * - Production upgrade path: swap mockEmbedding() for an OpenAI call
 *   or a local GGUF model via Ollama without changing any callers
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import { ASTExtractor, CodeNode, CodeEdge } from './extractor.js';
import {
  RuVectorClient,
  RuVectorNodeInput,
  RuVectorEdgeInput,
} from '../memory/ruVectorClient.js';

// ── Public types ──────────────────────────────────────────────────────────────

export interface IndexResult {
  /** Number of nodes written to RuVector. */
  nodesIndexed: number;
  /** Number of edges written to RuVector. */
  edgesIndexed: number;
  /** File path that was indexed. */
  filePath: string;
  /** Timestamp of the indexing run. */
  indexedAt: string;
}

export interface ASTIndexerConfig {
  /** Embedding dimensions — must match RuVectorClient dimensions. Default: 128 */
  dimensions?: number;
  /** Use deterministic mock embeddings (for tests). Default: false */
  useMockEmbeddings?: boolean;
}

// ── ASTIndexer ────────────────────────────────────────────────────────────────

export class ASTIndexer {
  private readonly extractor: ASTExtractor;
  private readonly client: RuVectorClient;
  private readonly dimensions: number;
  private readonly useMockEmbeddings: boolean;

  constructor(client: RuVectorClient, config: ASTIndexerConfig = {}) {
    this.client = client;
    this.dimensions = config.dimensions ?? 128;
    this.useMockEmbeddings = config.useMockEmbeddings ?? false;
    this.extractor = new ASTExtractor();
  }

  /**
   * Parse a TypeScript file and write all extracted nodes + edges to RuVector
   * in a single atomic batchInsert transaction.
   *
   * Returns a summary of what was indexed.
   */
  async indexFile(filePath: string): Promise<IndexResult> {
    const { nodes, edges } = this.extractor.parseFile(filePath);
    const nodeInputs = await this.buildNodeInputs(nodes);
    const edgeInputs = this.buildEdgeInputs(edges, nodes);

    await this.client.batchInsert(nodeInputs, edgeInputs);

    return {
      nodesIndexed: nodeInputs.length,
      edgesIndexed: edgeInputs.length,
      filePath,
      indexedAt: new Date().toISOString(),
    };
  }

  /**
   * Index multiple files and return per-file results.
   * Files are processed sequentially to avoid overwhelming the DB.
   */
  async indexFiles(filePaths: string[]): Promise<IndexResult[]> {
    const results: IndexResult[] = [];
    for (const fp of filePaths) {
      results.push(await this.indexFile(fp));
    }
    return results;
  }

  /**
   * Build a co-modification hyperedge for a set of files changed together
   * (e.g., in the same commit). Captures structural coupling in the graph.
   */
  async indexCoModification(
    nodeIds: string[],
    description: string
  ): Promise<string> {
    if (nodeIds.length < 2) {
      throw new Error('Co-modification hyperedge requires at least 2 nodes');
    }
    const embedding = this.buildTextEmbedding(description);
    return this.client.createHyperedge({
      nodes: nodeIds,
      description,
      embedding,
      confidence: 1.0,
      metadata: { type: 'co_modification' },
    });
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private async buildNodeInputs(
    nodes: CodeNode[]
  ): Promise<RuVectorNodeInput[]> {
    return nodes.map(node => ({
      id: node.id,
      labels: [this.capitalise(node.type)],
      embedding: this.buildNodeEmbedding(node),
      properties: {
        name: node.name,
        filePath: node.filePath,
        startLine: String(node.startLine),
        endLine: String(node.endLine),
        nodeType: node.type,
        embeddingStatus: 'complete',
      },
    }));
  }

  private buildEdgeInputs(
    edges: CodeEdge[],
    nodes: CodeNode[]
  ): RuVectorEdgeInput[] {
    const nodeIds = new Set(nodes.map(n => n.id));
    return edges
      .filter(e => nodeIds.has(e.sourceId) && nodeIds.has(e.targetId))
      .map(edge => ({
        from: edge.sourceId,
        to: edge.targetId,
        description: edge.type.toUpperCase(),
        embedding: this.buildTextEmbedding(edge.type),
        confidence: 1.0,
        metadata: { relationType: edge.type },
      }));
  }

  /**
   * Build a deterministic 128-dim embedding for a CodeNode.
   * Encodes: type + name + filePath + first 256 chars of code.
   */
  private buildNodeEmbedding(node: CodeNode): Float32Array {
    const text = [
      `type:${node.type}`,
      `name:${node.name}`,
      `file:${node.filePath}`,
      `code:${node.code.slice(0, 256)}`,
    ].join(' | ');
    return this.buildTextEmbedding(text);
  }

  /**
   * Deterministic mock embedding: 128-dim unit vector from char codes.
   * Production swap: replace with OpenAI text-embedding-3-small or
   * a local GGUF model served via Ollama.
   */
  private buildTextEmbedding(text: string): Float32Array {
    const dims = this.dimensions;
    const raw = new Float32Array(dims).fill(0);
    for (let i = 0; i < text.length; i++) {
      raw[i % dims] = (raw[i % dims] ?? 0) + text.charCodeAt(i);
    }
    const magnitude =
      Math.sqrt(raw.reduce((sum, v) => sum + v * v, 0)) || 1;
    return raw.map(v => v / magnitude);
  }

  private capitalise(s: string): string {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
}
