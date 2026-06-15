/**
 * Memory Layer — Public API
 *
 * Unified entry point that wires together:
 *   ASTExtractor → GraphWriter → RuVectorStore → EmbeddingPipeline
 *
 * Usage:
 *   const memory = new MemoryLayer('my-project');
 *   await memory.ingestFile('./src/server.ts');
 *   const similar = await memory.findSimilar('function that handles auth');
 */

import { ASTExtractor } from '../indexer/extractor.js';
import { GraphWriter } from '../graph/writer.js';
import { RuVectorStore } from './ruvectorStore.js';
import { EmbeddingPipeline, EmbeddingPipelineConfig } from './embeddingPipeline.js';
import type { SimilarityResult } from './ruvectorStore.js';
import type { PipelineResult } from './embeddingPipeline.js';

export interface MemoryLayerConfig {
  useInMemoryFallback?: boolean;
  embedding?: EmbeddingPipelineConfig;
}

export class MemoryLayer {
  private readonly extractor: ASTExtractor;
  private readonly writer: GraphWriter;
  private readonly store: RuVectorStore;
  private readonly pipeline: EmbeddingPipeline;

  constructor(namespace: string = 'default', config: MemoryLayerConfig = {}) {
    this.extractor = new ASTExtractor();
    this.writer = new GraphWriter();
    this.store = new RuVectorStore(
      namespace,
      config.useInMemoryFallback ?? false
    );
    this.pipeline = new EmbeddingPipeline(this.store, config.embedding);
  }

  /**
   * Ingest a single TypeScript file into the memory layer.
   * Extracts nodes, stores in graph, and generates embeddings.
   */
  public async ingestFile(filePath: string): Promise<PipelineResult> {
    const { nodes, edges } = this.extractor.parseFile(filePath);
    await this.store.bulkUpsertNodes(nodes);
    for (const edge of edges) {
      await this.store.upsertEdge(edge);
    }
    return this.pipeline.run(nodes);
  }

  /**
   * Find code nodes semantically similar to a natural language query.
   * Embeds the query text and performs HNSW similarity search.
   */
  public async findSimilar(
    queryText: string,
    topK = 5
  ): Promise<SimilarityResult[]> {
    // Use deterministic mock embedding for query to avoid extra API call in dev
    const queryVector = EmbeddingPipeline.getMockEmbedding(queryText);
    return this.store.querySimilar(queryVector, topK);
  }

  /**
   * Get all outgoing relationships from a given node ID.
   * Used by the reasoning layer to traverse the code graph.
   */
  public getRelationships(nodeId: string) {
    return this.store.getEdgesFrom(nodeId);
  }
}
