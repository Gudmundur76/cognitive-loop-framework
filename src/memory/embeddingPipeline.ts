/**
 * Embedding Pipeline
 *
 * Generates vector embeddings for CodeNode objects extracted by the
 * ASTExtractor and stores them in the RuVectorStore.
 *
 * Embedding strategy:
 * - Input text = node name + file path + code snippet (truncated)
 * - Model = text-embedding-3-small (OpenAI-compatible API)
 * - Batch size = 20 nodes per API call to respect rate limits
 * - Fallback = deterministic mock embeddings for test environments
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 */

import OpenAI from 'openai';
import { CodeNode } from '../indexer/extractor.js';
import { RuVectorStore } from './ruvectorStore.js';

export interface EmbeddingPipelineConfig {
  batchSize?: number;
  model?: string;
  useMockEmbeddings?: boolean;
}

export interface PipelineResult {
  processed: number;
  failed: number;
  skipped: number;
}

const DEFAULT_CONFIG: Required<EmbeddingPipelineConfig> = {
  batchSize: 20,
  model: 'text-embedding-3-small',
  useMockEmbeddings: false
};

export class EmbeddingPipeline {
  private readonly client: OpenAI;
  private readonly store: RuVectorStore;
  private readonly config: Required<EmbeddingPipelineConfig>;

  constructor(store: RuVectorStore, config: EmbeddingPipelineConfig = {}) {
    this.store = store;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.client = new OpenAI();
  }

  /**
   * Process all nodes in batches and store their embeddings.
   * Returns a summary of the pipeline run.
   */
  public async run(nodes: CodeNode[]): Promise<PipelineResult> {
    const result: PipelineResult = { processed: 0, failed: 0, skipped: 0 };
    const batches = this.chunk(nodes, this.config.batchSize);

    for (const batch of batches) {
      const batchResult = await this.processBatch(batch);
      result.processed += batchResult.processed;
      result.failed += batchResult.failed;
      result.skipped += batchResult.skipped;
    }

    return result;
  }

  /**
   * Process a single batch of nodes.
   * Generates embeddings and stores them in RuVector.
   */
  private async processBatch(nodes: CodeNode[]): Promise<PipelineResult> {
    const result: PipelineResult = { processed: 0, failed: 0, skipped: 0 };
    const texts = nodes.map(n => this.buildEmbeddingText(n));

    try {
      const vectors = this.config.useMockEmbeddings
        ? texts.map(t => this.mockEmbedding(t))
        : await this.fetchEmbeddings(texts);

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const vector = vectors[i];
        if (!node || !vector) { result.skipped++; continue; }
        await this.store.markEmbeddingComplete(node.id, vector);
        result.processed++;
      }
    } catch {
      result.failed += nodes.length;
    }

    return result;
  }

  /**
   * Build the text representation of a node for embedding.
   * Truncates code to 512 chars to stay within token limits.
   */
  private buildEmbeddingText(node: CodeNode): string {
    const codeSnippet = node.code.slice(0, 512);
    return [
      `type:${node.type}`,
      `name:${node.name}`,
      `file:${node.filePath}`,
      `code:${codeSnippet}`
    ].join(' | ');
  }

  /**
   * Call the OpenAI-compatible embeddings API for a batch of texts.
   */
  private async fetchEmbeddings(texts: string[]): Promise<number[][]> {
    const response = await this.client.embeddings.create({
      model: this.config.model,
      input: texts
    });
    return response.data
      .sort((a, b) => a.index - b.index)
      .map(d => d.embedding);
  }

  /**
   * Deterministic mock embedding for test environments.
   * Produces a 128-dim vector based on the text's char codes.
   */
  private mockEmbedding(text: string): number[] {
    const dims = 128;
    const vector = new Array<number>(dims).fill(0);
    for (let i = 0; i < text.length; i++) {
      vector[i % dims] = (vector[i % dims] ?? 0) + text.charCodeAt(i);
    }
    const magnitude = Math.sqrt(
      vector.reduce((sum, v) => sum + v * v, 0)
    ) || 1;
    return vector.map(v => v / magnitude);
  }

  /**
   * Split an array into fixed-size chunks.
   */
  private chunk<T>(arr: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  }
}
