/**
 * CompoundingLog
 *
 * Implements the structured memory log described in self_improvement_mechanics.md.
 * Each entry is a JSON record capturing a full repair/training cycle:
 *   trigger → diagnosis → patch → test_result → graph_delta → meta_review
 *
 * The log is NOT a diary. It is a training corpus for the next repair.
 *
 * Sprint 2 upgrade: querySimilar() now uses RuVector hyperedge search
 * (semantic vector similarity) instead of in-process TF-IDF cosine.
 * Each append() also writes a graph node + hyperedge to RuVector so
 * the compounding log is queryable as a living knowledge graph.
 *
 * Design constraints: max 250 lines, max 20 lines/function, max 3 params
 */
import * as fs from 'fs';
import * as path from 'path';
import { RuVectorClient } from './ruVectorClient.js';

// ── Types ─────────────────────────────────────────────────────────────────────
export type TriggerType =
  | 'test_failure'
  | 'verdict_event'
  | 'scheduled_scan'
  | 'meta_alert'
  | 'dream_hypothesis'
  | 'manual';

export type MetaReview = 'APPROVED' | 'REJECTED' | 'PENDING' | 'ESCALATED';

export interface GraphDelta {
  added: string[];
  removed: string[];
  modified: string[];
}

export interface CompoundingEntry {
  timestamp: string;
  trigger: TriggerType;
  /** The test name or claim ID that triggered this entry. */
  test: string;
  /** SLM-generated diagnosis of the root cause. */
  diagnosis: string;
  /** SHA-256 hash of the applied patch (or empty string if no patch). */
  patch_hash: string;
  /** Result of the Ralph Wiggum validation gate. */
  test_result: 'PASS' | 'FAIL' | 'SKIP';
  /** Changes to the code/knowledge graph caused by this entry. */
  graph_delta: GraphDelta;
  /** SLM confidence score for the diagnosis (0.0–1.0). */
  slm_confidence: number;
  /** Number of frontier evidence sources explored. */
  frontier_explored: number;
  /** L4 Meta-Agent review decision. */
  meta_review: MetaReview;
  /** Optional free-form metadata (layer, domain, etc.). */
  metadata?: Record<string, unknown>;
}

export interface QueryResult {
  entry: CompoundingEntry;
  /** Similarity score (0.0–1.0) — higher = more similar. */
  similarity: number;
}

// ── CompoundingLog ────────────────────────────────────────────────────────────
export class CompoundingLog {
  private readonly logPath: string;
  private readonly ruVector: RuVectorClient | null;

  /**
   * @param logPath   Absolute path to the JSONL log file.
   * @param ruVector  Optional RuVectorClient for hybrid search.
   *                  When omitted, querySimilar falls back to TF-IDF.
   */
  constructor(logPath: string, ruVector?: RuVectorClient) {
    this.logPath = logPath;
    this.ruVector = ruVector ?? null;
    fs.mkdirSync(path.dirname(logPath), { recursive: true });
  }

  /**
   * Append a new entry to the compounding log.
   * Also writes a graph node + hyperedge to RuVector (if configured)
   * so the log is queryable as a living knowledge graph.
   */
  public append(entry: Omit<CompoundingEntry, 'timestamp'>): CompoundingEntry {
    const stamped: CompoundingEntry = {
      ...entry,
      timestamp: new Date().toISOString(),
    };
    const line = JSON.stringify(stamped) + '\n';
    fs.appendFileSync(this.logPath, line, 'utf8');

    // Fire-and-forget graph write — never blocks the caller
    if (this.ruVector) {
      void this.writeToGraph(stamped);
    }

    return stamped;
  }

  /**
   * Read all entries from the log.
   * Returns entries in chronological order (oldest first).
   */
  public readAll(): CompoundingEntry[] {
    if (!fs.existsSync(this.logPath)) return [];
    const content = fs.readFileSync(this.logPath, 'utf8').trim();
    if (!content) return [];
    return content
      .split('\n')
      .filter(line => line.trim())
      .map(line => {
        try {
          return JSON.parse(line) as CompoundingEntry;
        } catch {
          return null;
        }
      })
      .filter((e): e is CompoundingEntry => e !== null);
  }

  /**
   * Query the log for entries similar to the given query text.
   *
   * When a RuVectorClient is configured: uses hyperedge semantic search.
   * Fallback: keyword-overlap TF-IDF cosine (no external dependency).
   */
  public async querySimilar(queryText: string, topK = 5): Promise<QueryResult[]> {
    if (this.ruVector) {
      return this.queryViaRuVector(queryText, topK);
    }
    return this.queryViaTfIdf(queryText, topK);
  }

  /**
   * Return all entries where the test_result is PASS for a given trigger type.
   * Used by the SLM to find successful prior repairs as positive examples.
   */
  public getSuccessfulRepairs(trigger?: TriggerType): CompoundingEntry[] {
    return this.readAll().filter(
      e =>
        e.test_result === 'PASS' &&
        e.meta_review === 'APPROVED' &&
        (trigger === undefined || e.trigger === trigger)
    );
  }

  /**
   * Return all entries where the test_result is FAIL.
   * Used by the SLM as negative examples to avoid repeating failed approaches.
   */
  public getFailedRepairs(trigger?: TriggerType): CompoundingEntry[] {
    return this.readAll().filter(
      e =>
        e.test_result === 'FAIL' &&
        (trigger === undefined || e.trigger === trigger)
    );
  }

  /**
   * Scan the log for contradictions: entries where the same test has both
   * PASS and FAIL results. Indicates flaky tests or non-deterministic patches.
   */
  public scanContradictions(): Array<{ test: string; passCount: number; failCount: number }> {
    const entries = this.readAll();
    const byTest = new Map<string, { pass: number; fail: number }>();

    for (const entry of entries) {
      const existing = byTest.get(entry.test) ?? { pass: 0, fail: 0 };
      if (entry.test_result === 'PASS') existing.pass++;
      else if (entry.test_result === 'FAIL') existing.fail++;
      byTest.set(entry.test, existing);
    }

    return Array.from(byTest.entries())
      .filter(([, counts]) => counts.pass > 0 && counts.fail > 0)
      .map(([test, counts]) => ({
        test,
        passCount: counts.pass,
        failCount: counts.fail,
      }));
  }

  /**
   * Return summary statistics for the Meta-Agent health dashboard.
   */
  public getStats(): {
    total: number;
    passRate: number;
    avgSlmConfidence: number;
    pendingReviews: number;
  } {
    const entries = this.readAll();
    if (entries.length === 0) {
      return { total: 0, passRate: 0, avgSlmConfidence: 0, pendingReviews: 0 };
    }
    const passed = entries.filter(e => e.test_result === 'PASS').length;
    const avgConf =
      entries.reduce((sum, e) => sum + e.slm_confidence, 0) / entries.length;
    const pending = entries.filter(e => e.meta_review === 'PENDING').length;
    return {
      total: entries.length,
      passRate: passed / entries.length,
      avgSlmConfidence: Math.round(avgConf * 1000) / 1000,
      pendingReviews: pending,
    };
  }

  // ── RuVector integration ───────────────────────────────────────────────────

  /**
   * Write a compounding entry to RuVector as a node + hyperedge.
   * Node ID = `log:<timestamp>` for uniqueness.
   * Hyperedge connects the node to all files in graph_delta.modified.
   */
  private async writeToGraph(entry: CompoundingEntry): Promise<void> {
    if (!this.ruVector) return;
    const nodeId = `log:${entry.timestamp}`;
    const embedding = this.buildEmbedding(
      `${entry.test} ${entry.diagnosis} ${entry.trigger}`
    );

    await this.ruVector.upsertNode({
      id: nodeId,
      labels: ['LogEntry', entry.test_result],
      embedding,
      properties: {
        trigger: entry.trigger,
        test: entry.test,
        test_result: entry.test_result,
        meta_review: entry.meta_review,
        slm_confidence: String(entry.slm_confidence),
        patch_hash: entry.patch_hash,
        timestamp: entry.timestamp,
      },
    });

    // Create a hyperedge connecting this log entry to all modified files
    const modifiedNodes = entry.graph_delta.modified.filter(id => id.length > 0);
    if (modifiedNodes.length > 0) {
      await this.ruVector.createHyperedge({
        nodes: [nodeId, ...modifiedNodes],
        description: `${entry.trigger}: ${entry.diagnosis.slice(0, 120)}`,
        embedding,
        confidence: entry.slm_confidence,
        metadata: { test_result: entry.test_result },
      }).catch(() => {
        // Nodes in graph_delta may not exist in the graph yet — that is fine
      });
    }
  }

  /**
   * Semantic search via RuVector hyperedge similarity.
   * Falls back to TF-IDF if no results are returned.
   */
  private async queryViaRuVector(
    queryText: string,
    topK: number
  ): Promise<QueryResult[]> {
    if (!this.ruVector) return this.queryViaTfIdf(queryText, topK);

    const embedding = this.buildEmbedding(queryText);
    const hits = await this.ruVector.searchHyperedges(embedding, topK);

    if (hits.length === 0) {
      return this.queryViaTfIdf(queryText, topK);
    }

    const entries = this.readAll();
    const results: QueryResult[] = [];

    for (const hit of hits) {
      // hit.id is the hyperedge ID — find the log entry by timestamp
      const ts = hit.id.replace(/^log:/, '');
      const entry = entries.find(e => e.timestamp === ts);
      if (entry) {
        results.push({ entry, similarity: hit.score });
      }
    }

    // Pad with TF-IDF results if RuVector returned fewer than topK
    if (results.length < topK) {
      const tfidfResults = this.queryViaTfIdf(queryText, topK - results.length);
      const seen = new Set(results.map(r => r.entry.timestamp));
      for (const r of tfidfResults) {
        if (!seen.has(r.entry.timestamp)) results.push(r);
      }
    }

    return results.slice(0, topK);
  }

  // ── TF-IDF fallback ────────────────────────────────────────────────────────

  private queryViaTfIdf(queryText: string, topK: number): QueryResult[] {
    const entries = this.readAll();
    if (entries.length === 0) return [];

    const queryTokens = this.tokenize(queryText);
    const scored = entries.map(entry => ({
      entry,
      similarity: this.cosineSimilarity(
        queryTokens,
        this.tokenize(`${entry.test} ${entry.diagnosis} ${entry.trigger}`)
      ),
    }));

    return scored
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  // ── Embedding helpers ──────────────────────────────────────────────────────

  /**
   * Deterministic 128-dim unit-vector embedding from text char codes.
   * Matches the strategy used by ASTIndexer so embeddings are comparable.
   */
  private buildEmbedding(text: string): Float32Array {
    const dims = 128;
    const raw = new Float32Array(dims).fill(0);
    for (let i = 0; i < text.length; i++) {
      raw[i % dims] = (raw[i % dims] ?? 0) + text.charCodeAt(i);
    }
    const magnitude = Math.sqrt(raw.reduce((sum, v) => sum + v * v, 0)) || 1;
    return raw.map(v => v / magnitude);
  }

  private tokenize(text: string): Map<string, number> {
    const tokens = text.toLowerCase().split(/\W+/).filter(t => t.length > 2);
    const freq = new Map<string, number>();
    for (const token of tokens) {
      freq.set(token, (freq.get(token) ?? 0) + 1);
    }
    return freq;
  }

  private cosineSimilarity(
    a: Map<string, number>,
    b: Map<string, number>
  ): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (const [token, countA] of a) {
      const countB = b.get(token) ?? 0;
      dot += countA * countB;
      normA += countA * countA;
    }
    for (const countB of b.values()) {
      normB += countB * countB;
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}
