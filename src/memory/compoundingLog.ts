/**
 * CompoundingLog
 *
 * Implements the structured memory log described in self_improvement_mechanics.md.
 * Each entry is a JSON record capturing a full repair/training cycle:
 *   trigger → diagnosis → patch → test_result → graph_delta → meta_review
 *
 * The log is NOT a diary. It is a training corpus for the next repair.
 * Future repairs query it via cosine similarity (vector search) to find
 * prior successful patches for similar failures.
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */
import * as fs from 'fs';
import * as path from 'path';

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
  /** Cosine similarity score (0.0–1.0) — higher = more similar. */
  similarity: number;
}

// ── CompoundingLog ────────────────────────────────────────────────────────────
export class CompoundingLog {
  private readonly logPath: string;

  constructor(logPath: string) {
    this.logPath = logPath;
    fs.mkdirSync(path.dirname(logPath), { recursive: true });
  }

  /**
   * Append a new entry to the compounding log.
   * Each entry is written as a single JSON line (JSONL format).
   */
  public append(entry: Omit<CompoundingEntry, 'timestamp'>): CompoundingEntry {
    const stamped: CompoundingEntry = {
      ...entry,
      timestamp: new Date().toISOString(),
    };
    const line = JSON.stringify(stamped) + '\n';
    fs.appendFileSync(this.logPath, line, 'utf8');
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
   * Uses a simple keyword-overlap similarity (no external vector DB required).
   * For production, replace with RuVector hybrid query.
   */
  public querySimilar(queryText: string, topK = 5): QueryResult[] {
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

  // ── Private helpers ────────────────────────────────────────────────────────
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
