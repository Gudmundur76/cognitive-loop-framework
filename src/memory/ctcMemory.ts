/**
 * ctcMemory.ts — MRAgent Cue-Tag-Content memory layer for cognitive-loop-framework
 *
 * Replaces the flat JSONL compounding log with a CTC graph that supports:
 *   - Active reconstruction: answer "what happened when X failed?" via iterative traversal
 *   - Temporal queries: "what changed in the last 7 days?"
 *   - Causal chains: "what was the root cause of the auth regression?"
 *   - Learning extraction: "what patterns led to successful repairs?"
 *
 * Architecture:
 *   - CTC graph stored in SQLite via Python sidecar (evolva_mragent)
 *   - TypeScript bridge calls sidecar for write (ingest) and read (reconstruct)
 *   - Backward compatible: existing CompoundingLog still writes JSONL as fallback
 *
 * Paper: "Memory is Reconstructed, Not Retrieved" (Ji, Li, Hooi — ICML 2026)
 */

import { spawn } from 'child_process';
import { join } from 'path';
import { homedir } from 'os';
import { existsSync, mkdirSync } from 'fs';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CycleRecord {
  /** Unique cycle identifier, e.g. "cycle-2026-06-28T14:32:00Z" */
  cycle_id: string;
  /** ISO timestamp of cycle start */
  timestamp: string;
  /** The task or claim being processed */
  task: string;
  /** Phase outputs: { Observe: "...", Think: "...", Plan: "...", Act: "...", Verify: "..." } */
  phases: Record<string, string>;
  /** Final outcome: "success" | "failure" | "partial" | "blocked" */
  outcome: string;
  /** What was learned from this cycle (optional) */
  learned?: string;
  /** Test name if this was a repair cycle */
  test_name?: string;
  /** Error output if this was a repair cycle */
  error_output?: string;
}

export interface ReconstructionResult {
  question: string;
  answer: string;
  supports: string[];
  confidence: 'high' | 'medium' | 'low';
  reasoning: string;
  tool_calls_made: number;
  rounds: number;
  evidence_texts: string[];
}

export interface TemporalQueryResult {
  start_date: string;
  end_date: string;
  events: Array<{
    event_id: string;
    text: string;
    origin: string;
    time: string;
    domain: string;
  }>;
  count: number;
}

// ── Paths ─────────────────────────────────────────────────────────────────────

const EVOLVA_MRAGENT_PATH = join(homedir(), 'evolva-mragent');
const CTC_DB_PATH = join(homedir(), '.codebase-memory', 'ctc_clf_graph.db');
const SIDECAR_PATH = join(EVOLVA_MRAGENT_PATH, 'integrations', 'cognitive-loop-framework', 'ctc_clf_sidecar.py');
const PYTHON = process.env['PYTHON_BIN'] ?? 'python3';

// ── Sidecar communication ─────────────────────────────────────────────────────

async function callSidecar<T>(method: string, args: Record<string, unknown>): Promise<T> {
  return new Promise((resolve, reject) => {
    const input = JSON.stringify({ method, args });
    const proc = spawn(PYTHON, [SIDECAR_PATH], {
      env: { ...process.env, PYTHONPATH: EVOLVA_MRAGENT_PATH },
    });

    let stdout = '';
    let stderr = '';

    proc.stdin.write(input + '\n');
    proc.stdin.end();
    proc.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
    proc.stderr.on('data', (d: Buffer) => (stderr += d.toString()));

    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error('CTC sidecar timeout (60s)'));
    }, 60_000);

    proc.on('close', (code: number) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`CTC sidecar exited ${code}: ${stderr.slice(0, 500)}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()) as T);
      } catch (e) {
        reject(new Error(`CTC sidecar JSON parse error: ${stdout.slice(0, 200)}`));
      }
    });

    proc.on('error', reject);
  });
}

// ── CTCMemory class ───────────────────────────────────────────────────────────

export class CTCMemory {
  private readonly dbPath: string;
  private readonly enabled: boolean;

  constructor(dbPath = CTC_DB_PATH) {
    this.dbPath = dbPath;
    this.enabled = existsSync(EVOLVA_MRAGENT_PATH) && existsSync(SIDECAR_PATH);
    if (!this.enabled) {
      console.warn('[CTCMemory] evolva-mragent not found — CTC memory disabled');
    }
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Ingest a cognitive cycle record into the CTC graph.
   * Runs the 3-stage MRAgent pipeline: rewrite → keywords → store.
   * Non-blocking: errors are logged but do not throw.
   */
  async ingestCycle(cycle: CycleRecord): Promise<void> {
    if (!this.enabled) return;
    try {
      await callSidecar<{ ok: boolean }>('ingest_cycle', {
        cycle,
        db_path: this.dbPath,
      });
    } catch (e) {
      console.error('[CTCMemory] ingestCycle error:', (e as Error).message);
    }
  }

  /**
   * Run the MRAgent active reconstruction loop to answer a question
   * about past cognitive cycles.
   *
   * Examples:
   *   "What was the root cause of the auth test failure on June 20?"
   *   "What repair strategies have succeeded for TypeScript type errors?"
   *   "What did the agent learn about the citation pipeline last week?"
   */
  async reconstruct(question: string): Promise<ReconstructionResult> {
    if (!this.enabled) {
      return {
        question,
        answer: 'CTC memory not available.',
        supports: [],
        confidence: 'low',
        reasoning: 'evolva-mragent not installed',
        tool_calls_made: 0,
        rounds: 0,
        evidence_texts: [],
      };
    }
    try {
      return await callSidecar<ReconstructionResult>('reconstruct', {
        question,
        domain: 'cognitive_loop',
        db_path: this.dbPath,
      });
    } catch (e) {
      return {
        question,
        answer: `Reconstruction failed: ${(e as Error).message}`,
        supports: [],
        confidence: 'low',
        reasoning: (e as Error).message,
        tool_calls_made: 0,
        rounds: 0,
        evidence_texts: [],
      };
    }
  }

  /**
   * Query events in a date range.
   * Useful for "what happened this week?" style queries.
   */
  async temporalQuery(startDate: string, endDate: string): Promise<TemporalQueryResult> {
    if (!this.enabled) {
      return { start_date: startDate, end_date: endDate, events: [], count: 0 };
    }
    try {
      return await callSidecar<TemporalQueryResult>('temporal_query', {
        start_date: startDate,
        end_date: endDate,
        db_path: this.dbPath,
      });
    } catch (e) {
      return { start_date: startDate, end_date: endDate, events: [], count: 0 };
    }
  }

  /**
   * Get all cues (keywords) associated with an event.
   * Useful for exploring what a specific cycle was about.
   */
  async getEventKeywords(eventId: string): Promise<Array<{ key: string; tags: string[] }>> {
    if (!this.enabled) return [];
    try {
      const result = await callSidecar<{ keywords: Array<{ key: string; tags: string[] }> }>(
        'event_keywords',
        { event_id: eventId, db_path: this.dbPath }
      );
      return result.keywords ?? [];
    } catch {
      return [];
    }
  }

  /**
   * Follow a cue→tag→content edge.
   * The primary traversal primitive for the MRAgent loop.
   */
  async edgesByTag(
    key: string,
    tag: string
  ): Promise<Array<{ event_id: string; text: string; origin: string; time: string }>> {
    if (!this.enabled) return [];
    try {
      const result = await callSidecar<{
        events: Array<{ event_id: string; text: string; origin: string; time: string }>;
      }>('edges_by_tag', { key, tag, db_path: this.dbPath });
      return result.events ?? [];
    } catch {
      return [];
    }
  }
}

// ─── Singleton ────────────────────────────────────────────────────────────────
export const ctcMemory = new CTCMemory();
