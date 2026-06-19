/**
 * CorpusWatcher
 *
 * Monitors the ttruthdesk claims DB for new verified claims. When the
 * number of new claims since the last training run reaches the threshold,
 * emits a corpus_ready_for_training callback after a 5-minute debounce
 * (to batch multiple rapid claims into a single training run).
 *
 * Also supports the legacy file-based check() for backwards compatibility.
 *
 * Acceptance: After 50 new claims are added to ttruthdesk, the watcher
 * emits the event within 6 minutes.
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import * as fs from 'fs';
import { countNewVerifiedClaims } from './ttruthdeskBridge.js';

export interface CorpusReadyStats {
  newExamplesCount: number;
  totalExamples: number;
}

export type ReadyCallback = (stats: CorpusReadyStats) => void;

const DEBOUNCE_MS = 5 * 60 * 1000; // 5 minutes

export class CorpusWatcher {
  private readonly corpusPath: string;
  private readonly threshold: number;
  private baselineCount: number;
  private readyCallback: ReadyCallback | null;
  /** Timestamp of the last completed training run */
  private lastTrainingAt: Date;
  /** Timer handle for the 5-minute debounce */
  private debounceTimer: ReturnType<typeof setTimeout> | null;

  constructor(corpusPath: string, threshold: number) {
    this.corpusPath = corpusPath;
    this.threshold = threshold;
    this.baselineCount = this.countLines();
    this.readyCallback = null;
    this.lastTrainingAt = new Date(0);
    this.debounceTimer = null;
  }

  /**
   * Register a callback to be invoked when the corpus is ready for training.
   * Only one callback is supported — calling again replaces the previous one.
   */
  public onReady(callback: ReadyCallback): void {
    this.readyCallback = callback;
  }

  /**
   * DB-backed check: query the ttruthdesk claims table for new verified
   * claims since the last training run. When count >= threshold, schedule
   * the ready callback after the 5-minute debounce.
   *
   * Call this periodically (e.g., every 5 minutes via setInterval).
   */
  public async checkDb(): Promise<void> {
    const newCount = await countNewVerifiedClaims(this.lastTrainingAt);
    if (newCount < this.threshold) return;
    if (this.debounceTimer !== null) return; // already scheduled

    console.log(
      `[CorpusWatcher] ${newCount} new verified claims detected — ` +
      `triggering training in 5 minutes`
    );

    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      this.lastTrainingAt = new Date();
      const total = this.countLines();
      this.readyCallback?.({ newExamplesCount: newCount, totalExamples: total });
    }, DEBOUNCE_MS);
  }

  /**
   * Legacy file-based check: count JSONL lines in the corpus file.
   * Kept for backwards compatibility with existing tests and the
   * createTrainingPipeline factory.
   */
  public check(): void {
    const total = this.countLines();
    const newCount = total - this.baselineCount;
    if (newCount >= this.threshold) {
      this.baselineCount = total;
      this.readyCallback?.({ newExamplesCount: newCount, totalExamples: total });
    }
  }

  /**
   * Update the lastTrainingAt timestamp after a successful training run.
   * Call this from the IncrementalTrainer's onReady callback.
   */
  public markTrainingComplete(): void {
    this.lastTrainingAt = new Date();
    this.baselineCount = this.countLines();
  }

  /**
   * Cancel any pending debounce timer. Call on shutdown.
   */
  public destroy(): void {
    if (this.debounceTimer !== null) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────

  private countLines(): number {
    if (!fs.existsSync(this.corpusPath)) return 0;
    const content = fs.readFileSync(this.corpusPath, 'utf8').trim();
    if (content.length === 0) return 0;
    return content.split('\n').length;
  }
}
