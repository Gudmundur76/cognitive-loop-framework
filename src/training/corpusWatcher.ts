/**
 * CorpusWatcher
 *
 * Monitors the JSONL corpus file for growth. When the number of new
 * training examples since the last training run exceeds the threshold,
 * fires a corpus_ready_for_training callback.
 *
 * This prevents continuous retraining on every single claim while
 * ensuring the model stays current as new claims accumulate.
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import * as fs from 'fs';

export interface CorpusReadyStats {
  newExamplesCount: number;
  totalExamples: number;
}

export type ReadyCallback = (stats: CorpusReadyStats) => void;

export class CorpusWatcher {
  private readonly corpusPath: string;
  private readonly threshold: number;
  private baselineCount: number;
  private readyCallback: ReadyCallback | null;

  constructor(corpusPath: string, threshold: number) {
    this.corpusPath = corpusPath;
    this.threshold = threshold;
    this.baselineCount = this.countLines();
    this.readyCallback = null;
  }

  /**
   * Register a callback to be invoked when the corpus is ready for training.
   * Only one callback is supported — calling again replaces the previous one.
   */
  public onReady(callback: ReadyCallback): void {
    this.readyCallback = callback;
  }

  /**
   * Check whether the corpus has grown enough to trigger training.
   * Call this periodically (e.g., from a cron job or after each event).
   */
  public check(): void {
    const total = this.countLines();
    const newCount = total - this.baselineCount;

    if (newCount >= this.threshold) {
      this.baselineCount = total;
      this.readyCallback?.({ newExamplesCount: newCount, totalExamples: total });
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
