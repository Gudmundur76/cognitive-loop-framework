/**
 * pipeline.test.ts
 *
 * Tests for the createTrainingPipeline factory — specifically the
 * markTrainingComplete() wiring added in this sprint:
 *
 *   - watcher.onReady fires trainer.run()
 *   - on success, watcher.markTrainingComplete() is called
 *   - on failure, watcher.markTrainingComplete() is NOT called
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { createTrainingPipeline } from '../../src/training/index.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function writePairs(filePath: string, count: number): void {
  const lines = Array.from({ length: count }, (_, i) =>
    JSON.stringify({
      instruction: `Verify claim ${i}`,
      input: '',
      output: JSON.stringify({ verdict: 'Supported', confidence: 0.9 }),
    })
  );
  fs.writeFileSync(filePath, lines.join('\n') + '\n', 'utf8');
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('createTrainingPipeline — markTrainingComplete wiring', () => {
  let tmpDir: string;
  let corpusPath: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'clf-pipeline-test-'));
    corpusPath = path.join(tmpDir, 'corpus.jsonl');
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('calls markTrainingComplete after a successful training run', async () => {
    // Create pipeline first (baseline = 0), then write pairs so newCount = 60
    const pipeline = createTrainingPipeline({
      corpusPath,
      minPairsThreshold: 50,
      scriptPath: '/fake/finetunePipeline.py',
      outputPath: path.join(tmpDir, 'output'),
    });

    // Write 60 pairs after pipeline creation so baseline stays 0 and newCount = 60
    writePairs(corpusPath, 60);

    // Stub trainer.run() to return success
    const runSpy = vi
      .spyOn(pipeline.trainer, 'run')
      .mockResolvedValue({ success: true, durationMs: 100 });

    // Spy on markTrainingComplete
    const markSpy = vi.spyOn(pipeline.watcher, 'markTrainingComplete');

    // Trigger the watcher manually
    pipeline.watcher.check();

    // Allow the async onReady callback to complete
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(runSpy).toHaveBeenCalledOnce();
    expect(markSpy).toHaveBeenCalledOnce();

    pipeline.watcher.destroy();
  });

  it('does NOT call markTrainingComplete when training fails', async () => {
    const pipeline = createTrainingPipeline({
      corpusPath,
      minPairsThreshold: 50,
      scriptPath: '/fake/finetunePipeline.py',
      outputPath: path.join(tmpDir, 'output'),
    });

    writePairs(corpusPath, 60);

    vi.spyOn(pipeline.trainer, 'run').mockResolvedValue({
      success: false,
      durationMs: 50,
      error: 'Python script exited with code 1',
    });

    const markSpy = vi.spyOn(pipeline.watcher, 'markTrainingComplete');

    pipeline.watcher.check();
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(markSpy).not.toHaveBeenCalled();

    pipeline.watcher.destroy();
  });

  it('does not re-trigger training on the same examples after markTrainingComplete', async () => {
    // Create pipeline first (baseline = 0), then write 60 pairs
    const pipeline = createTrainingPipeline({
      corpusPath,
      minPairsThreshold: 50,
      scriptPath: '/fake/finetunePipeline.py',
      outputPath: path.join(tmpDir, 'output'),
    });

    writePairs(corpusPath, 60);

    const runSpy = vi
      .spyOn(pipeline.trainer, 'run')
      .mockResolvedValue({ success: true, durationMs: 100 });

    // First trigger — 60 lines, baseline 0 → newCount 60 ≥ 50 → fires
    pipeline.watcher.check();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(runSpy).toHaveBeenCalledTimes(1);

    // check() already advances baselineCount to 60 before calling the callback.
    // markTrainingComplete() also sets baselineCount = 60 (same value).
    // A second check with no new lines → newCount = 0 < 50 → no re-trigger.
    pipeline.watcher.check();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(runSpy).toHaveBeenCalledTimes(1);

    pipeline.watcher.destroy();
  });

  it('re-triggers training after enough new examples are added post-completion', async () => {
    // Create pipeline first (baseline = 0), then write 60 pairs
    const pipeline = createTrainingPipeline({
      corpusPath,
      minPairsThreshold: 50,
      scriptPath: '/fake/finetunePipeline.py',
      outputPath: path.join(tmpDir, 'output'),
    });

    writePairs(corpusPath, 60);

    const runSpy = vi
      .spyOn(pipeline.trainer, 'run')
      .mockResolvedValue({ success: true, durationMs: 100 });

    // First training run — 60 lines fires, baseline advances to 60
    pipeline.watcher.check();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(runSpy).toHaveBeenCalledTimes(1);

    // Add 55 more examples → total 115, newCount = 115 - 60 = 55 ≥ 50
    writePairs(corpusPath, 115);

    // Second trigger — should fire again
    pipeline.watcher.check();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(runSpy).toHaveBeenCalledTimes(2);

    pipeline.watcher.destroy();
  });

  it('returns all three pipeline components', () => {
    const pipeline = createTrainingPipeline({ corpusPath });
    expect(pipeline.generator).toBeDefined();
    expect(pipeline.watcher).toBeDefined();
    expect(pipeline.trainer).toBeDefined();
    pipeline.watcher.destroy();
  });
});
