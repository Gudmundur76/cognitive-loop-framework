/**
 * Training Module — Public Entry Point
 *
 * Exports the autonomous training flywheel components:
 * - ClaimsCorpusGenerator: converts verified claim events into JSONL training pairs
 * - CorpusWatcher: monitors the corpus file and triggers training when ready
 * - IncrementalTrainer: runs fine-tuning jobs via Ollama or OpenAI
 *
 * Usage:
 *   import { createTrainingPipeline } from './training/index.js';
 *   const pipeline = createTrainingPipeline({ corpusPath: '/data/corpus.jsonl' });
 *   pipeline.generator.processVerdictEvent(event);
 */
export { ClaimsCorpusGenerator } from './claimsCorpusGenerator.js';
export type {
  VerdictEvent,
  ContradictionEvent,
  EntityRecord,
  ClaimsTrainingPair,
} from './claimsCorpusGenerator.js';

export { CorpusWatcher } from './corpusWatcher.js';
export type { CorpusReadyStats, ReadyCallback } from './corpusWatcher.js';

export { IncrementalTrainer } from './incrementalTrainer.js';
export type { TrainerConfig, TrainingResult } from './incrementalTrainer.js';

export { SIADatasetGenerator } from './siaDatasetGenerator.js';
export type {
  SIAPublicRecord,
  SIAGroundTruthRecord,
  SIADatasetConfig,
} from './siaDatasetGenerator.js';

import { ClaimsCorpusGenerator } from './claimsCorpusGenerator.js';
import { CorpusWatcher } from './corpusWatcher.js';
import { IncrementalTrainer } from './incrementalTrainer.js';

export interface TrainingPipelineConfig {
  /** Absolute path to the JSONL corpus file. */
  corpusPath?: string;
  /** Minimum number of training pairs before a training run is triggered. */
  minPairsThreshold?: number;
  /** Absolute path to the finetunePipeline.py script. */
  scriptPath?: string;
  /** Absolute path to write the fine-tuned model output. */
  outputPath?: string;
  /** Ollama model name to fine-tune. */
  ollamaModelName?: string;
}

export interface TrainingPipeline {
  generator: ClaimsCorpusGenerator;
  watcher: CorpusWatcher;
  trainer: IncrementalTrainer;
}

/**
 * Factory function that assembles the full training pipeline.
 * Returns the generator, watcher, and trainer as a coordinated unit.
 *
 * The watcher is wired to trigger the trainer automatically when the
 * corpus reaches the minimum threshold.
 */
export function createTrainingPipeline(
  config: TrainingPipelineConfig = {}
): TrainingPipeline {
  const corpusPath =
    config.corpusPath ??
    process.env['TRAINING_CORPUS_PATH'] ??
    '/data/training/claims_corpus.jsonl';

  const generator = new ClaimsCorpusGenerator(corpusPath);

  const trainer = new IncrementalTrainer({
    corpusPath,
    scriptPath:
      config.scriptPath ??
      process.env['FINETUNE_SCRIPT_PATH'] ??
      '/opt/cognitive-loop/scripts/finetunePipeline.py',
    outputPath:
      config.outputPath ??
      process.env['TRAINING_OUTPUT_PATH'] ??
      '/data/training/output',
    ollamaModelName:
      config.ollamaModelName ??
      process.env['TRAINING_MODEL'] ??
      'qwen2.5-coder:7b',
  });

  const watcher = new CorpusWatcher(
    corpusPath,
    config.minPairsThreshold ?? 50
  );

  // Wire the watcher to trigger the trainer automatically
  watcher.onReady(async (_stats) => {
    await trainer.run();
  });

  return { generator, watcher, trainer };
}
