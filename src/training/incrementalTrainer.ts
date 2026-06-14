/**
 * IncrementalTrainer
 *
 * Catches corpus_ready_for_training events and runs the fine-tune
 * pipeline on new examples only, using the previous model as a
 * starting checkpoint (incremental fine-tuning, not full retrain).
 *
 * Each run takes minutes not hours because it only trains on the delta.
 * After training, updates the Ollama model weights automatically.
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import * as path from 'path';

export interface TrainerConfig {
  scriptPath: string;
  corpusPath: string;
  outputPath: string;
  ollamaModelName?: string;
  modelfilePath?: string;
}

type ExecFunction = (
  cmd: string,
  callback: (error: Error | null, stdout: string, stderr: string) => void
) => void;

export interface TrainingResult {
  success: boolean;
  durationMs: number;
  error?: string;
}

export class IncrementalTrainer {
  private readonly config: TrainerConfig;
  private execFn: ExecFunction;

  constructor(config: TrainerConfig) {
    this.config = config;
    this.execFn = this.defaultExec;
  }

  /**
   * Inject a custom exec function for testing without spawning real processes.
   */
  public setExecFunction(execFn: ExecFunction): void {
    this.execFn = execFn;
  }

  /**
   * Run the incremental fine-tune pipeline.
   * Executes finetunePipeline.py --cpu, then refreshes the Ollama model.
   */
  public async run(): Promise<TrainingResult> {
    const start = Date.now();

    try {
      await this.runPythonPipeline();
      await this.refreshOllamaModel();
      return { success: true, durationMs: Date.now() - start };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, durationMs: Date.now() - start, error: message };
    }
  }

  // ── Private steps ────────────────────────────────────────────────

  private runPythonPipeline(): Promise<void> {
    const cmd = this.buildPythonCommand();
    return this.execAsync(cmd);
  }

  private refreshOllamaModel(): Promise<void> {
    const modelName = this.config.ollamaModelName ?? 'claims-slm';
    const modelfilePath = this.config.modelfilePath
      ?? path.join(path.dirname(this.config.scriptPath), 'Modelfile');
    const cmd = `ollama create ${modelName} -f ${modelfilePath}`;
    return this.execAsync(cmd);
  }

  private buildPythonCommand(): string {
    const parts = [
      'python',
      this.config.scriptPath,
      `--corpus ${this.config.corpusPath}`,
      `--output ${this.config.outputPath}`,
      '--cpu'
    ];
    return parts.join(' ');
  }

  private execAsync(cmd: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.execFn(cmd, (error) => {
        if (error) reject(error);
        else resolve();
      });
    });
  }

  private defaultExec: ExecFunction = (cmd, callback) => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { exec } = require('child_process') as typeof import('child_process');
    exec(cmd, (error, stdout, stderr) => callback(error, stdout, stderr));
  };
}
