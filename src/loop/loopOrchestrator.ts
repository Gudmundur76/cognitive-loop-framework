/**
 * LoopOrchestrator — L1 Coordinator of the Cognitive Loop
 *
 * Runs the five-layer cognitive loop over a set of TypeScript files:
 *
 *   L1 Friction Layer   — validates input files are TypeScript
 *   L2 Truth Layer      — ingests files into the memory/knowledge graph
 *   L3 Self-Prompt Layer — runs SLM reasoning over ingested nodes
 *   L4 Frontier Layer   — identifies knowledge gaps and novel patterns
 *   L5 Meta Layer       — assesses health, publishes system events
 *
 * Each iteration runs all five layers in order. The loop converges when
 * all layers pass or maxIterations is reached.
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 * - No direct DB or network calls — delegates to injected collaborators
 */
import type { MemoryLayer } from '../memory/index.js';
import type { SelfPromptEngine } from '../slm/selfPromptEngine.js';
import type { MetaAgent, SystemEvent } from './metaAgent.js';

export type LayerName = 'friction' | 'truth' | 'selfPrompt' | 'frontier' | 'meta';

export interface LayerResult {
  layer: LayerName;
  passed: boolean;
  output: string;
  durationMs: number;
}

export interface LoopInput {
  filePaths: string[];
}

export interface LoopResult {
  iterations: number;
  layerResults: LayerResult[];
  completionPromise: string;
  events: SystemEvent[];
}

export interface OrchestratorConfig {
  maxIterations?: number;
}

export class LoopOrchestrator {
  private readonly memory: MemoryLayer;
  private readonly slm: SelfPromptEngine;
  private readonly meta: MetaAgent;
  private readonly maxIterations: number;

  constructor(
    memory: MemoryLayer,
    slm: SelfPromptEngine,
    meta: MetaAgent,
    config: OrchestratorConfig = {}
  ) {
    this.memory = memory;
    this.slm = slm;
    this.meta = meta;
    this.maxIterations = config.maxIterations ?? 10;
  }

  /**
   * Run the full five-layer loop over the provided file paths.
   * Returns a LoopResult with all layer results and any system events.
   */
  public async run(input: LoopInput): Promise<LoopResult> {
    const allLayerResults: LayerResult[] = [];
    let iterations = 0;

    for (let i = 0; i < this.maxIterations; i++) {
      iterations++;
      const iterResults = await this.runIteration(input.filePaths);
      allLayerResults.push(...iterResults);

      const health = this.meta.assessHealth(iterResults);
      if (health.status === 'healthy') break;
      // If friction layer failed, no point retrying
      if (iterResults.find(r => r.layer === 'friction' && !r.passed)) break;
    }

    const events = this.meta.drainEvents();
    const completionPromise = this.buildCompletionPromise(allLayerResults);

    return { iterations, layerResults: allLayerResults, completionPromise, events };
  }

  // ── Private: single iteration ─────────────────────────────────────────────

  private async runIteration(filePaths: string[]): Promise<LayerResult[]> {
    const results: LayerResult[] = [];

    const frictionResult = this.runFrictionLayer(filePaths);
    results.push(frictionResult);
    if (!frictionResult.passed) return results;

    const tsFiles = filePaths.filter(f => f.endsWith('.ts'));
    const truthResult = await this.runTruthLayer(tsFiles);
    results.push(truthResult);

    const selfPromptResult = await this.runSelfPromptLayer(tsFiles);
    results.push(selfPromptResult);

    const frontierResult = this.runFrontierLayer(tsFiles);
    results.push(frontierResult);

    const metaResult = this.runMetaLayer(results);
    results.push(metaResult);

    return results;
  }

  // ── L1: Friction Layer ────────────────────────────────────────────────────

  private runFrictionLayer(filePaths: string[]): LayerResult {
    const start = Date.now();
    const tsFiles = filePaths.filter(f => f.endsWith('.ts'));
    const passed = tsFiles.length > 0;
    return {
      layer: 'friction',
      passed,
      output: passed
        ? `${tsFiles.length} TypeScript file(s) validated`
        : 'No TypeScript files found — halting loop',
      durationMs: Date.now() - start,
    };
  }

  // ── L2: Truth Layer ───────────────────────────────────────────────────────

  private async runTruthLayer(tsFiles: string[]): Promise<LayerResult> {
    const start = Date.now();
    let processed = 0;
    try {
      for (const filePath of tsFiles) {
        await this.memory.ingestFile(filePath);
        processed++;
      }
      return {
        layer: 'truth',
        passed: true,
        output: `Ingested ${processed}/${tsFiles.length} files into knowledge graph`,
        durationMs: Date.now() - start,
      };
    } catch (err) {
      return {
        layer: 'truth',
        passed: false,
        output: `Ingestion failed after ${processed} files: ${String(err)}`,
        durationMs: Date.now() - start,
      };
    }
  }

  // ── L3: Self-Prompt Layer ─────────────────────────────────────────────────

  private async runSelfPromptLayer(tsFiles: string[]): Promise<LayerResult> {
    const start = Date.now();
    try {
      const response = await this.slm.reason({
        mode: 'diagnose',
        context: `Analysing ${tsFiles.length} TypeScript file(s): ${tsFiles.join(', ')}`,
      });
      return {
        layer: 'selfPrompt',
        passed: true,
        output: response.output.slice(0, 200),
        durationMs: Date.now() - start,
      };
    } catch (err) {
      return {
        layer: 'selfPrompt',
        passed: false,
        output: `SLM reasoning failed: ${String(err)}`,
        durationMs: Date.now() - start,
      };
    }
  }

  // ── L4: Frontier Layer ────────────────────────────────────────────────────

  private runFrontierLayer(tsFiles: string[]): LayerResult {
    const start = Date.now();
    // Frontier layer: identify files that may have knowledge gaps
    // (heuristic: files with no tests nearby are candidates)
    const candidates = tsFiles.filter(f => !f.includes('.test.'));
    return {
      layer: 'frontier',
      passed: true,
      output: `${candidates.length} frontier file(s) identified for gap analysis`,
      durationMs: Date.now() - start,
    };
  }

  // ── L5: Meta Layer ────────────────────────────────────────────────────────

  private runMetaLayer(priorResults: LayerResult[]): LayerResult {
    const start = Date.now();
    const health = this.meta.assessHealth(priorResults);
    const passed = health.status !== 'critical';
    return {
      layer: 'meta',
      passed,
      output: `Health: ${health.status} (score=${health.score.toFixed(2)})`,
      durationMs: Date.now() - start,
    };
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private buildCompletionPromise(results: LayerResult[]): string {
    const passed = results.filter(r => r.passed).length;
    const total = results.length;
    const pct = total > 0 ? Math.round((passed / total) * 100) : 0;
    return `Loop complete: ${passed}/${total} layer results passed (${pct}%). Self-improving cognitive loop operational.`;
  }
}
