/**
 * Loop Module — Public Entry Point
 *
 * Exports the full cognitive loop system:
 * - LoopOrchestrator: runs the five-layer loop
 * - MetaAgent: monitors health and publishes events
 * - ManusDispatcher: dispatches repair tasks to Manus API
 *
 * Usage:
 *   import { createLoop } from './loop/index.js';
 *   const loop = createLoop({ manusApiKey: '...', dryRun: true });
 *   const result = await loop.run({ filePaths: ['src/...'] });
 */

export { LoopOrchestrator } from './loopOrchestrator.js';
export type { LoopInput, LoopResult, LayerResult, LayerName, OrchestratorConfig } from './loopOrchestrator.js';

export { MetaAgent } from './metaAgent.js';
export type { SystemEvent, EventType, EventSeverity, HealthAssessment } from './metaAgent.js';

export { ManusDispatcher } from './manusDispatcher.js';
export type { ManusTask, ManusApiConfig, DispatchResult } from './manusDispatcher.js';

import { MemoryLayer } from '../memory/index.js';
import { SelfPromptEngine } from '../slm/selfPromptEngine.js';
import { MetaAgent } from './metaAgent.js';
import { LoopOrchestrator } from './loopOrchestrator.js';
import { ManusDispatcher } from './manusDispatcher.js';

export interface CognitiveLoopConfig {
  manusApiKey?: string;
  manusProjectId?: string;
  dryRun?: boolean;
  maxIterations?: number;
  ollamaUrl?: string;
  ollamaModel?: string;
  fallbackToOpenAI?: boolean;
}

/**
 * Factory function that assembles the full cognitive loop system.
 * Returns the orchestrator and dispatcher as a paired unit.
 */
export function createLoop(config: CognitiveLoopConfig = {}): {
  orchestrator: LoopOrchestrator;
  dispatcher: ManusDispatcher;
  meta: MetaAgent;
} {
  const memory = new MemoryLayer();
  const slm = new SelfPromptEngine({
    ollamaUrl: config.ollamaUrl,
    ollamaModel: config.ollamaModel,
    fallbackToOpenAI: config.fallbackToOpenAI ?? true
  });
  const meta = new MetaAgent();
  const orchestrator = new LoopOrchestrator(memory, slm, meta, {
    maxIterations: config.maxIterations ?? 10
  });
  const dispatcher = new ManusDispatcher(meta, {
    apiKey: config.manusApiKey ?? process.env['MANUS_API_KEY'] ?? '',
    projectId: config.manusProjectId ?? process.env['MANUS_PROJECT_ID'],
    dryRun: config.dryRun ?? !config.manusApiKey
  });

  return { orchestrator, dispatcher, meta };
}
