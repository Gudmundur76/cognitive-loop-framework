/**
 * MetaAgent — L4 of the Cognitive Loop
 *
 * Responsibilities:
 * - Maintain an internal event queue for system-level signals
 * - Assess the health of each loop iteration from layer results
 * - Publish `system_capability_required` events when failures are detected
 * - Drain the event queue for the LoopOrchestrator and ManusDispatcher
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 * - No external dependencies — pure TypeScript
 */

import type { LayerResult } from './loopOrchestrator.js';

export type EventType =
  | 'system_capability_required'
  | 'knowledge_gap_detected'
  | 'verdict_flip_detected'
  | 'health_degraded'
  | 'loop_converged';

export type EventSeverity = 'info' | 'warning' | 'critical';

export interface SystemEvent {
  type: EventType;
  severity: EventSeverity;
  message: string;
  filePath: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface HealthAssessment {
  score: number;       // 0.0 – 1.0
  status: 'healthy' | 'degraded' | 'critical';
  failingLayers: string[];
}

export class MetaAgent {
  private readonly eventQueue: SystemEvent[] = [];
  private readonly maxQueueSize: number;

  constructor(maxQueueSize = 100) {
    this.maxQueueSize = maxQueueSize;
  }

  /**
   * Publish a system event to the internal queue.
   * Oldest events are dropped when the queue is full.
   */
  public publishEvent(event: SystemEvent): void {
    const stamped: SystemEvent = {
      ...event,
      timestamp: new Date().toISOString()
    };
    if (this.eventQueue.length >= this.maxQueueSize) {
      this.eventQueue.shift();
    }
    this.eventQueue.push(stamped);
  }

  /**
   * Drain all queued events and return them.
   * Clears the internal queue after draining.
   */
  public drainEvents(): SystemEvent[] {
    return this.eventQueue.splice(0, this.eventQueue.length);
  }

  /**
   * Peek at queued events without draining.
   */
  public peekEvents(): readonly SystemEvent[] {
    return this.eventQueue;
  }

  /**
   * Assess the health of the current loop iteration.
   * Returns a score (0–1), a status label, and the list of failing layers.
   */
  public assessHealth(results: LayerResult[]): HealthAssessment {
    if (results.length === 0) {
      return { score: 0, status: 'critical', failingLayers: [] };
    }

    const failingLayers = results
      .filter(r => !r.passed)
      .map(r => r.layer);

    const score = (results.length - failingLayers.length) / results.length;

    const status = this.scoreToStatus(score, failingLayers);

    if (status === 'critical') {
      this.publishEvent({
        type: 'health_degraded',
        severity: 'critical',
        message: `Critical health: ${failingLayers.join(', ')} failed`,
        filePath: '',
        metadata: { score, failingLayers }
      });
    } else if (status === 'degraded') {
      this.publishEvent({
        type: 'health_degraded',
        severity: 'warning',
        message: `Degraded health: ${failingLayers.join(', ')} failed`,
        filePath: '',
        metadata: { score, failingLayers }
      });
    }

    return { score, status, failingLayers };
  }

  /**
   * Determine if a set of events contains a repair-triggering condition.
   * Used by ManusDispatcher to decide whether to spawn a development agent.
   */
  public requiresRepair(events: SystemEvent[]): boolean {
    return events.some(
      e =>
        e.type === 'system_capability_required' &&
        (e.severity === 'critical' || e.severity === 'warning')
    );
  }

  /**
   * Build a structured repair context string from a set of events.
   * This is the prompt context passed to the Manus development agent.
   */
  public buildRepairContext(events: SystemEvent[]): string {
    const repairEvents = events.filter(
      e => e.type === 'system_capability_required'
    );

    if (repairEvents.length === 0) return '';

    const lines = repairEvents.map(e =>
      `[${e.severity.toUpperCase()}] ${e.message}${e.filePath ? ` (${e.filePath})` : ''}`
    );

    return [
      'The following system capabilities are required:',
      ...lines,
      '',
      'Please diagnose each issue, write a fix, add a test, and open a PR.'
    ].join('\n');
  }

  // ── Private helpers ──────────────────────────────────────────────

  private scoreToStatus(
    score: number,
    failingLayers: string[]
  ): HealthAssessment['status'] {
    if (failingLayers.includes('friction') || score < 0.4) return 'critical';
    if (score < 0.8) return 'degraded';
    return 'healthy';
  }
}
