/**
 * ManusDispatcher
 *
 * The bridge between the cognitive loop and the Manus platform.
 * When the MetaAgent publishes a `system_capability_required` event,
 * the ManusDispatcher catches it and spawns a Manus Development Agent
 * to diagnose, fix, and open a PR for the failing component.
 *
 * This closes the self-building loop:
 *   MetaAgent detects failure
 *     → ManusDispatcher dispatches repair task
 *       → Manus agent clones repo, writes fix, opens PR
 *         → Loop re-runs and verifies the fix
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 * - All external calls are mockable via constructor injection
 */

import type { SystemEvent } from './metaAgent.js';
import { MetaAgent } from './metaAgent.js';

export interface ManusTask {
  id: string;
  title: string;
  prompt: string;
  status: 'dispatched' | 'acknowledged' | 'failed';
  dispatchedAt: string;
}

export interface ManusApiConfig {
  apiKey: string;
  baseUrl?: string;
  projectId?: string;
  dryRun?: boolean;
}

export interface DispatchResult {
  dispatched: number;
  tasks: ManusTask[];
  skipped: number;
  dryRun: boolean;
}

const DEFAULT_BASE_URL = 'https://api.manus.im/v1';

export class ManusDispatcher {
  private readonly config: Required<ManusApiConfig>;
  private readonly meta: MetaAgent;
  private readonly dispatchHistory: ManusTask[] = [];

  constructor(meta: MetaAgent, config: ManusApiConfig) {
    this.meta = meta;
    this.config = {
      baseUrl: DEFAULT_BASE_URL,
      projectId: '',
      dryRun: false,
      ...config
    };
  }

  /**
   * Process a batch of system events.
   * For each event that requires repair, dispatch a Manus development agent.
   * Returns a summary of what was dispatched.
   */
  public async processEvents(events: SystemEvent[]): Promise<DispatchResult> {
    const repairEvents = events.filter(
      e => e.type === 'system_capability_required'
    );

    if (repairEvents.length === 0) {
      return { dispatched: 0, tasks: [], skipped: events.length, dryRun: this.config.dryRun };
    }

    const tasks: ManusTask[] = [];
    for (const event of repairEvents) {
      const task = await this.dispatchRepairTask(event);
      if (task) tasks.push(task);
    }

    return {
      dispatched: tasks.length,
      tasks,
      skipped: events.length - repairEvents.length,
      dryRun: this.config.dryRun
    };
  }

  /**
   * Return the full dispatch history for this session.
   */
  public getHistory(): readonly ManusTask[] {
    return this.dispatchHistory;
  }

  /**
   * Check whether a repair task has already been dispatched for a given file.
   * Prevents duplicate repair tasks for the same failure.
   */
  public isAlreadyDispatched(filePath: string): boolean {
    return this.dispatchHistory.some(t => t.prompt.includes(filePath));
  }

  // ── Private helpers ──────────────────────────────────────────────

  private async dispatchRepairTask(
    event: SystemEvent
  ): Promise<ManusTask | null> {
    if (event.filePath && this.isAlreadyDispatched(event.filePath)) {
      return null;
    }

    const prompt = this.buildRepairPrompt(event);
    const title = `[AUTO-REPAIR] ${event.message.slice(0, 80)}`;

    const task: ManusTask = {
      id: this.generateTaskId(),
      title,
      prompt,
      status: 'dispatched',
      dispatchedAt: new Date().toISOString()
    };

    if (!this.config.dryRun) {
      await this.callManusApi(task);
    }

    this.dispatchHistory.push(task);
    return task;
  }

  private buildRepairPrompt(event: SystemEvent): string {
    const context = this.meta.buildRepairContext([event]);
    return [
      `# Autonomous Repair Task`,
      ``,
      `## Issue`,
      event.message,
      event.filePath ? `**File:** \`${event.filePath}\`` : '',
      `**Severity:** ${event.severity}`,
      ``,
      `## Instructions`,
      context,
      ``,
      `## Requirements`,
      `1. Clone the repository`,
      `2. Reproduce the failure with a failing test (RED phase)`,
      `3. Write the minimum fix to make the test pass (GREEN phase)`,
      `4. Verify all existing tests still pass`,
      `5. Open a pull request with the fix`,
      ``,
      `Follow the DEVELOPMENT_DISCIPLINE.md in the repository root.`
    ]
      .filter(Boolean)
      .join('\n');
  }

  private async callManusApi(task: ManusTask): Promise<void> {
    const url = `${this.config.baseUrl}/tasks`;
    const body: Record<string, unknown> = {
      title: task.title,
      prompt: task.prompt
    };
    if (this.config.projectId) {
      body['projectId'] = this.config.projectId;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.apiKey}`
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      task.status = 'failed';
      throw new Error(
        `Manus API error: ${response.status} ${response.statusText}`
      );
    }

    task.status = 'acknowledged';
  }

  private generateTaskId(): string {
    return `manus-repair-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }
}
