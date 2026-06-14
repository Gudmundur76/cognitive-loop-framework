/**
 * L2 Self-Prompt Reasoning Engine
 *
 * This is the TypeScript interface to the Ollama-deployed SLM.
 * It sits at Layer 2 of the five-layer cognitive loop and is
 * responsible for interpreting events from L1 (Truth Layer) and
 * generating repair candidates, refactoring hypotheses, and
 * diagnostic reports.
 *
 * In production: calls the local Ollama instance running the
 * fine-tuned codebase-slm model.
 *
 * In development/test: falls back to the OpenAI-compatible API
 * (OPENAI_API_BASE + OPENAI_API_KEY from environment).
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 */

export type ReasoningMode =
  | 'diagnose'
  | 'repair'
  | 'explain'
  | 'dream'
  | 'relate';

export interface ReasoningRequest {
  mode: ReasoningMode;
  context: string;
  nodeId?: string;
  failureDescription?: string;
}

export interface ReasoningResponse {
  mode: ReasoningMode;
  output: string;
  model: string;
  durationMs: number;
  tokenCount?: number;
}

export interface SelfPromptEngineConfig {
  ollamaUrl?: string;
  ollamaModel?: string;
  fallbackToOpenAI?: boolean;
  timeoutMs?: number;
}

const DEFAULT_CONFIG: Required<SelfPromptEngineConfig> = {
  ollamaUrl: process.env['OLLAMA_URL'] ?? 'http://localhost:11434',
  ollamaModel: process.env['OLLAMA_MODEL'] ?? 'codebase-slm',
  fallbackToOpenAI: true,
  timeoutMs: 30_000
};

export class SelfPromptEngine {
  private readonly config: Required<SelfPromptEngineConfig>;

  constructor(config: SelfPromptEngineConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Send a reasoning request to the SLM.
   * Automatically falls back to OpenAI if Ollama is unavailable.
   */
  public async reason(req: ReasoningRequest): Promise<ReasoningResponse> {
    const prompt = this.buildPrompt(req);
    const start = Date.now();

    const ollamaAvailable = await this.checkOllama();
    if (ollamaAvailable) {
      return this.callOllama(prompt, req.mode, start);
    }
    if (this.config.fallbackToOpenAI) {
      return this.callOpenAI(prompt, req.mode, start);
    }
    throw new Error('Ollama unavailable and OpenAI fallback is disabled.');
  }

  /**
   * Build the structured prompt for a given reasoning mode.
   */
  private buildPrompt(req: ReasoningRequest): string {
    const modeInstructions: Record<ReasoningMode, string> = {
      diagnose: `Diagnose the following failure. Output DIAGNOSIS, LOCATION, FIX, TEST, RISK.`,
      repair: `Generate a TypeScript fix for the following issue. Follow all project conventions.`,
      explain: `Explain what the following code does in plain language.`,
      dream: `Generate a hypothesis about what could be improved in the following code. Be creative but grounded.`,
      relate: `Identify all dependencies and relationships in the following code.`
    };
    const instruction = modeInstructions[req.mode];
    const context = req.failureDescription
      ? `${req.failureDescription}\n\n${req.context}`
      : req.context;
    return `### Instruction:\n${instruction}\n\n### Input:\n${context}\n\n### Response:\n`;
  }

  /**
   * Check if the Ollama instance is running and the model is loaded.
   */
  private async checkOllama(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${this.config.ollamaUrl}/api/tags`, {
        signal: controller.signal
      });
      clearTimeout(timeout);
      if (!res.ok) return false;
      const data = await res.json() as { models?: Array<{ name: string }> };
      return data.models?.some(m => m.name.includes(this.config.ollamaModel)) ?? false;
    } catch {
      return false;
    }
  }

  /**
   * Call the local Ollama instance with the given prompt.
   */
  private async callOllama(
    prompt: string,
    mode: ReasoningMode,
    start: number
  ): Promise<ReasoningResponse> {
    const res = await fetch(`${this.config.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.ollamaModel,
        prompt,
        stream: false,
        options: { temperature: 0.2, num_predict: 1024 }
      })
    });
    const data = await res.json() as {
      response: string;
      eval_count?: number;
    };
    return {
      mode,
      output: data.response,
      model: this.config.ollamaModel,
      durationMs: Date.now() - start,
      tokenCount: data.eval_count
    };
  }

  /**
   * Fall back to the OpenAI-compatible API when Ollama is unavailable.
   * Uses the OPENAI_API_BASE and OPENAI_API_KEY environment variables.
   */
  private async callOpenAI(
    prompt: string,
    mode: ReasoningMode,
    start: number
  ): Promise<ReasoningResponse> {
    const baseUrl = process.env['OPENAI_API_BASE'] ?? 'https://api.openai.com/v1';
    const apiKey = process.env['OPENAI_API_KEY'] ?? '';
    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.2,
        max_tokens: 1024
      })
    });
    const data = await res.json() as {
      choices: Array<{ message: { content: string } }>;
      usage?: { total_tokens: number };
    };
    return {
      mode,
      output: data.choices[0]?.message.content ?? '',
      model: 'openai-fallback',
      durationMs: Date.now() - start,
      tokenCount: data.usage?.total_tokens
    };
  }
}
