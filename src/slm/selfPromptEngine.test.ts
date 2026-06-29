/**
 * Tests for SelfPromptEngine — Ornith routing, reasoning extraction, fallback chain.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SelfPromptEngine } from './selfPromptEngine.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeFetchMock(response: object, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    json: () => Promise.resolve(response),
  });
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('SelfPromptEngine — Ornith routing', () => {
  let originalFetch: typeof global.fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('calls Ornith /v1/chat/completions when ornithBaseUrl is set', async () => {
    const mockFetch = makeFetchMock({
      choices: [{ message: { content: 'The answer is 42.' } }],
      usage: { total_tokens: 100 },
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: 'http://ornith:8000' });
    const result = await engine.reason({ mode: 'diagnose', context: 'some code' });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url] = mockFetch.mock.calls[0] as [string, ...unknown[]];
    expect(url).toBe('http://ornith:8000/v1/chat/completions');
    expect(result.model).toBe('ornith-1.0-9b');
    expect(result.output).toBe('The answer is 42.');
  });

  it('uses custom ornithModel when configured', async () => {
    const mockFetch = makeFetchMock({
      choices: [{ message: { content: 'ok' } }],
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({
      ornithBaseUrl: 'http://ornith:8000',
      ornithModel: 'ornith-custom-7b',
    });
    await engine.reason({ mode: 'explain', context: 'x' });

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe('ornith-custom-7b');
  });

  it('extracts server-side reasoning_content when present', async () => {
    const mockFetch = makeFetchMock({
      choices: [{
        message: {
          content: 'Final answer here.',
          reasoning_content: 'I thought about it carefully.',
        },
      }],
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: 'http://ornith:8000' });
    const result = await engine.reason({ mode: 'dream', context: 'ctx' });

    expect(result.output).toBe('Final answer here.');
    expect(result.reasoningTrace).toBe('I thought about it carefully.');
  });

  it('extracts <think> block from content when reasoning_content is absent', async () => {
    const mockFetch = makeFetchMock({
      choices: [{
        message: {
          content: '<think>Step 1: analyse.\nStep 2: conclude.</think>\n\nThe conclusion is X.',
        },
      }],
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: 'http://ornith:8000' });
    const result = await engine.reason({ mode: 'repair', context: 'ctx' });

    expect(result.output).toBe('The conclusion is X.');
    expect(result.reasoningTrace).toContain('Step 1: analyse.');
    expect(result.reasoningTrace).toContain('Step 2: conclude.');
  });

  it('returns no reasoningTrace when content has no <think> block', async () => {
    const mockFetch = makeFetchMock({
      choices: [{ message: { content: 'Plain answer.' } }],
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: 'http://ornith:8000' });
    const result = await engine.reason({ mode: 'relate', context: 'ctx' });

    expect(result.output).toBe('Plain answer.');
    expect(result.reasoningTrace).toBeUndefined();
  });

  it('includes tokenCount from usage when present', async () => {
    const mockFetch = makeFetchMock({
      choices: [{ message: { content: 'ans' } }],
      usage: { total_tokens: 512 },
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: 'http://ornith:8000' });
    const result = await engine.reason({ mode: 'diagnose', context: 'ctx' });

    expect(result.tokenCount).toBe(512);
  });
});

describe('SelfPromptEngine — fallback chain (no Ornith)', () => {
  let originalFetch: typeof global.fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('skips Ornith and checks Ollama when ornithBaseUrl is empty', async () => {
    // Ollama tags endpoint returns model available
    const mockFetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ models: [{ name: 'codebase-slm' }] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ response: 'ollama answer', eval_count: 50 }),
      });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: '' });
    const result = await engine.reason({ mode: 'diagnose', context: 'ctx' });

    expect(result.model).toBe('codebase-slm');
    expect(result.output).toBe('ollama answer');
    expect(result.reasoningTrace).toBeUndefined();
  });

  it('falls back to OpenAI when Ollama is unavailable and ornithBaseUrl is empty', async () => {
    // Ollama check fails, OpenAI succeeds
    const mockFetch = vi.fn()
      .mockRejectedValueOnce(new Error('ECONNREFUSED'))
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'openai answer' } }],
          usage: { total_tokens: 200 },
        }),
      });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({
      ornithBaseUrl: '',
      fallbackToOpenAI: true,
    });
    const result = await engine.reason({ mode: 'explain', context: 'ctx' });

    expect(result.model).toBe('openai-fallback');
    expect(result.output).toBe('openai answer');
  });

  it('throws when Ollama unavailable and fallback disabled and no Ornith', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED'));

    const engine = new SelfPromptEngine({
      ornithBaseUrl: '',
      fallbackToOpenAI: false,
    });

    await expect(engine.reason({ mode: 'diagnose', context: 'ctx' }))
      .rejects.toThrow('Ollama unavailable');
  });
});

describe('SelfPromptEngine — mode routing', () => {
  afterEach(() => vi.restoreAllMocks());

  it('includes mode-specific instruction in the prompt body', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        choices: [{ message: { content: 'ok' } }],
      }),
    });
    global.fetch = mockFetch;

    const engine = new SelfPromptEngine({ ornithBaseUrl: 'http://ornith:8000' });
    await engine.reason({ mode: 'dream', context: 'some context' });

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    const prompt: string = body.messages[0].content;
    expect(prompt).toContain('hypothesis');
    expect(prompt).toContain('some context');
  });
});
