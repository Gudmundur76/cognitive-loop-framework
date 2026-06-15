/**
 * cognitiveLoopServer.test.ts
 *
 * Tests for the CognitiveLoopServer — the 5-layer HTTP API that exposes
 * the cognitive loop to ttruthdesk and other consumers.
 *
 * Ralph Wiggum loop: RED → GREEN → VALIDATE → COMPLETE
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as os from 'os';
import * as path from 'path';
import { CognitiveLoopServer } from '../../src/loop/cognitiveLoopServer.js';

// ── Test server setup ─────────────────────────────────────────────────────────
const TEST_PORT = 13100;
const BASE = `http://localhost:${TEST_PORT}`;
let server: CognitiveLoopServer;

beforeAll(async () => {
  server = new CognitiveLoopServer({
    port: TEST_PORT,
    corpusPath: path.join(os.tmpdir(), `test_corpus_server_${Date.now()}.jsonl`),
    compoundingLogPath: path.join(os.tmpdir(), `test_log_server_${Date.now()}.jsonl`),
    webhookSecret: '',
    trainingThreshold: 50,
  });
  await server.start();
});

afterAll(async () => {
  await server.stop();
});

// ── Helper ────────────────────────────────────────────────────────────────────
async function post(path: string, body: unknown) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: res.status, body: await res.json() as Record<string, unknown> };
}

async function get(path: string) {
  const res = await fetch(`${BASE}${path}`);
  return { status: res.status, body: await res.json() as Record<string, unknown> };
}

// ── GET /health ───────────────────────────────────────────────────────────────
describe('GET /health', () => {
  it('returns 200 with ok:true', async () => {
    const { status, body } = await get('/health');
    expect(status).toBe(200);
    expect(body['ok']).toBe(true);
    expect(body['status']).toBe('healthy');
    expect(typeof body['uptimeMs']).toBe('number');
  });
});

// ── GET /cognitive/status ─────────────────────────────────────────────────────
describe('GET /cognitive/status', () => {
  it('returns 200 with flywheel:operational', async () => {
    const { status, body } = await get('/cognitive/status');
    expect(status).toBe(200);
    expect(body['ok']).toBe(true);
    expect(body['flywheel']).toBe('operational');
    expect(body['log']).toBeDefined();
  });
});

// ── POST /cognitive/ingest ────────────────────────────────────────────────────
describe('POST /cognitive/ingest', () => {
  it('returns 400 when event is missing', async () => {
    const { status, body } = await post('/cognitive/ingest', {});
    expect(status).toBe(400);
    expect(body['ok']).toBe(false);
  });

  it('returns 400 when event is missing required fields', async () => {
    const { status, body } = await post('/cognitive/ingest', {
      event: { claimText: 'Protein XYZ binds to receptor ABC.' },
    });
    expect(status).toBe(400);
    expect(body['ok']).toBe(false);
  });

  it('accepts a valid verdict event and returns pairsGenerated', async () => {
    const { status, body } = await post('/cognitive/ingest', {
      event: {
        claimId: 'claim-test-001',
        claimText: 'Protein XYZ binds to receptor ABC.',
        verdict: 'Supported',
        confidence: 0.95,
        contextSentence: 'Our experiments demonstrate that protein XYZ binds to receptor ABC.',
        entities: [
          { type: 'protein', name: 'XYZ', canonicalId: 'P12345' },
          { type: 'receptor', name: 'ABC', canonicalId: 'R67890' },
        ],
        provenance: 'Paper 123 → Figure 4 → Supported',
      },
    });
    expect(status).toBe(200);
    expect(body['ok']).toBe(true);
    expect(typeof body['pairsGenerated']).toBe('number');
    expect((body['pairsGenerated'] as number)).toBeGreaterThan(0);
    expect(body['claimId']).toBe('claim-test-001');
  });

  it('returns 401 when webhook secret is set and signature is missing', async () => {
    const securedServer = new CognitiveLoopServer({
      port: TEST_PORT + 1,
      corpusPath: path.join(os.tmpdir(), `test_corpus_sec_${Date.now()}.jsonl`),
      compoundingLogPath: path.join(os.tmpdir(), `test_log_sec_${Date.now()}.jsonl`),
      webhookSecret: 'my-secret',
    });
    await securedServer.start();
    try {
      const res = await fetch(`http://localhost:${TEST_PORT + 1}/cognitive/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event: { claimId: 'x', claimText: 'x', verdict: 'Supported' } }),
      });
      expect(res.status).toBe(401);
    } finally {
      await securedServer.stop();
    }
  });
});

// ── POST /cognitive/verdict ───────────────────────────────────────────────────
describe('POST /cognitive/verdict', () => {
  it('returns 400 when claimText is missing', async () => {
    const { status, body } = await post('/cognitive/verdict', {});
    expect(status).toBe(400);
    expect(body['ok']).toBe(false);
  });

  it('returns 200 with priorSimilarVerdicts array', async () => {
    const { status, body } = await post('/cognitive/verdict', {
      claimText: 'Protein XYZ binds to receptor ABC.',
      domain: 'biotech',
    });
    expect(status).toBe(200);
    expect(body['ok']).toBe(true);
    expect(Array.isArray(body['priorSimilarVerdicts'])).toBe(true);
    expect(body['domain']).toBe('biotech');
  });
});

// ── POST /cognitive/repair ────────────────────────────────────────────────────
describe('POST /cognitive/repair', () => {
  it('returns 400 when testName is missing', async () => {
    const { status, body } = await post('/cognitive/repair', {
      errorOutput: 'TypeError: Cannot read property of undefined',
    });
    expect(status).toBe(400);
    expect(body['ok']).toBe(false);
  });

  it('returns 400 when errorOutput is missing', async () => {
    const { status, body } = await post('/cognitive/repair', {
      testName: 'verifyClaim.should_handle_429',
    });
    expect(status).toBe(400);
    expect(body['ok']).toBe(false);
  });

  it('returns 200 with repair context for a valid request', async () => {
    const { status, body } = await post('/cognitive/repair', {
      testName: 'verifyClaim.should_handle_429',
      errorOutput: 'Error: rate limiter exceeded',
      filePath: 'server/verifyClaim.ts',
    });
    expect(status).toBe(200);
    expect(body['ok']).toBe(true);
    expect(typeof body['context']).toBe('string');
    expect((body['context'] as string)).toContain('verifyClaim.should_handle_429');
    expect(typeof body['entryTimestamp']).toBe('string');
  });
});

// ── 404 ───────────────────────────────────────────────────────────────────────
describe('Unknown routes', () => {
  it('returns 404 for unknown GET routes', async () => {
    const { status } = await get('/unknown/route');
    expect(status).toBe(404);
  });
});
