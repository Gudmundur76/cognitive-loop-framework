/**
 * CognitiveLoopServer
 *
 * HTTP API server that exposes the 5-layer cognitive loop as described
 * in the product spec and self_improvement_mechanics.md.
 *
 * Endpoints:
 *   POST /cognitive/ingest   — L0: receive a verdict event from ttruthdesk
 *   POST /cognitive/verdict  — L1+L2: get a structured verdict for a claim
 *   POST /cognitive/repair   — L3+L4: trigger self-healing on a test failure
 *   GET  /cognitive/graph    — query the RuVector knowledge graph
 *   GET  /health             — health check
 *   GET  /cognitive/status   — loop status and metrics
 *
 * Sprint 2: RuVectorClient injected into constructor and wired into
 * ingest (graph node write), verdict (hybrid search), and graph endpoint.
 *
 * Design constraints: max 300 lines, max 20 lines/function, max 3 params
 */
import * as http from 'http';
import * as crypto from 'crypto';
import { createTrainingPipeline, type TrainingPipeline } from '../training/index.js';
import { CompoundingLog } from '../memory/compoundingLog.js';
import { RuVectorClient, type RuVectorClientConfig } from '../memory/ruVectorClient.js';
import { ASTIndexer } from '../indexer/astIndexer.js';
import type { VerdictEvent } from '../training/claimsCorpusGenerator.js';

// ── Types ─────────────────────────────────────────────────────────────────────
export interface ServerConfig {
  port?: number;
  corpusPath?: string;
  compoundingLogPath?: string;
  webhookSecret?: string;
  trainingThreshold?: number;
  scriptPath?: string;
  outputPath?: string;
  /** RuVector config — omit to disable graph memory. */
  ruVector?: RuVectorClientConfig;
}

interface IngestRequest {
  event: VerdictEvent;
}

interface VerdictRequest {
  claimText: string;
  domain?: string;
}

interface RepairRequest {
  testName: string;
  errorOutput: string;
  filePath?: string;
}

interface GraphQueryRequest {
  cypher?: string;
  nodeId?: string;
  hops?: number;
}

type JsonBody = Record<string, unknown>;

// ── Server ────────────────────────────────────────────────────────────────────
export class CognitiveLoopServer {
  private readonly config: Required<Omit<ServerConfig, 'ruVector'>> & { ruVector?: RuVectorClientConfig };
  private readonly pipeline: TrainingPipeline;
  private readonly log: CompoundingLog;
  private readonly ruVector: RuVectorClient | null;
  private readonly indexer: ASTIndexer | null;
  private server: http.Server | null = null;
  private startTime = Date.now();

  constructor(config: ServerConfig = {}) {
    this.config = {
      port: config.port ?? Number(process.env['PORT'] ?? 3100),
      corpusPath:
        config.corpusPath ??
        process.env['CORPUS_PATH'] ??
        '/data/corpus/corpus.jsonl',
      compoundingLogPath:
        config.compoundingLogPath ??
        process.env['COMPOUNDING_LOG_PATH'] ??
        '/data/corpus/compounding_log.jsonl',
      webhookSecret:
        config.webhookSecret ??
        process.env['WEBHOOK_SECRET'] ??
        '',
      trainingThreshold:
        config.trainingThreshold ??
        Number(process.env['CORPUS_TRAINING_THRESHOLD'] ?? 50),
      scriptPath:
        config.scriptPath ??
        process.env['FINETUNE_SCRIPT_PATH'] ??
        '/opt/cognitive-loop/finetunePipeline.py',
      outputPath:
        config.outputPath ??
        process.env['TRAINING_OUTPUT_PATH'] ??
        '/data/adapter',
      ruVector: config.ruVector,
    };

    this.pipeline = createTrainingPipeline({
      corpusPath: this.config.corpusPath,
      minPairsThreshold: this.config.trainingThreshold,
      scriptPath: this.config.scriptPath,
      outputPath: this.config.outputPath,
    });

    // Wire RuVectorClient into CompoundingLog for hybrid search
    this.ruVector = this.config.ruVector
      ? new RuVectorClient(this.config.ruVector)
      : null;

    this.log = new CompoundingLog(
      this.config.compoundingLogPath,
      this.ruVector ?? undefined
    );

    this.indexer = this.ruVector
      ? new ASTIndexer(this.ruVector, { useMockEmbeddings: true })
      : null;
  }

  /** Start the HTTP server. */
  public start(): Promise<void> {
    return new Promise(resolve => {
      this.server = http.createServer((req, res) => {
        this.handleRequest(req, res).catch(err => {
          this.sendJson(res, 500, { ok: false, error: String(err) });
        });
      });
      this.server.listen(this.config.port, () => {
        console.log(`[cognitive-loop] Server listening on :${this.config.port}`);
        resolve();
      });
    });
  }

  /** Stop the HTTP server. */
  public stop(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.server) { resolve(); return; }
      this.server.close(err => (err ? reject(err) : resolve()));
    });
  }

  // ── Request router ─────────────────────────────────────────────────────────
  private async handleRequest(
    req: http.IncomingMessage,
    res: http.ServerResponse
  ): Promise<void> {
    const url = req.url ?? '/';
    const method = req.method ?? 'GET';

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('X-Powered-By', 'cognitive-loop-framework');

    if (method === 'GET' && url === '/health') {
      return this.handleHealth(res);
    }
    if (method === 'GET' && url === '/cognitive/status') {
      return this.handleStatus(res);
    }
    if (method === 'POST' && url === '/cognitive/ingest') {
      const body = await this.readBody(req);
      return this.handleIngest(req, res, body);
    }
    if (method === 'POST' && url === '/cognitive/verdict') {
      const body = await this.readBody(req);
      return this.handleVerdict(res, body);
    }
    if (method === 'POST' && url === '/cognitive/repair') {
      const body = await this.readBody(req);
      return this.handleRepair(res, body);
    }
    if (method === 'POST' && url === '/cognitive/graph') {
      const body = await this.readBody(req);
      return this.handleGraph(res, body);
    }

    this.sendJson(res, 404, { ok: false, error: 'Not found' });
  }

  // ── L0: Ingest — receive verdict event from ttruthdesk ────────────────────
  private handleIngest(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    body: JsonBody
  ): void {
    if (this.config.webhookSecret) {
      const sig = req.headers['x-webhook-signature'] as string | undefined;
      if (!sig || !this.verifySignature(JSON.stringify(body), sig)) {
        this.sendJson(res, 401, { ok: false, error: 'Invalid webhook signature' });
        return;
      }
    }

    const event = body['event'] as VerdictEvent | undefined;
    if (!event?.claimId || !event.claimText || !event.verdict) {
      this.sendJson(res, 400, { ok: false, error: 'Missing required event fields' });
      return;
    }

    const pairs = this.pipeline.generator.processVerdictEvent(event);
    this.pipeline.watcher.check();

    this.log.append({
      trigger: 'verdict_event',
      test: event.claimId,
      diagnosis: `Verdict: ${event.verdict} (confidence: ${event.confidence})`,
      patch_hash: '',
      test_result: 'PASS',
      graph_delta: { added: [event.claimId], removed: [], modified: [] },
      slm_confidence: event.confidence,
      frontier_explored: 0,
      meta_review: 'APPROVED',
      metadata: { verdict: event.verdict, pairsGenerated: pairs.length },
    });

    this.sendJson(res, 200, {
      ok: true,
      pairsGenerated: pairs.length,
      types: pairs.map(p => p.type),
      claimId: event.claimId,
      graphEnabled: this.ruVector !== null,
    });
  }

  // ── L1+L2: Verdict — classify a claim using the local SLM ─────────────────
  private handleVerdict(res: http.ServerResponse, body: JsonBody): void {
    const req = body as VerdictRequest;
    if (!req.claimText) {
      this.sendJson(res, 400, { ok: false, error: 'claimText is required' });
      return;
    }

    // Hybrid search: RuVector semantic (if available) + TF-IDF fallback
    const similarPromise = this.log.querySimilar(req.claimText, 3);

    void similarPromise.then(similar => {
      this.sendJson(res, 200, {
        ok: true,
        claimText: req.claimText,
        domain: req.domain ?? 'general',
        priorSimilarVerdicts: similar.map(r => ({
          test: r.entry.test,
          diagnosis: r.entry.diagnosis,
          similarity: r.similarity,
        })),
        graphEnabled: this.ruVector !== null,
        note: 'Full L1+L2 inference requires Ollama. Query the SLM at POST /api/generate on :11434.',
      });
    });
  }

  // ── L3+L4: Repair — self-healing on test failure ──────────────────────────
  private handleRepair(res: http.ServerResponse, body: JsonBody): void {
    const req = body as RepairRequest;
    if (!req.testName || !req.errorOutput) {
      this.sendJson(res, 400, { ok: false, error: 'testName and errorOutput are required' });
      return;
    }

    void this.log.querySimilar(req.errorOutput, 6).then(similar => {
      const priorSuccess = similar.filter(r => r.entry.test_result === 'PASS');
      const priorFail = similar.filter(r => r.entry.test_result === 'FAIL');

      const entry = this.log.append({
        trigger: 'test_failure',
        test: req.testName,
        diagnosis: `Repair triggered for: ${req.errorOutput.slice(0, 200)}`,
        patch_hash: '',
        test_result: 'FAIL',
        graph_delta: {
          added: [],
          removed: [],
          modified: req.filePath ? [req.filePath] : [],
        },
        slm_confidence: 0,
        frontier_explored: 0,
        meta_review: 'PENDING',
      });

      this.sendJson(res, 200, {
        ok: true,
        entryTimestamp: entry.timestamp,
        priorSuccessfulRepairs: priorSuccess.length,
        priorFailedRepairs: priorFail.length,
        context: this.buildRepairContext(req, priorSuccess.map(r => r.entry)),
      });
    });
  }

  // ── Graph — query the RuVector knowledge graph ────────────────────────────
  private handleGraph(res: http.ServerResponse, body: JsonBody): void {
    if (!this.ruVector) {
      this.sendJson(res, 503, {
        ok: false,
        error: 'RuVector not configured. Pass ruVector config to ServerConfig.',
      });
      return;
    }

    const req = body as GraphQueryRequest;

    if (req.cypher) {
      void this.ruVector.query(req.cypher).then(result => {
        this.sendJson(res, 200, { ok: true, ...result });
      }).catch(err => {
        this.sendJson(res, 500, { ok: false, error: String(err) });
      });
      return;
    }

    if (req.nodeId) {
      const hops = req.hops ?? 2;
      void this.ruVector.kHopNeighbors(req.nodeId, hops).then(neighbors => {
        this.sendJson(res, 200, { ok: true, nodeId: req.nodeId, hops, neighbors });
      }).catch(err => {
        this.sendJson(res, 500, { ok: false, error: String(err) });
      });
      return;
    }

    void this.ruVector.stats().then(stats => {
      this.sendJson(res, 200, { ok: true, stats });
    }).catch(err => {
      this.sendJson(res, 500, { ok: false, error: String(err) });
    });
  }

  // ── Health check ──────────────────────────────────────────────────────────
  private handleHealth(res: http.ServerResponse): void {
    this.sendJson(res, 200, {
      ok: true,
      status: 'healthy',
      uptimeMs: Date.now() - this.startTime,
      version: '0.2.0',
      graphEnabled: this.ruVector !== null,
    });
  }

  // ── Status ────────────────────────────────────────────────────────────────
  private handleStatus(res: http.ServerResponse): void {
    const stats = this.log.getStats();
    const contradictions = this.log.scanContradictions();
    this.sendJson(res, 200, {
      ok: true,
      uptimeMs: Date.now() - this.startTime,
      corpus: { path: this.config.corpusPath, threshold: this.config.trainingThreshold },
      log: stats,
      contradictions: contradictions.length,
      flywheel: 'operational',
      graphEnabled: this.ruVector !== null,
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  private buildRepairContext(
    req: RepairRequest,
    priorRepairs: Array<{ test: string; diagnosis: string }>
  ): string {
    const lines = [
      `Test failure: ${req.testName}`,
      `Error: ${req.errorOutput.slice(0, 500)}`,
    ];
    if (req.filePath) lines.push(`File: ${req.filePath}`);
    if (priorRepairs.length > 0) {
      lines.push('', 'Prior successful repairs for similar failures:');
      for (const r of priorRepairs) {
        lines.push(`  - ${r.test}: ${r.diagnosis}`);
      }
    }
    return lines.join('\n');
  }

  private verifySignature(payload: string, signature: string): boolean {
    const expected = crypto
      .createHmac('sha256', this.config.webhookSecret)
      .update(payload)
      .digest('hex');
    try {
      return crypto.timingSafeEqual(
        Buffer.from(signature),
        Buffer.from(expected)
      );
    } catch {
      return false;
    }
  }

  private readBody(req: http.IncomingMessage): Promise<JsonBody> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on('data', chunk => chunks.push(chunk as Buffer));
      req.on('end', () => {
        try {
          const text = Buffer.concat(chunks).toString('utf8');
          resolve(text ? (JSON.parse(text) as JsonBody) : {});
        } catch (err) {
          reject(err);
        }
      });
      req.on('error', reject);
    });
  }

  private sendJson(
    res: http.ServerResponse,
    status: number,
    body: unknown
  ): void {
    res.writeHead(status);
    res.end(JSON.stringify(body));
  }
}
