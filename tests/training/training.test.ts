import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { ClaimsCorpusGenerator } from '../../src/training/claimsCorpusGenerator.js';
import { CorpusWatcher } from '../../src/training/corpusWatcher.js';
import { IncrementalTrainer } from '../../src/training/incrementalTrainer.js';

// ── Fixtures ─────────────────────────────────────────────────────

const mockVerdictEvent = {
  claimId: 'claim-123',
  claimText: 'Protein XYZ binds to receptor ABC.',
  verdict: 'Supported',
  confidence: 0.95,
  contextSentence: 'Our experiments demonstrate that protein XYZ binds to receptor ABC with high affinity.',
  entities: [
    { type: 'protein', name: 'XYZ', canonicalId: 'P12345' },
    { type: 'receptor', name: 'ABC', canonicalId: 'R67890' }
  ],
  provenance: 'Paper 123 -> Figure 4 -> "Our experiments demonstrate..." -> Supported'
};

const mockContradictionEvent = {
  claimId1: 'claim-123',
  claimText1: 'Protein XYZ binds to receptor ABC.',
  claimId2: 'claim-456',
  claimText2: 'Protein XYZ does not bind to receptor ABC.',
  relationship: 'Contradicts'
};

// ── Typed exec mock helper ────────────────────────────────────────

type ExecCallback = (error: Error | null, stdout: string, stderr: string) => void;
type ExecFn = (cmd: string, callback: ExecCallback) => void;

function makeExecMock(result: { error?: Error; stdout?: string; stderr?: string } = {}): ExecFn & { mock: { calls: Array<[string, ExecCallback]> } } {
  const calls: Array<[string, ExecCallback]> = [];
  const fn = vi.fn((cmd: string, callback: ExecCallback) => {
    calls.push([cmd, callback]);
    callback(result.error ?? null, result.stdout ?? '', result.stderr ?? '');
  }) as ExecFn & { mock: { calls: Array<[string, ExecCallback]> } };
  return fn;
}

// ── ClaimsCorpusGenerator Tests ───────────────────────────────────

describe('ClaimsCorpusGenerator', () => {
  let generator: ClaimsCorpusGenerator;
  let tmpDir: string;
  let outPath: string;

  beforeEach(() => {
    tmpDir = os.tmpdir();
    outPath = path.join(tmpDir, `test_claims_corpus_${Date.now()}_${Math.random().toString(36).slice(2)}.jsonl`);
    generator = new ClaimsCorpusGenerator(outPath);
  });
  afterEach(() => {
    if (fs.existsSync(outPath)) fs.unlinkSync(outPath);
  });

  it('generates classify, extract, provenance, and score pairs from a verdict event', () => {
    const pairs = generator.processVerdictEvent(mockVerdictEvent);
    
    expect(pairs).toHaveLength(4);
    
    const types = pairs.map(p => p.type);
    expect(types).toContain('classify');
    expect(types).toContain('extract');
    expect(types).toContain('provenance');
    expect(types).toContain('score');
  });

  it('classify pair formats correctly', () => {
    const pairs = generator.processVerdictEvent(mockVerdictEvent);
    const classify = pairs.find(p => p.type === 'classify');
    
    expect(classify?.instruction).toContain('Classify the scientific claim');
    expect(classify?.input).toBe(mockVerdictEvent.claimText);
    expect(classify?.output).toContain('Supported');
    expect(classify?.output).toContain('0.95');
  });

  it('extract pair formats correctly', () => {
    const pairs = generator.processVerdictEvent(mockVerdictEvent);
    const extract = pairs.find(p => p.type === 'extract');
    
    expect(extract?.input).toBe(mockVerdictEvent.contextSentence);
    expect(extract?.output).toContain('P12345');
    expect(extract?.output).toContain('R67890');
  });

  it('generates contradict pair from a contradiction event', () => {
    const pairs = generator.processContradictionEvent(mockContradictionEvent);
    
    expect(pairs).toHaveLength(1);
    expect(pairs[0]?.type).toBe('contradict');
    expect(pairs[0]?.input).toContain(mockContradictionEvent.claimText1);
    expect(pairs[0]?.input).toContain(mockContradictionEvent.claimText2);
    expect(pairs[0]?.output).toBe('Contradicts');
  });

  it('appends pairs to the JSONL file', () => {
    generator.processVerdictEvent(mockVerdictEvent);
    
    expect(fs.existsSync(outPath)).toBe(true);
    const lines = fs.readFileSync(outPath, 'utf8').trim().split('\n');
    expect(lines).toHaveLength(4);
    
    // Process another event
    generator.processContradictionEvent(mockContradictionEvent);
    const newLines = fs.readFileSync(outPath, 'utf8').trim().split('\n');
    expect(newLines).toHaveLength(5);
  });
});

// ── CorpusWatcher Tests ───────────────────────────────────────────

describe('CorpusWatcher', () => {
  let watcher: CorpusWatcher;
  let tmpDir: string;
  let corpusPath: string;

  beforeEach(() => {
    tmpDir = os.tmpdir();
    corpusPath = path.join(tmpDir, `watch_corpus_${Date.now()}.jsonl`);
    fs.writeFileSync(corpusPath, '', 'utf8');
    // Threshold of 3 new examples triggers training
    watcher = new CorpusWatcher(corpusPath, 3);
  });

  it('does not fire event when below threshold', () => {
    const mockCallback = vi.fn();
    watcher.onReady(mockCallback);
    
    // Add 2 lines
    fs.appendFileSync(corpusPath, '{"test": 1}\n{"test": 2}\n', 'utf8');
    watcher.check();
    
    expect(mockCallback).not.toHaveBeenCalled();
  });

  it('fires event when threshold is crossed', () => {
    const mockCallback = vi.fn();
    watcher.onReady(mockCallback);
    
    // Add 4 lines (threshold is 3)
    fs.appendFileSync(corpusPath, '{"test": 1}\n{"test": 2}\n{"test": 3}\n{"test": 4}\n', 'utf8');
    watcher.check();
    
    expect(mockCallback).toHaveBeenCalledOnce();
    expect(mockCallback).toHaveBeenCalledWith({ newExamplesCount: 4, totalExamples: 4 });
  });

  it('resets baseline after firing', () => {
    const mockCallback = vi.fn();
    watcher.onReady(mockCallback);
    
    // Add 4 lines
    fs.appendFileSync(corpusPath, '{"test": 1}\n{"test": 2}\n{"test": 3}\n{"test": 4}\n', 'utf8');
    watcher.check();
    expect(mockCallback).toHaveBeenCalledTimes(1);
    
    // Add 2 more lines (total 6, but only 2 since last fire)
    fs.appendFileSync(corpusPath, '{"test": 5}\n{"test": 6}\n', 'utf8');
    watcher.check();
    expect(mockCallback).toHaveBeenCalledTimes(1); // Should not have fired again
    
    // Add 1 more line (total 7, 3 since last fire)
    fs.appendFileSync(corpusPath, '{"test": 7}\n', 'utf8');
    watcher.check();
    expect(mockCallback).toHaveBeenCalledTimes(2); // Should fire now
  });
});

// ── IncrementalTrainer Tests ──────────────────────────────────────

describe('IncrementalTrainer', () => {
  let trainer: IncrementalTrainer;

  beforeEach(() => {
    trainer = new IncrementalTrainer({
      scriptPath: '/path/to/finetunePipeline.py',
      corpusPath: '/path/to/corpus.jsonl',
      outputPath: '/path/to/models'
    });
  });

  it('executes python script with --cpu flag when run is called', async () => {
    const execMock = makeExecMock({ stdout: 'Training complete' });
    trainer.setExecFunction(execMock);
    
    await trainer.run();
    
    // run() calls exec twice: once for python pipeline, once for ollama create
    expect(execMock).toHaveBeenCalledTimes(2);
    const cmd = execMock.mock.calls[0]?.[0] ?? '';
    expect(cmd).toContain('python');
    expect(cmd).toContain('finetunePipeline.py');
    expect(cmd).toContain('--cpu');
    expect(cmd).toContain('--corpus /path/to/corpus.jsonl');
  });

  it('updates ollama model after training completes', async () => {
    const execMock = makeExecMock({ stdout: 'Success' });
    trainer.setExecFunction(execMock);
    
    await trainer.run();
    
    // Should have called exec twice: once for python, once for ollama
    expect(execMock).toHaveBeenCalledTimes(2);
    const cmd2 = execMock.mock.calls[1]?.[0] ?? '';
    expect(cmd2).toContain('ollama create');
  });
});
