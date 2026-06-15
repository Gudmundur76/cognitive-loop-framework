/**
 * SIADatasetGenerator
 *
 * Generates synthetic SIA task datasets for the citation-integrity task
 * when a live database connection is not available (CI, development, testing).
 *
 * Produces the same schema as generate_dataset.mjs:
 *   - claims.jsonl  (200 public claim-source pairs)
 *   - ground_truth.jsonl (100 private held-out records)
 *
 * Synthetic data is deterministic (seeded by run ID) and covers all four
 * citation states in a realistic distribution:
 *   verified 40% / contested 25% / implied 25% / beyond_evidence 10%
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */
import * as fs from 'fs';
import * as path from 'path';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SIAPublicRecord {
  claim_id: string;
  claim_text: string;
  source_title: string;
  source_abstract: string;
  source_full_text: string | null;
  domain: string;
}

export interface SIAGroundTruthRecord {
  claim_id: string;
  citation_state: 'verified' | 'contested' | 'implied' | 'beyond_evidence';
  confidence: number;
  source_passage: string | null;
  misrepresentation_pattern: string | null;
}

export interface SIADatasetConfig {
  publicCount: number;
  privateCount: number;
  seed: number;
  domain: string;
}

const DEFAULT_CONFIG: SIADatasetConfig = {
  publicCount: 200,
  privateCount: 100,
  seed: 42,
  domain: 'protein_biology',
};

// ─── Synthetic data templates ─────────────────────────────────────────────────

const CLAIM_TEMPLATES = [
  'Protein {A} binds to receptor {B} with high affinity in {C} cells.',
  '{A} inhibits {B} activity by blocking the {C} domain.',
  'Overexpression of {A} leads to increased {B} levels in {C} tissue.',
  '{A} knockout mice show reduced {B} expression in {C}.',
  'Treatment with {A} significantly reduces {B} in {C} patients.',
  '{A} and {B} interact via the {C} interface.',
  'The {A} pathway regulates {B} through {C} signalling.',
  '{A} is upregulated in {C} under {B} conditions.',
];

const PROTEINS = ['lysozyme', 'p53', 'BRCA1', 'mTOR', 'AKT1', 'VEGF', 'TNF-α', 'IL-6'];
const RECEPTORS = ['EGFR', 'HER2', 'ACE2', 'TLR4', 'PPAR-γ', 'AR', 'ER-α', 'GLP-1R'];
const CONTEXTS = ['HeLa', 'MCF-7', 'HEK293', 'murine', 'human hepatic', 'pancreatic', 'neuronal'];
const MISREP_PATTERNS = [
  'strength_overclaim', 'scope_overclaim', 'causation_from_correlation',
  'omitted_contradicting_evidence', null, null, null,
];
const CITATION_STATES: SIAGroundTruthRecord['citation_state'][] = [
  'verified', 'verified', 'verified', 'verified',
  'contested', 'contested', 'contested',
  'implied', 'implied', 'implied',
  'beyond_evidence',
];

// ─── Seeded PRNG (mulberry32) ─────────────────────────────────────────────────

function makePrng(seed: number) {
  let s = seed;
  return () => {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function pick<T>(arr: T[], rand: () => number): T {
  return arr[Math.floor(rand() * arr.length)];
}

// ─── Record builders ──────────────────────────────────────────────────────────

function buildPublicRecord(id: number, rand: () => number, domain: string): SIAPublicRecord {
  const template = pick(CLAIM_TEMPLATES, rand);
  const a = pick(PROTEINS, rand);
  const b = pick(RECEPTORS, rand);
  const c = pick(CONTEXTS, rand);
  const claimText = template
    .replace('{A}', a).replace('{B}', b).replace('{C}', c);
  const sourceAbstract =
    `We investigated the role of ${a} in ${c} cells. Our results demonstrate ` +
    `that ${a} modulates ${b} activity. These findings suggest a potential ` +
    `therapeutic target in ${domain.replace('_', ' ')} research.`;
  return {
    claim_id: `syn-${String(id).padStart(5, '0')}`,
    claim_text: claimText,
    source_title: `${a} and ${b}: a mechanistic study in ${c} models`,
    source_abstract: sourceAbstract,
    source_full_text: null,
    domain,
  };
}

function buildGroundTruthRecord(
  publicRecord: SIAPublicRecord,
  rand: () => number,
): SIAGroundTruthRecord {
  const state = pick(CITATION_STATES, rand);
  const confidence =
    state === 'verified' ? 0.75 + rand() * 0.24 :
    state === 'contested' ? 0.45 + rand() * 0.25 :
    state === 'implied' ? 0.35 + rand() * 0.30 :
    0.10 + rand() * 0.25;
  const misrep = state === 'verified' ? null : pick(MISREP_PATTERNS, rand);
  const passage =
    state === 'beyond_evidence' ? null :
    `${publicRecord.source_abstract.split('.')[0]}.`;
  return {
    claim_id: publicRecord.claim_id,
    citation_state: state,
    confidence: Math.round(confidence * 100) / 100,
    source_passage: passage,
    misrepresentation_pattern: misrep,
  };
}

// ─── Main class ───────────────────────────────────────────────────────────────

export class SIADatasetGenerator {
  private readonly config: SIADatasetConfig;

  constructor(config: Partial<SIADatasetConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  generate(outputDir: string): { publicPath: string; privatePath: string } {
    const rand = makePrng(this.config.seed);
    const publicDir = path.join(outputDir, 'data', 'public');
    const privateDir = path.join(outputDir, 'data', 'private');
    fs.mkdirSync(publicDir, { recursive: true });
    fs.mkdirSync(privateDir, { recursive: true });

    const total = this.config.publicCount + this.config.privateCount;
    const publicRecords: SIAPublicRecord[] = [];
    for (let i = 0; i < total; i++) {
      publicRecords.push(buildPublicRecord(i + 1, rand, this.config.domain));
    }

    const publicSlice = publicRecords.slice(0, this.config.publicCount);
    const privateSlice = publicRecords.slice(this.config.publicCount);

    const publicPath = path.join(publicDir, 'claims.jsonl');
    fs.writeFileSync(publicPath, publicSlice.map(r => JSON.stringify(r)).join('\n') + '\n');

    const groundTruth = privateSlice.map(r => buildGroundTruthRecord(r, rand));
    const privatePath = path.join(privateDir, 'ground_truth.jsonl');
    fs.writeFileSync(privatePath, groundTruth.map(r => JSON.stringify(r)).join('\n') + '\n');

    return { publicPath, privatePath };
  }

  /** Returns the expected citation state distribution for a given seed */
  distribution(): Record<string, number> {
    const rand = makePrng(this.config.seed);
    const counts: Record<string, number> = {};
    for (let i = 0; i < this.config.privateCount; i++) {
      const state = pick(CITATION_STATES, rand);
      counts[state] = (counts[state] ?? 0) + 1;
    }
    return counts;
  }
}
