/**
 * ClaimsCorpusGenerator
 *
 * Converts verified claim events from ttruthdesk.claims into labelled
 * JSONL training pairs for narrow domain SLM fine-tuning.
 *
 * This is the data flywheel entry point: every claim verified by the
 * ttruthdesk pipeline becomes a training example that improves the model.
 *
 * Training pair types generated:
 *   classify  — claim text → verdict + confidence
 *   extract   — sentence → structured entity JSON
 *   contradict — two claims → relationship label
 *   provenance — claim + evidence → chain explanation
 *   score     — claim + context → confidence 0.0-1.0
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import * as fs from 'fs';
import * as path from 'path';

export interface VerdictEvent {
  claimId: string;
  claimText: string;
  verdict: string;
  confidence: number;
  contextSentence: string;
  entities: EntityRecord[];
  provenance: string;
}

export interface ContradictionEvent {
  claimId1: string;
  claimText1: string;
  claimId2: string;
  claimText2: string;
  relationship: string;
}

export interface EntityRecord {
  type: string;
  name: string;
  canonicalId: string;
}

export interface ClaimsTrainingPair {
  instruction: string;
  input: string;
  output: string;
  type: 'classify' | 'extract' | 'contradict' | 'provenance' | 'score';
  claimId: string;
}

export class ClaimsCorpusGenerator {
  private readonly outPath: string;

  constructor(outPath: string) {
    this.outPath = outPath;
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
  }

  /**
   * Process a verdict_complete event.
   * Generates 4 training pairs and appends them to the JSONL corpus.
   */
  public processVerdictEvent(event: VerdictEvent): ClaimsTrainingPair[] {
    const pairs: ClaimsTrainingPair[] = [
      this.buildClassifyPair(event),
      this.buildExtractPair(event),
      this.buildProvenancePair(event),
      this.buildScorePair(event)
    ];
    this.appendToCorpus(pairs);
    return pairs;
  }

  /**
   * Process a contradiction event.
   * Generates 1 training pair and appends it to the JSONL corpus.
   */
  public processContradictionEvent(event: ContradictionEvent): ClaimsTrainingPair[] {
    const pairs: ClaimsTrainingPair[] = [this.buildContradictPair(event)];
    this.appendToCorpus(pairs);
    return pairs;
  }

  // ── Pair builders ────────────────────────────────────────────────

  private buildClassifyPair(event: VerdictEvent): ClaimsTrainingPair {
    return {
      instruction: 'Classify the scientific claim as Supported, Refuted, Inconclusive, or Needs Context. Return the verdict and confidence score.',
      input: event.claimText,
      output: `Verdict: ${event.verdict}\nConfidence: ${event.confidence.toFixed(2)}`,
      type: 'classify',
      claimId: event.claimId
    };
  }

  private buildExtractPair(event: VerdictEvent): ClaimsTrainingPair {
    const entityJson = JSON.stringify(
      event.entities.map(e => ({
        type: e.type,
        name: e.name,
        canonical_id: e.canonicalId
      })),
      null,
      2
    );
    return {
      instruction: 'Extract all scientific entities from the sentence. Return a JSON array with type, name, and canonical_id fields.',
      input: event.contextSentence,
      output: entityJson,
      type: 'extract',
      claimId: event.claimId
    };
  }

  private buildContradictPair(event: ContradictionEvent): ClaimsTrainingPair {
    return {
      instruction: 'Determine the relationship between the two scientific claims. Answer with exactly one of: Contradicts, Supports, Unrelated.',
      input: `Claim 1: ${event.claimText1}\nClaim 2: ${event.claimText2}`,
      output: event.relationship,
      type: 'contradict',
      claimId: event.claimId1
    };
  }

  private buildProvenancePair(event: VerdictEvent): ClaimsTrainingPair {
    return {
      instruction: 'Explain the provenance chain that supports or refutes this scientific claim.',
      input: event.claimText,
      output: event.provenance,
      type: 'provenance',
      claimId: event.claimId
    };
  }

  private buildScorePair(event: VerdictEvent): ClaimsTrainingPair {
    return {
      instruction: 'Given the scientific claim and its context sentence, assign a confidence score from 0.0 to 1.0 for how well the evidence supports the claim.',
      input: `Claim: ${event.claimText}\nContext: ${event.contextSentence}`,
      output: event.confidence.toFixed(2),
      type: 'score',
      claimId: event.claimId
    };
  }

  // ── File I/O ─────────────────────────────────────────────────────

  private appendToCorpus(pairs: ClaimsTrainingPair[]): void {
    const lines = pairs.map(p => JSON.stringify({
      instruction: p.instruction,
      input: p.input,
      output: p.output
    }));
    fs.appendFileSync(this.outPath, lines.join('\n') + '\n', 'utf8');
  }
}
