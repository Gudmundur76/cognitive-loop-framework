/**
 * Training Corpus Generator
 *
 * Generates structured Q&A training pairs from CodeNode objects
 * extracted by the ASTExtractor. These pairs are used to fine-tune
 * a small language model (SLM) on the codebase's own patterns.
 *
 * Training pair types generated:
 *   1. EXPLAIN   — "What does [function] do?" → code + docstring
 *   2. LOCATE    — "Where is [function] defined?" → file + line
 *   3. DIAGNOSE  — "What could go wrong in [function]?" → risk analysis
 *   4. RELATE    — "What does [function] call?" → dependency list
 *   5. REPAIR    — "Fix this failing test for [function]" → implementation
 *
 * Design constraints:
 * - Max 200 lines per file
 * - Max 20 lines per function
 */

import * as fs from 'fs';
import * as path from 'path';
import { CodeNode, CodeEdge } from '../indexer/extractor.js';

export interface TrainingPair {
  instruction: string;
  input: string;
  output: string;
  type: 'explain' | 'locate' | 'diagnose' | 'relate' | 'repair';
  sourceNodeId: string;
}

export interface CorpusStats {
  totalPairs: number;
  byType: Record<TrainingPair['type'], number>;
  sourceFiles: number;
}

export class CorpusGenerator {
  /**
   * Generate all training pairs for a set of nodes and edges.
   * Returns pairs sorted by type for balanced fine-tuning batches.
   */
  public generate(
    nodes: CodeNode[],
    edges: CodeEdge[]
  ): TrainingPair[] {
    const pairs: TrainingPair[] = [];
    const edgeMap = this.buildEdgeMap(edges);

    for (const node of nodes) {
      pairs.push(...this.generateExplainPair(node));
      pairs.push(...this.generateLocatePair(node));
      pairs.push(...this.generateDiagnosePair(node));
      pairs.push(...this.generateRelatePair(node, edgeMap));
    }

    return pairs.sort((a, b) => a.type.localeCompare(b.type));
  }

  /**
   * Write the corpus to a JSONL file for use with TRL/Unsloth.
   * Each line is a JSON object with instruction/input/output fields.
   */
  public writeJsonl(pairs: TrainingPair[], outputPath: string): CorpusStats {
    const lines = pairs.map(p => JSON.stringify({
      instruction: p.instruction,
      input: p.input,
      output: p.output
    }));
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, lines.join('\n') + '\n', 'utf8');
    return this.computeStats(pairs);
  }

  // ── Pair generators ──────────────────────────────────────────────

  private generateExplainPair(node: CodeNode): TrainingPair[] {
    if (node.code.length < 20) return [];
    return [{
      instruction: `Explain what the ${node.type} \`${node.name}\` does.`,
      input: node.code.slice(0, 800),
      output: [
        `\`${node.name}\` is a ${node.type} defined in \`${node.filePath}\``,
        `at lines ${node.startLine}–${node.endLine}.`,
        `It ${this.inferPurpose(node)}.`
      ].join(' '),
      type: 'explain',
      sourceNodeId: node.id
    }];
  }

  private generateLocatePair(node: CodeNode): TrainingPair[] {
    return [{
      instruction: `Where is \`${node.name}\` defined in the codebase?`,
      input: '',
      output: `\`${node.name}\` is defined in \`${node.filePath}\`, ` +
        `lines ${node.startLine}–${node.endLine}.`,
      type: 'locate',
      sourceNodeId: node.id
    }];
  }

  private generateDiagnosePair(node: CodeNode): TrainingPair[] {
    const risks = this.inferRisks(node);
    if (risks.length === 0) return [];
    return [{
      instruction: `What could go wrong in \`${node.name}\`?`,
      input: node.code.slice(0, 600),
      output: `Potential issues in \`${node.name}\`:\n` +
        risks.map((r, i) => `${i + 1}. ${r}`).join('\n'),
      type: 'diagnose',
      sourceNodeId: node.id
    }];
  }

  private generateRelatePair(
    node: CodeNode,
    edgeMap: Map<string, CodeEdge[]>
  ): TrainingPair[] {
    const edges = edgeMap.get(node.id) ?? [];
    if (edges.length === 0) return [];
    const callList = edges
      .map(e => `\`${e.targetId.split(':')[1] ?? e.targetId}\``)
      .join(', ');
    return [{
      instruction: `What does \`${node.name}\` depend on or call?`,
      input: '',
      output: `\`${node.name}\` has ${edges.length} outgoing relationship(s): ${callList}.`,
      type: 'relate',
      sourceNodeId: node.id
    }];
  }

  // ── Helpers ──────────────────────────────────────────────────────

  private buildEdgeMap(edges: CodeEdge[]): Map<string, CodeEdge[]> {
    const map = new Map<string, CodeEdge[]>();
    for (const edge of edges) {
      const existing = map.get(edge.sourceId) ?? [];
      existing.push(edge);
      map.set(edge.sourceId, existing);
    }
    return map;
  }

  private inferPurpose(node: CodeNode): string {
    const name = node.name.toLowerCase();
    if (name.startsWith('get') || name.startsWith('fetch')) {
      return 'retrieves or fetches data';
    }
    if (name.startsWith('set') || name.startsWith('update')) {
      return 'updates or modifies state';
    }
    if (name.startsWith('create') || name.startsWith('build')) {
      return 'creates or constructs a new entity';
    }
    if (name.startsWith('delete') || name.startsWith('remove')) {
      return 'removes or deletes an entity';
    }
    if (name.startsWith('validate') || name.startsWith('check')) {
      return 'validates or checks a condition';
    }
    if (name.startsWith('handle') || name.startsWith('process')) {
      return 'handles or processes an event or request';
    }
    return 'performs a specific operation within the system';
  }

  private inferRisks(node: CodeNode): string[] {
    const risks: string[] = [];
    const code = node.code;
    if (code.includes('JSON.parse')) {
      risks.push('Unhandled JSON parse errors if input is malformed');
    }
    if (code.includes('as any')) {
      risks.push('Type safety bypassed with `as any` cast');
    }
    if (code.includes('!') && !code.includes('!==') && !code.includes('!=')) {
      risks.push('Non-null assertion operator used — potential runtime null error');
    }
    if (code.includes('console.log') || code.includes('console.error')) {
      risks.push('Debug logging left in production code');
    }
    if (code.includes('setTimeout') || code.includes('setInterval')) {
      risks.push('Timer-based logic may cause memory leaks if not cleared');
    }
    return risks;
  }

  private computeStats(pairs: TrainingPair[]): CorpusStats {
    const byType = { explain: 0, locate: 0, diagnose: 0, relate: 0, repair: 0 };
    const files = new Set<string>();
    for (const p of pairs) {
      byType[p.type]++;
      files.add(p.sourceNodeId.split(':')[0] ?? '');
    }
    return { totalPairs: pairs.length, byType, sourceFiles: files.size };
  }
}
