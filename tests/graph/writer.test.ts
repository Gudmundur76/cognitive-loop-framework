import { describe, it, expect } from 'vitest';
import { GraphWriter } from '../../src/graph/writer';
import { CodeNode, CodeEdge } from '../../src/indexer/extractor';

describe('GraphWriter', () => {
  it('formats nodes and edges correctly', () => {
    const writer = new GraphWriter();
    
    const nodes: CodeNode[] = [{
      id: 'file.ts:MyClass',
      type: 'class',
      name: 'MyClass',
      startLine: 1,
      endLine: 10,
      filePath: 'file.ts',
      code: 'class MyClass {}'
    }];

    const edges: CodeEdge[] = [];

    const result = writer.formatForGraph(nodes, edges);

    expect(result.nodes).toHaveLength(1);
    expect(result.nodes[0].id).toBe('file.ts:MyClass');
    expect(result.nodes[0].properties.nodeType).toBe('class');
    expect(result.nodes[0].properties.embedding_status).toBe('pending');
  });

  it('generates valid Cypher queries', () => {
    const writer = new GraphWriter();
    
    const nodes: CodeNode[] = [{
      id: 'file.ts:MyClass',
      type: 'class',
      name: 'MyClass',
      startLine: 1,
      endLine: 10,
      filePath: 'file.ts',
      code: 'class MyClass {}'
    }];

    const result = writer.formatForGraph(nodes, []);
    const cypher = writer.generateCypherInsert(result);

    expect(cypher).toContain('MERGE (n:CodeNode {id: "file.ts:MyClass"})');
    expect(cypher).toContain('nodeType:"class"');
  });
});
