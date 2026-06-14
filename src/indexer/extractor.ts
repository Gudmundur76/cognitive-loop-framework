import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';
import * as fs from 'fs';

export interface CodeNode {
  id: string;
  type: 'function' | 'class' | 'interface' | 'variable';
  name: string;
  startLine: number;
  endLine: number;
  filePath: string;
  code: string;
}

export interface CodeEdge {
  sourceId: string;
  targetId: string;
  type: 'calls' | 'imports' | 'implements' | 'extends';
}

export class ASTExtractor {
  private parser: Parser;

  constructor() {
    this.parser = new Parser();
    this.parser.setLanguage(TypeScript.typescript);
  }

  public parseFile(filePath: string): { nodes: CodeNode[]; edges: CodeEdge[] } {
    const sourceCode = fs.readFileSync(filePath, 'utf8');
    const tree = this.parser.parse(sourceCode);
    
    const nodes: CodeNode[] = [];
    const edges: CodeEdge[] = [];

    // Basic traversal for demonstration
    // A production implementation would use tree-sitter queries
    this.traverse(tree.rootNode, filePath, sourceCode, nodes);

    return { nodes, edges };
  }

  private traverse(node: Parser.SyntaxNode, filePath: string, sourceCode: string, nodes: CodeNode[]) {
    if (node.type === 'function_declaration' || node.type === 'method_definition') {
      const nameNode = node.childForFieldName('name');
      if (nameNode) {
        nodes.push({
          id: `${filePath}:${nameNode.text}`,
          type: 'function',
          name: nameNode.text,
          startLine: node.startPosition.row + 1,
          endLine: node.endPosition.row + 1,
          filePath,
          code: sourceCode.substring(node.startIndex, node.endIndex)
        });
      }
    } else if (node.type === 'class_declaration') {
      const nameNode = node.childForFieldName('name');
      if (nameNode) {
        nodes.push({
          id: `${filePath}:${nameNode.text}`,
          type: 'class',
          name: nameNode.text,
          startLine: node.startPosition.row + 1,
          endLine: node.endPosition.row + 1,
          filePath,
          code: sourceCode.substring(node.startIndex, node.endIndex)
        });
      }
    }

    for (let i = 0; i < node.childCount; i++) {
      const child = node.child(i);
      if (child) {
        this.traverse(child, filePath, sourceCode, nodes);
      }
    }
  }
}
