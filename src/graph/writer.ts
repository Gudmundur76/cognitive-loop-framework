import { CodeNode, CodeEdge } from '../indexer/extractor';

export interface GraphExport {
  nodes: Array<{
    id: string;
    type: string;
    properties: Record<string, any>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    type: string;
    properties: Record<string, any>;
  }>;
}

export class GraphWriter {
  public formatForGraph(nodes: CodeNode[], edges: CodeEdge[]): GraphExport {
    const formattedNodes = nodes.map(node => ({
      id: node.id,
      type: 'CodeNode',
      properties: {
        nodeType: node.type,
        name: node.name,
        filePath: node.filePath,
        startLine: node.startLine,
        endLine: node.endLine,
        code: node.code,
        // The framework will eventually compute embeddings for this field
        embedding_status: 'pending'
      }
    }));

    const formattedEdges = edges.map(edge => ({
      source: edge.sourceId,
      target: edge.targetId,
      type: 'CODE_RELATION',
      properties: {
        relationType: edge.type
      }
    }));

    return {
      nodes: formattedNodes,
      edges: formattedEdges
    };
  }

  public generateCypherInsert(graph: GraphExport): string {
    let cypher = '';
    
    // Generate node MERGE statements
    for (const node of graph.nodes) {
      const props = JSON.stringify(node.properties).replace(/"([^"]+)":/g, '$1:');
      cypher += `MERGE (n:CodeNode {id: "${node.id}"}) SET n += ${props};\n`;
    }

    // Generate edge MERGE statements
    for (const edge of graph.edges) {
      const props = JSON.stringify(edge.properties).replace(/"([^"]+)":/g, '$1:');
      cypher += `MATCH (source:CodeNode {id: "${edge.source}"})\n`;
      cypher += `MATCH (target:CodeNode {id: "${edge.target}"})\n`;
      cypher += `MERGE (source)-[r:CODE_RELATION]->(target) SET r += ${props};\n`;
    }

    return cypher;
  }
}
