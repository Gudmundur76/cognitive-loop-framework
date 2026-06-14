import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';

const parser = new Parser();
parser.setLanguage(TypeScript.typescript);

const sourceCode = 'let x = 1; console.log(x);';
const tree = parser.parse(sourceCode);

console.log('AST root node type:', tree.rootNode.type);
console.log('AST child count:', tree.rootNode.childCount);
console.log('Tree-sitter is working successfully!');
