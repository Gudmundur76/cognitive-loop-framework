import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';

const parser = new Parser();
parser.setLanguage(TypeScript.typescript);

const sourceCode = 'let x = 1; console.log(x);';
const tree = parser.parse(sourceCode);

console.log(tree.rootNode.toString());
