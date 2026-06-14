import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { ASTExtractor } from '../../src/indexer/extractor';
import * as fs from 'fs';
import * as path from 'path';

describe('ASTExtractor', () => {
  const testFilePath = path.join(__dirname, 'test-fixture.ts');

  beforeAll(() => {
    const fixtureCode = `
      export class TestClass {
        public testMethod() {
          console.log('hello');
        }
      }

      function standaloneFunction() {
        return true;
      }
    `;
    fs.writeFileSync(testFilePath, fixtureCode);
  });

  afterAll(() => {
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }
  });

  it('extracts classes and functions correctly', () => {
    const extractor = new ASTExtractor();
    const result = extractor.parseFile(testFilePath);

    expect(result.nodes).toHaveLength(3);
    
    const classNode = result.nodes.find(n => n.type === 'class');
    expect(classNode).toBeDefined();
    expect(classNode?.name).toBe('TestClass');

    const methodNode = result.nodes.find(n => n.name === 'testMethod');
    expect(methodNode).toBeDefined();
    expect(methodNode?.type).toBe('function');

    const funcNode = result.nodes.find(n => n.name === 'standaloneFunction');
    expect(funcNode).toBeDefined();
    expect(funcNode?.type).toBe('function');
  });
});
