import { describe, it, expect } from "vitest";

// ─── ScaffoldEvaluator ────────────────────────────────────────────────────────

describe("ScaffoldEvaluator", () => {
  it("exports ScaffoldEvaluator class and createScaffoldEvaluator factory", async () => {
    const mod = await import("./scaffoldEvaluator");
    expect(mod.ScaffoldEvaluator).toBeDefined();
    expect(mod.createScaffoldEvaluator).toBeDefined();
  });

  it("has enqueue, evaluatePending, and recentVerdicts methods", async () => {
    const { ScaffoldEvaluator } = await import("./scaffoldEvaluator");
    // ScaffoldEvaluator requires a MetaAgent — use a minimal mock
    const mockMeta = { publishEvent: () => {} } as any;
    const evaluator = new ScaffoldEvaluator(mockMeta);
    expect(typeof evaluator.enqueue).toBe("function");
    expect(typeof evaluator.evaluatePending).toBe("function");
    expect(typeof evaluator.recentVerdicts).toBe("function");
  });

  it("enqueue adds a proposal and recentVerdicts is initially empty", async () => {
    const { ScaffoldEvaluator } = await import("./scaffoldEvaluator");
    const mockMeta = { publishEvent: () => {} } as any;
    const evaluator = new ScaffoldEvaluator(mockMeta);
    const dreamResponse = { output: "Test scaffold proposal", reasoning: "", model: "kimi" } as any;
    evaluator.enqueue(dreamResponse, "auth-test-failure", 42);
    expect(evaluator.recentVerdicts(10)).toHaveLength(0);
  });
});

// ─── CTCMemory ────────────────────────────────────────────────────────────────

describe("CTCMemory", () => {
  it("exports CTCMemory class and singleton", async () => {
    const mod = await import("../memory/ctcMemory");
    expect(mod.CTCMemory).toBeDefined();
    expect(mod.ctcMemory).toBeDefined();
  });

  it("has ingestCycle, reconstruct, temporalQuery, and edgesByTag methods", async () => {
    const { CTCMemory } = await import("../memory/ctcMemory");
    const mem = new CTCMemory();
    expect(typeof mem.ingestCycle).toBe("function");
    expect(typeof mem.reconstruct).toBe("function");
    expect(typeof mem.temporalQuery).toBe("function");
    expect(typeof mem.edgesByTag).toBe("function");
  });
});
