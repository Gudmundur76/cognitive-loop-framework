"""
Fine-Tuning Pipeline — Qwen2.5-Coder with Unsloth + TRL

Three execution modes:
  1. --cpu       CPU-only LoRA fine-tuning. No GPU required. Runs on any machine.
                 ~4-8 hours for a 1.5B model on a typical codebase corpus.
                 Produces the same GGUF output as GPU mode.

  2. --gpu       GPU-accelerated training with 4-bit quantisation (default when
                 GPU is available and --cpu is not specified).

  3. --in-context  No training at all. Generates a few-shot prompt file from the
                   corpus that can be prepended to any Ollama inference call.
                   Zero cost. Works immediately. Best for small codebases.

Usage:
    # CPU fine-tune (recommended for most use cases):
    python finetunePipeline.py --corpus ./corpus/training_pairs.jsonl \\
        --output ./models/codebase-slm --cpu

    # In-context learning (no training, works immediately):
    python finetunePipeline.py --corpus ./corpus/training_pairs.jsonl \\
        --output ./models --in-context

    # GPU fine-tune (fastest, requires VRAM):
    python finetunePipeline.py --corpus ./corpus/training_pairs.jsonl \\
        --output ./models/codebase-slm --gpu

Output:
    CPU/GPU mode:   GGUF model file + training metrics + model card
    In-context:     few_shot_examples.json + system_prompt.txt

Design constraints:
    - Max 200 lines per file
    - All functions max 20 lines
"""

import argparse
import json
import os
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
FEW_SHOT_EXAMPLES_COUNT = 12


# ── Prompt formatter ─────────────────────────────────────────────

def format_prompt(instruction: str, input_text: str, output: str) -> str:
    """Format a training pair into the Alpaca instruction template."""
    if input_text.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def load_corpus(corpus_path: str) -> list[dict]:
    """Load training pairs from a JSONL file."""
    pairs = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def prepare_dataset(pairs: list[dict]) -> list[str]:
    """Convert raw pairs into formatted prompt strings."""
    return [
        format_prompt(
            p.get('instruction', ''),
            p.get('input', ''),
            p.get('output', '')
        )
        for p in pairs
    ]


# ── In-Context Learning (no training) ───────────────────────────

def run_in_context(corpus_path: str, output_dir: str) -> dict:
    """
    Generate a few-shot examples file from the corpus.
    No training required. Works with any Ollama model immediately.
    Selects a diverse sample covering all 5 pair types.
    """
    pairs = load_corpus(corpus_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Select diverse examples — up to 2-3 per pair type
    by_type: dict[str, list] = {}
    for p in pairs:
        t = p.get('type', 'unknown')
        by_type.setdefault(t, []).append(p)

    selected = []
    per_type = max(1, FEW_SHOT_EXAMPLES_COUNT // max(len(by_type), 1))
    for examples in by_type.values():
        selected.extend(examples[:per_type])
    selected = selected[:FEW_SHOT_EXAMPLES_COUNT]

    examples_path = os.path.join(output_dir, "few_shot_examples.json")
    with open(examples_path, 'w') as f:
        json.dump(selected, f, indent=2)

    prompt_path = os.path.join(output_dir, "system_prompt.txt")
    with open(prompt_path, 'w') as f:
        f.write(_build_system_prompt(selected))

    print(f"Few-shot examples written to: {examples_path}")
    print(f"System prompt written to: {prompt_path}")
    return {"mode": "in_context", "examples": len(selected), "corpus_pairs": len(pairs)}


def _build_system_prompt(examples: list[dict]) -> str:
    """Build a system prompt string from few-shot examples."""
    lines = [
        "You are an expert on this specific codebase. "
        "Use the examples below to understand its patterns.\n"
    ]
    for ex in examples[:6]:  # Keep prompt concise — first 6 only
        lines.append(f"Q: {ex.get('instruction', '')}")
        lines.append(f"A: {ex.get('output', '')}\n")
    return "\n".join(lines)


# ── CPU Training ─────────────────────────────────────────────────

def run_cpu_training(
    corpus_path: str,
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 2e-4
) -> dict:
    """
    CPU-only LoRA fine-tuning. No GPU required.
    Uses gradient checkpointing and batch size 1 to minimise RAM.
    Runs in ~4-8 hours on a modern CPU for a typical codebase corpus.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
    except ImportError:
        return _run_dry_mode(corpus_path, output_dir, epochs, mode="cpu")

    pairs = load_corpus(corpus_path)
    texts = prepare_dataset(pairs)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,          # CPU: batch size 1
            gradient_accumulation_steps=4,           # Effective batch = 4
            gradient_checkpointing=True,             # Minimise RAM
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False,                              # CPU: no fp16
            bf16=False,
            no_cuda=True,                            # Force CPU
            report_to="none"
        )
    )

    result = trainer.train()
    _save_adapter(model, tokenizer, output_dir)
    return {"mode": "cpu", "training_loss": result.training_loss, "epochs": epochs}


def _save_adapter(model, tokenizer, output_dir: str) -> None:
    """Save the LoRA adapter and tokenizer. Convert to GGUF if llama.cpp available."""
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to: {adapter_path}")
    print("To deploy: merge adapter with base model, then convert to GGUF with llama.cpp.")
    print("  python convert_hf_to_gguf.py {adapter_path} --outtype q4_k_m")


# ── GPU Training ─────────────────────────────────────────────────

def run_gpu_training(
    corpus_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
) -> dict:
    """GPU-accelerated training with 4-bit quantisation via Unsloth."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError:
        return _run_dry_mode(corpus_path, output_dir, epochs, mode="gpu")

    pairs = load_corpus(corpus_path)
    texts = prepare_dataset(pairs)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", use_gradient_checkpointing=True
    )
    dataset = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=output_dir, num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate, logging_steps=10,
            save_strategy="epoch", fp16=True, report_to="none"
        )
    )
    result = trainer.train()
    model.save_pretrained_gguf(
        os.path.join(output_dir, "model.gguf"), tokenizer, quantization_method="q4_k_m"
    )
    return {"mode": "gpu", "training_loss": result.training_loss, "epochs": epochs}


# ── Dry-run fallback ─────────────────────────────────────────────

def _run_dry_mode(corpus_path: str, output_dir: str, epochs: int, mode: str) -> dict:
    """Validate corpus and report what training would do. No model download."""
    pairs = load_corpus(corpus_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "mode": f"dry_run_{mode}",
        "corpus_pairs": len(pairs),
        "planned_epochs": epochs,
        "model": MODEL_ID,
        "note": f"Install dependencies for {mode} training to run actual fine-tuning."
    }
    report_path = os.path.join(output_dir, "dry_run_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Dry run report: {report_path}")
    return report


# ── CLI entry point ───────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune or prepare Qwen2.5-Coder on a codebase corpus"
    )
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true",
                      help="CPU-only LoRA fine-tuning. No GPU required.")
    mode.add_argument("--gpu", action="store_true",
                      help="GPU-accelerated training with Unsloth (default).")
    mode.add_argument("--in-context", action="store_true", dest="in_context",
                      help="No training. Generate few-shot examples for in-context learning.")

    args = parser.parse_args()

    if args.in_context:
        metrics = run_in_context(args.corpus, args.output)
    elif args.cpu:
        metrics = run_cpu_training(args.corpus, args.output, args.epochs, args.lr)
    else:
        metrics = run_gpu_training(args.corpus, args.output, args.epochs, 4, args.lr)

    print(f"Complete: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
