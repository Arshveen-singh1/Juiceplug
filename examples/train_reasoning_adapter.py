"""
Train a general chain-of-thought reasoning adapter for JuicePlug.

This script fine-tunes a small base model on chain-of-thought reasoning
examples using LoRA, producing a lightweight adapter that can be loaded
via ``JuicedModel.from_pretrained(reasoning_adapter=...)``.

Dataset
-------
Uses ``kaist-ai/CoT-Collection`` from the Hugging Face Hub — a curated
collection of chain-of-thought reasoning examples covering math, logic,
commonsense, and science domains.

Usage
-----
    # Train (requires GPU with >=8 GB VRAM):
    python examples/train_reasoning_adapter.py

    # Evaluate against held-out questions:
    python examples/train_reasoning_adapter.py --eval-only --adapter-path ./adapters_local/juiceplug-reasoning-general

    # Push to HF Hub after training:
    python examples/train_reasoning_adapter.py --push-to-hub --hub-id your-username/juiceplug-reasoning-general-4bit

Requirements
------------
    pip install juiceplug[train]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET = "kaist-ai/CoT-Collection"
DEFAULT_OUTPUT_DIR = "./adapters_local/juiceplug-reasoning-general"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-4

EVAL_PROMPTS = [
    "What is 847 minus 389? Show your reasoning step by step.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think step by step.",
    "A farmer has 17 sheep. All but 9 die. How many are left? Reason through this carefully.",
    "Sort these numbers from smallest to largest: 42, 7, 19, 3, 88. Show your process.",
]

def load_and_format_dataset(dataset_name: str, max_samples: int, tokenizer: Any) -> Any:
    from datasets import load_dataset

    logger.info("Loading dataset: %s (max_samples=%d)", dataset_name, max_samples)
    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    system_msg = (
        "You are a helpful reasoning assistant. When asked a question, "
        "think through it step by step before giving your final answer. "
        "Show your reasoning clearly."
    )

    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        question = example.get("source", example.get("instruction", ""))
        rationale = example.get("rationale", "")
        answer = example.get("target", example.get("answer", ""))

        response = f"{rationale}\n\nFinal Answer: {answer}" if rationale else answer

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = f"<|system|>{system_msg}<|user|>{question}<|assistant|>{response}"
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds.column_names)
    logger.info("Dataset formatted: %d examples", len(ds))
    return ds

def train_adapter(
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_name: str = DEFAULT_DATASET,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> str:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Training requires a GPU.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_and_format_dataset(dataset_name, max_samples, tokenizer)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        learning_rate=learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.1,
        weight_decay=0.01, logging_steps=10, bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), optim="paged_adamw_8bit",
        report_to="none", gradient_checkpointing=True, max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer, max_seq_length=max_seq_len,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "base_model": base_model, "dataset": dataset_name, "max_samples": max_samples,
                "lora_rank": lora_rank, "lora_alpha": lora_alpha, "epochs": epochs,
                "batch_size": batch_size, "learning_rate": learning_rate, "max_seq_len": max_seq_len,
            },
            f, indent=2,
        )
    return output_dir

def evaluate_adapter(
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_path: Optional[str] = DEFAULT_OUTPUT_DIR,
    prompts: Optional[List[str]] = None,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    prompts = prompts or EVAL_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    def generate_answer(mdl, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant. Think step by step before giving your final answer."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            output_ids = mdl.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    results = []
    print("\n" + "=" * 80 + "\nEVALUATION: base model vs. base + adapter\n" + "=" * 80)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}/{len(prompts)} ---\nQ: {prompt}\n")
        base_answer = generate_answer(model, prompt)
        print(f"[BASE MODEL]\n{base_answer}\n")
        result = {"prompt": prompt, "base_answer": base_answer}

        if adapter_path and os.path.isdir(adapter_path):
            adapted_model = PeftModel.from_pretrained(model, adapter_path)
            adapted_answer = generate_answer(adapted_model, prompt)
            print(f"[BASE + ADAPTER]\n{adapted_answer}\n")
            result["adapter_answer"] = adapted_answer
            del adapted_model
            torch.cuda.empty_cache()

        results.append(result)

    output_path = os.path.join(adapter_path or ".", "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def push_to_hub(adapter_path: str, hub_id: str) -> None:
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.push_to_hub(hub_id)

    api = HfApi()
    api.upload_folder(folder_path=adapter_path, repo_id=hub_id, repo_type="model")
    logger.info("Adapter pushed to: https://huggingface.co/%s", hub_id)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a JuicePlug reasoning adapter (LoRA on CoT data).")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-id", default=None)

    args = parser.parse_args()

    if args.eval_only:
        evaluate_adapter(base_model=args.base_model, adapter_path=args.adapter_path or args.output_dir)
        return

    output = train_adapter(
        base_model=args.base_model, dataset_name=args.dataset, output_dir=args.output_dir,
        max_samples=args.max_samples, lora_rank=args.lora_rank, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
    )
    evaluate_adapter(base_model=args.base_model, adapter_path=output)

    if args.push_to_hub:
        if not args.hub_id:
            logger.error("--hub-id is required when using --push-to-hub")
            sys.exit(1)
        push_to_hub(output, args.hub_id)

if __name__ == "__main__":
    main()
