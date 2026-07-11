# Adapters directory

This folder is the drop-in location for trained LoRA reasoning adapters.

## Using a published adapter

Pass the Hugging Face Hub id directly — no need to download anything
manually:

```python
model = JuicedModel.from_pretrained(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    reasoning_adapter="your-username/juiceplug-reasoning-general",
)
```

## Training your own adapter

See `examples/train_reasoning_adapter.py` for a complete training script
using PEFT + TRL.

### Quick checklist

1. **Pick a domain** — general CoT, code debugging, legal reasoning, etc.
2. **Prepare a dataset** — 500–2,000 chain-of-thought examples is a good
   starting point.  Format: each example should have an `instruction` and a
   `response` that walks through reasoning step by step.
3. **Train** — use LoRA (rank 8–16) on a small base model (1–3B params) in
   4-bit to keep VRAM reasonable.
4. **Evaluate** — run 10–20 held-out questions through the base model vs.
   base + adapter and compare quality.
5. **Publish** — push to the Hugging Face Hub:
   ```python
   model.push_to_hub("your-username/juiceplug-reasoning-DOMAIN")
   ```

## Adapter naming convention

`juiceplug-reasoning-{domain}-{quantization}`

Examples:
- `juiceplug-reasoning-general-4bit`
- `juiceplug-reasoning-code-4bit`
- `juiceplug-reasoning-legal-4bit`
