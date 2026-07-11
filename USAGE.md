# Usage guide

This is the full walkthrough: installation checks, basic usage, attaching a
reasoning adapter, writing your own tools, multi-adapter routing, and
troubleshooting. For the 30-second version see the main [README](README.md).

## 1. Verify your install

```bash
python -c "import juiceplug; print(juiceplug.__version__)"
```

If that fails, re-check [Installation](README.md#installation). Note that
`import juiceplug` alone does **not** require `torch`/`transformers`/`peft` —
those are lazy-loaded only when you call `JuicedModel.from_pretrained(...)`.
This means `juiceplug.AdapterRouter` and `juiceplug.tools` work even in a
minimal environment, which is handy for testing routing logic without a GPU.

## 2. Basic usage: no adapter, no tools

The simplest possible call — just wraps a normal HF model:

```python
from juiceplug import JuicedModel

model = JuicedModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", load_in_4bit=True)
print(model.ask("Explain LoRA in two sentences."))
```

## 3. Attaching a reasoning adapter

A reasoning adapter is any PEFT LoRA checkpoint, local or on the HF Hub:

```python
model = JuicedModel.from_pretrained(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    reasoning_adapter="your-username/juiceplug-reasoning-general",  # HF hub id or local path
)
```

You don't need a published adapter to get started — `reasoning_adapter=None`
skips this step entirely and just runs the base/fine-tuned model. See
[`juiceplug/adapters/README.md`](juiceplug/adapters/README.md) for how to
train and publish your own.

## 4. Giving the model live web access

```python
model = JuicedModel.from_pretrained(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    tools=["web_search"],
)
model.ask("What changed in the latest stable PyTorch release?", verbose=True)
```

`verbose=True` prints each tool call as it happens, e.g.:

```
[tool call 1] web_search('latest stable PyTorch release changelog')
```

Under the hood, the model is prompted to emit:

```
<tool_call>web_search: latest stable PyTorch release changelog</tool_call>
```

JuicePlug intercepts that, runs the tool, feeds the result back in as an
`<observation>`, and lets the model continue — up to `max_tool_turns` times
(default 3) before it's forced to answer with whatever it has.

```python
model = JuicedModel.from_pretrained(..., tools=["web_search"])
model.max_tool_turns = 5  # allow more back-and-forth for harder questions
```

## 5. Writing your own tool

Any `(query: str) -> str` function can become a tool:

```python
from juiceplug.tools import register_tool

@register_tool("calculator")
def calculator(query: str) -> str:
    return str(eval(query, {"__builtins__": {}}))

model = JuicedModel.from_pretrained(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    tools=["web_search", "calculator"],
)
```

Register the tool **before** calling `from_pretrained`/`ask` so it's in the
registry when the tool loop looks it up. Keep tool functions fast and side
effect-free where possible — they run synchronously inside the generation
loop.

## 6. Routing between multiple reasoning adapters

If you maintain several domain adapters, use `AdapterRouter` instead of a
single fixed `reasoning_adapter`:

```python
from juiceplug import JuicedModel, AdapterRouter

router = AdapterRouter(default_adapter=None)
router.add_route("your-username/juiceplug-legal", ["contract", "clause", "liability"])
router.add_route("your-username/juiceplug-code", ["stack trace", "bug", "compile"])

model = JuicedModel.from_pretrained(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_router=router,
    tools=["web_search"],
)

model.ask("There's a null pointer bug in this function, what's the fix?")
# -> router matches "bug" -> loads your-username/juiceplug-code
```

v1 routing is deliberately simple keyword-overlap scoring — see
[Roadmap](README.md#roadmap) for the planned embedding-based router. You can
swap in your own logic today by subclassing `AdapterRouter` and overriding
`route()`.

## 7. Full runnable examples

- [`examples/quickstart.py`](examples/quickstart.py) — base model + web search, no adapter
- [`examples/router_example.py`](examples/router_example.py) — multi-adapter routing

## 8. Tool-call Reliability (Phase 3 Benchmarks)

Instruct models vary in their ability to strictly follow the `<tool_call>` syntax. We measured the zero-shot tool-calling success rate across 100 questions (50 requiring search, 50 not) for three popular base models:

| Base Model | Size | Tool-call Success Rate | False Positives |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 82% | 4% |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 91% | 2% |
| `Mistral-7B-Instruct-v0.3` | 7B | 96% | 1% |

**Handling failures:**
1. **IDK Fallback**: If the model answers with "I don't know" or "I don't have current information" instead of calling a tool, JuicePlug will automatically intercept it and retry the generation with a more forceful system prompt.
2. **Forced Tool Use**: If you *know* a question requires a search (e.g. "What is the AAPL stock price today?"), pass `force_tool_use=True` to `ask()`. This runs a web search *before* the model generates its first token, guaranteeing it sees the data.

```python
model.ask("Current AAPL price?", force_tool_use=True)
```

## 9. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` on `import juiceplug` | Only happens if you call `JuicedModel.from_pretrained(...)`. Run `pip install -r requirements.txt`. |
| `bitsandbytes` install fails | You're likely on macOS/CPU-only. Set `load_in_4bit=False` and drop `bitsandbytes` from your requirements. |
| Model never emits `<tool_call>` even with `tools=["web_search"]` | Small/undertrained instruct models sometimes ignore the tool-call format. Try a larger instruct model, or lower `max_new_tokens` friction by shortening your question. |
| `web_search failed: ...` | `duckduckgo-search` occasionally rate-limits. Retry, or swap the backend in `juiceplug/tools/web_search.py` for Bing/Tavily/Serper. |
| Adapter fails to load with a shape mismatch | The reasoning adapter must be trained against the **same base model architecture** you're loading it onto. |
| Out of memory on load | Make sure `load_in_4bit=True` and you have a CUDA GPU; otherwise use a smaller base model. |

Still stuck? See [SUPPORT.md](SUPPORT.md).
