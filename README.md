# JuicePlug ⚡

[![PyPI version](https://img.shields.io/pypi/v/juiceplug.svg)](https://pypi.org/project/juiceplug/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/Arshveen-singh1/Juiceplug/actions/workflows/tests.yml/badge.svg)](https://github.com/Arshveen-singh1/Juiceplug/actions/workflows/tests.yml)

**Plug-and-play reasoning adapters + live web tools for any fine-tuned or from-scratch LLM.**

JuicePlug lets you take a model you already trained — or a stock base model — and bolt on, without retraining it from scratch:

- 🧠 **A quantized reasoning adapter** — a small LoRA checkpoint trained on domain chain-of-thought data, loaded 4-bit so it's cheap and fast.
- 🌐 **Live web access** — a tool-calling loop so the model can search the web mid-answer instead of relying only on frozen training data.
- 🧭 **Automatic adapter routing** — if you maintain several domain adapters (legal, code, medical...), `AdapterRouter` picks the right one per query.

It does **not** (yet) auto-train itself from the open web unsupervised — that's the riskiest part of this idea (data quality, catastrophic forgetting, cost) and is intentionally out of scope for v1. See [Roadmap](#roadmap).

---

## Table of contents

- [Why JuicePlug](#why-juiceplug)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Usage guide](#usage-guide)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Getting help](#getting-help)
- [License](#license)

## Why JuicePlug

Most people fine-tune a model, then hit three walls at once: it can't see anything past its training cutoff, it's not especially strong at multi-step reasoning in their specific domain, and bolting on a full RAG + reasoning stack from scratch is a lot of glue code. JuicePlug is that glue code, packaged as a small, inspectable library instead of a framework you have to fight.

## Requirements

- Python 3.9+
- A CUDA GPU is **strongly recommended** for 4-bit loading (`bitsandbytes`); CPU works but is slow and `load_in_4bit` should be set to `False`
- See [`requirements.txt`](requirements.txt) for exact package versions

## Installation

```bash
git clone https://github.com/YOURNAME/juiceplug.git
cd juiceplug
pip install -r requirements.txt
pip install -e .
```

Contributors / running tests:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

> **Apple Silicon / no GPU?** `bitsandbytes` 4-bit quantization needs CUDA. Skip it: pass `load_in_4bit=False` to `from_pretrained` and drop `bitsandbytes` from your install.

## Quickstart

```python
from juiceplug import JuicedModel

model = JuicedModel.from_pretrained(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",   # swap for your own fine-tuned model
    reasoning_adapter=None,                     # e.g. "you/juiceplug-reasoning-general"
    tools=["web_search"],
    load_in_4bit=True,
)

print(model.ask("What's the most recent stable release of PyTorch?"))
```

Runnable version: [`examples/quickstart.py`](examples/quickstart.py).

For the full walkthrough — attaching your own reasoning adapter, writing custom tools, multi-adapter routing, and troubleshooting — see **[USAGE.md](USAGE.md)**.

## How it works

```
your model  ──►  + quantized LoRA reasoning adapter  ──►  JuicedModel
                            │
                  tool-calling loop (<tool_call> tags)
                            │
                     web_search / your own tools
```

`JuicedModel.ask()` runs a small ReAct-style loop: the model either answers directly or emits `<tool_call>tool_name: query</tool_call>`, JuicePlug executes the tool and feeds the result back in, up to `max_tool_turns` times.

## Usage guide

The short version is above. For the complete guide — including how to attach a custom reasoning adapter, register your own tools, use `AdapterRouter` for multi-domain routing, and common errors — see **[USAGE.md](USAGE.md)**.

## Project structure

```
juiceplug/
├── juiceplug/
│   ├── __init__.py       # public API (JuicedModel, AdapterRouter)
│   ├── core.py            # JuicedModel: loading, quantization, tool loop
│   ├── router.py           # AdapterRouter: picks a reasoning adapter per query
│   ├── tools/               # pluggable tool registry (web_search, your own)
│   └── adapters/             # docs + drop-in location for trained LoRA adapters
├── examples/                  # runnable quickstart + router examples
├── tests/                      # unit tests (no GPU required)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── README.md
├── USAGE.md
├── CONTRIBUTING.md
└── SUPPORT.md
```

## Roadmap

- [x] Quantized LoRA reasoning adapters (PEFT-compatible)
- [x] Web search tool loop
- [x] Keyword-based adapter router
- [ ] Embedding-based adapter router
- [ ] Reference reasoning adapters (general / code / legal) published on HF Hub
- [ ] Opt-in, human-reviewed self-training loop from tool-use transcripts

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Getting help

See [SUPPORT.md](SUPPORT.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
