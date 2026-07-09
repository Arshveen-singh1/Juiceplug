# Getting help

## Before opening an issue

1. Check [USAGE.md](USAGE.md#8-troubleshooting) — most install/runtime
   problems (bitsandbytes on CPU, missing tool calls, OOM) are covered there.
2. Search [existing Issues](../../issues) to see if it's already reported.

## Where to ask

- **Bugs** (something errors or behaves incorrectly) → open an
  [Issue](../../issues/new). Include: base model, whether you set a
  `reasoning_adapter`/`adapter_router`, which `tools` were enabled, and the
  full traceback.
- **Usage questions** ("how do I...") → open a
  [Discussion](../../discussions) if enabled, otherwise an Issue tagged
  `question`.
- **Feature requests** → open an Issue tagged `enhancement`. Check the
  [Roadmap](README.md#roadmap) first — it may already be planned.
- **Security concerns** → do not open a public issue for anything that
  looks like a security vulnerability; contact the maintainer directly
  (add your contact method here once the repo is live).

## What to include in a bug report

```
- juiceplug version: (python -c "import juiceplug; print(juiceplug.__version__)")
- base_model:
- reasoning_adapter / adapter_router:
- tools enabled:
- GPU / load_in_4bit setting:
- Full error traceback:
```

The more of this you include, the faster it can be diagnosed.
