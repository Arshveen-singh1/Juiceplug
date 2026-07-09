# Contributing to JuicePlug

Thanks for considering a contribution — this project is early and there's a
lot of open surface area (see the [Roadmap](README.md#roadmap)).

## Setup

```bash
git clone https://github.com/YOURNAME/juiceplug.git
cd juiceplug
pip install -r requirements-dev.txt
pip install -e .
pytest tests/
```

## Ways to contribute

- **Tools** — add a new tool under `juiceplug/tools/` (calculator, code
  execution, a real search API, a vector-store retriever, etc.) using the
  `@register_tool("name")` pattern in `juiceplug/tools/__init__.py`.
- **Reasoning adapters** — train and publish a domain adapter following
  [`juiceplug/adapters/README.md`](juiceplug/adapters/README.md), then open
  a PR linking it from the main README so others can find it.
- **Router improvements** — the v1 `AdapterRouter` is keyword-overlap only;
  an embedding-similarity router is on the roadmap.
- **Bug fixes / tests** — anything in `tests/` failing, or missing coverage
  for `core.py`'s tool loop, is fair game.

## Pull request checklist

1. Fork the repo and create a branch: `git checkout -b feature/my-change`
2. Run `pytest tests/` and make sure everything passes
3. Run `black .` and `ruff check .` to keep formatting consistent
4. Keep PRs focused — one feature or fix per PR is much easier to review
5. Describe *what* changed and *why* in the PR description

## Code style

- Format with `black`, lint with `ruff` (both in `requirements-dev.txt`)
- Type hints on public functions where practical
- Keep tool functions synchronous, side-effect-light, and defensive about
  exceptions (they run inside the generation loop — an unhandled exception
  there breaks the whole `ask()` call)

## Reporting issues

Open a GitHub Issue with:
- What you ran (base model, adapter, tools enabled)
- What you expected vs. what happened
- Full error traceback if there was one

See also [SUPPORT.md](SUPPORT.md) for general help vs. bug reports.
