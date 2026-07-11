"""
Router example — multi-adapter routing with JuicePlug.

Demonstrates how ``AdapterRouter`` picks the right LoRA adapter
based on keyword overlap with the user's query.

Usage
-----
    python examples/router_example.py

Requirements
------------
    pip install juiceplug[gpu]
"""

from juiceplug import AdapterRouter, EmbeddingAdapterRouter, JuicedModel


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Set up a keyword-based router
    # ------------------------------------------------------------------
    router = AdapterRouter(default_adapter=None)

    # Register routes — replace these with real published adapter ids
    router.add_route(
        "your-username/juiceplug-reasoning-general",
        ["reason", "think", "step by step", "explain", "why"],
    )
    router.add_route(
        "your-username/juiceplug-code",
        ["code", "bug", "stack trace", "compile", "function", "debug"],
    )
    router.add_route(
        "your-username/juiceplug-legal",
        ["contract", "clause", "liability", "statute", "regulation"],
    )

    # ------------------------------------------------------------------
    # 2.  Show routing decisions (no GPU needed for this part)
    # ------------------------------------------------------------------
    test_queries = [
        "Explain step by step how backpropagation works.",
        "There's a null pointer bug in this function, what's the fix?",
        "Is this liability clause enforceable under California law?",
        "What's the weather like today?",  # -> default (None)
    ]

    print("=== V1: Keyword-overlap Routing ===")
    for q in test_queries:
        adapter = router.route(q)
        print(f"  Q: {q}")
        print(f"  -> {adapter or '(base model, no adapter)'}\n")

    # ------------------------------------------------------------------
    # 3.  Set up an embedding-based router (requires [router] extra)
    # ------------------------------------------------------------------
    try:
        embed_router = EmbeddingAdapterRouter(default_adapter=None, threshold=0.2)
        # Note: We can pass actual descriptions instead of just keywords!
        embed_router.add_route(
            "your-username/juiceplug-reasoning-general",
            ["Provides step by step logical reasoning, explanation, and thought process."]
        )
        embed_router.add_route(
            "your-username/juiceplug-code",
            ["Code debugging, writing functions, fixing bugs, and reading stack traces."]
        )
        embed_router.add_route(
            "your-username/juiceplug-legal",
            ["Legal contracts, liability clauses, statutes, and regulation analysis."]
        )

        print("=== V2: Embedding-based Routing ===")
        for q in test_queries:
            adapter = embed_router.route(q)
            print(f"  Q: {q}")
            print(f"  -> {adapter or '(base model, no adapter)'}\n")
    except ImportError:
        print("Skipping EmbeddingAdapterRouter demo (sentence-transformers not installed).")

    # ------------------------------------------------------------------
    # 4.  (Optional) Load a model with the router
    # ------------------------------------------------------------------
    # Uncomment the lines below once you have real adapters published:
    #
    # model = JuicedModel.from_pretrained(
    #     base_model="Qwen/Qwen2.5-1.5B-Instruct",
    #     adapter_router=router,
    #     tools=["web_search"],
    #     load_in_4bit=True,
    # )
    # answer = model.ask("Explain step by step how LoRA works.")
    # print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
