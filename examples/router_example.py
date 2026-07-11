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

from juiceplug import AdapterRouter, JuicedModel


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

    print("=== Routing decisions ===\n")
    for q in test_queries:
        adapter = router.route(q)
        print(f"  Q: {q}")
        print(f"  -> {adapter or '(base model, no adapter)'}\n")

    # ------------------------------------------------------------------
    # 3.  (Optional) Load a model with the router
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
