"""
Quickstart example — JuicePlug with web search, no adapter.

Usage
-----
    python examples/quickstart.py

Requirements
------------
    pip install juiceplug[gpu]
    # or: pip install -r requirements.txt && pip install -e .
"""

from juiceplug import JuicedModel


def main() -> None:
    # Load a small instruct model with web search enabled.
    # Set reasoning_adapter to a real published adapter id when available,
    # e.g. "Arshveen-singh1/juiceplug-reasoning-general-4bit"
    model = JuicedModel.from_pretrained(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        reasoning_adapter=None,  # TODO: replace with a published adapter
        tools=["web_search"],
        load_in_4bit=True,
    )

    question = "What's the most recent stable release of PyTorch?"
    print(f"Question: {question}\n")
    answer = model.ask(question, verbose=True)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
