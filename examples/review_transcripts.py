"""
Phase 6: Transcript review and self-training data generation.

When JuicedModel is initialized with `log_transcripts=True`, it saves
all tool-use sessions to `transcripts/tool_use_log.jsonl`.

This script provides a lightweight CLI for a human to review those
transcripts. Accepted transcripts are formatted into high-quality
chain-of-thought examples and saved to `transcripts/approved_for_training.jsonl`.
These can later be merged into your training dataset for the reasoning adapter.

Usage:
    python examples/review_transcripts.py
"""

import json
import os
import sys

LOG_FILE = "transcripts/tool_use_log.jsonl"
OUT_FILE = "transcripts/approved_for_training.jsonl"


def load_jsonl(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(filepath: str, data: list[dict], append: bool = True) -> None:
    mode = "a" if append else "w"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_as_training_example(log_entry: dict) -> dict:
    """Convert a logged tool-use session into a CoT training example."""
    question = log_entry["question"]
    tool_calls = log_entry.get("tool_calls", [])
    final_answer = log_entry.get("final_answer", "")

    # Construct the rationale (chain of thought)
    rationale_parts = []
    for turn in tool_calls:
        t_name = turn["tool"]
        t_query = turn["query"]
        t_obs = turn.get("observation", "")
        rationale_parts.append(
            f"I need to use a tool to find this out.\n"
            f"<tool_call>{t_name}: {t_query}</tool_call>\n"
            f"Observation: {t_obs}"
        )

    rationale = "\n\n".join(rationale_parts)
    
    return {
        "instruction": question,
        "rationale": rationale,
        "response": final_answer,
        "source": "juiceplug_self_training_loop"
    }


def main():
    print(f"=== JuicePlug Transcript Reviewer ===")
    
    all_logs = load_jsonl(LOG_FILE)
    if not all_logs:
        print(f"No transcripts found at {LOG_FILE}. Enable log_transcripts=True in JuicedModel to start collecting data.")
        sys.exit(0)

    # Load already approved ones to avoid re-reviewing
    approved = load_jsonl(OUT_FILE)
    approved_questions = {a["instruction"] for a in approved}

    pending = [log for log in all_logs if log["question"] not in approved_questions]
    
    if not pending:
        print(f"All {len(all_logs)} transcripts have already been reviewed.")
        sys.exit(0)

    print(f"Found {len(pending)} new transcripts to review.\n")
    
    newly_approved = []

    for i, log in enumerate(pending, 1):
        print("-" * 60)
        print(f"Transcript {i}/{len(pending)}")
        print(f"Question: {log['question']}")
        print("-" * 60)
        
        for turn in log.get("tool_calls", []):
            print(f"🔧 Tool Call : {turn['tool']}({turn['query']!r})")
            obs = turn.get('observation', '')
            obs_preview = obs[:100] + "..." if len(obs) > 100 else obs
            print(f"👀 Observation: {obs_preview}")
            
        print("-" * 60)
        print(f"Final Answer:\n{log.get('final_answer', '')}")
        print("-" * 60)
        
        while True:
            choice = input("Approve this transcript for training? (y/n/q to quit): ").strip().lower()
            if choice in ('y', 'n', 'q'):
                break
            print("Invalid choice.")
            
        if choice == 'q':
            break
        elif choice == 'y':
            training_ex = format_as_training_example(log)
            newly_approved.append(training_ex)
            print("✅ Approved.")
        else:
            print("❌ Rejected.")
            
        print()
        
    if newly_approved:
        save_jsonl(OUT_FILE, newly_approved)
        print(f"Saved {len(newly_approved)} approved examples to {OUT_FILE}")
    else:
        print("No new examples approved.")


if __name__ == "__main__":
    main()
