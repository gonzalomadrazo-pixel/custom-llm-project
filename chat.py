"""A terminal interface to your saved nanoGPT. Each prompt starts a fresh context."""
import argparse
import json
from pathlib import Path

from run_evals import generate_reply, load_model, model_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--transcript", type=Path, default=Path("chat_transcript.json"))
    args = parser.parse_args()
    if args.transcript.exists():
        parser.error("Choose a new --transcript filename to preserve previous conversations.")
    model, vocabulary, saved = load_model(args.model)
    record = {"model": str(args.model), "model_sha256": model_hash(model),
              "completed_steps": saved.get("completed_steps"), "fresh_context_per_prompt": True,
              "temperature": 0.8, "max_tokens": 24, "turns": []}
    print("Tiny language model: short continuations, not a general assistant.")
    print(f"Each prompt starts fresh. Context: {model.config.block_size} tokens. Type /quit to exit.")
    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                break
            if prompt == "/quit":
                break
            if not prompt:
                continue
            seed = 2026 + len(record["turns"])
            reply = generate_reply(model, vocabulary, prompt, seed=seed)
            print("Model:", reply["response"] or "[empty response]")
            if reply["unknown_prompt_words"]:
                print("Unknown words:", ", ".join(reply["unknown_prompt_words"]))
            if reply["prompt_truncated"]:
                print("Long prompt: only the most recent context tokens were used.")
            record["turns"].append({"prompt": prompt, "seed": seed, **reply})
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        args.transcript.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Saved transcript:", args.transcript)


if __name__ == "__main__":
    main()
