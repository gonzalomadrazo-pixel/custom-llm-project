"""Fixed synthetic language tests. This module never trains or builds a vocabulary."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE = ROOT / "evals/language_evals.json"


def word_tokens(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def normalized(text):
    return " " + " ".join(word_tokens(text)) + " "


def load_suite(path=DEFAULT_SUITE):
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = suite["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Eval case IDs must be unique and the suite must not be empty.")
    for case in cases:
        choices = case["choices"]
        if len(choices) != 4 or len(set(choices)) != 4 or case["answer"] not in choices:
            raise ValueError(f"{case['id']}: require four distinct choices and one answer.")
        if not word_tokens(case["prompt"]) or any(len(word_tokens(c)) != 1 for c in choices):
            raise ValueError(f"{case['id']}: use a nonempty prompt and single-token choices.")
    return suite


def suite_hash(suite):
    data = json.dumps(suite, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def matching_cases(text, suite):
    content = normalized(text)
    return [case["id"] for case in suite["cases"] if normalized(case["prompt"]) in content]


def reject_eval_leakage(text, suite, source):
    matches = matching_cases(text, suite)
    if matches:
        raise ValueError(f"Eval leakage in {source}: {', '.join(matches)}. "
                         "Remove the exact test prompts. Write different teaching examples.")


def reserve_classroom_passages(passages, suite):
    """Withhold all starter sentences containing a test prefix BEFORE the data split."""
    retained, excluded = [], []
    for passage in passages:
        matches = matching_cases(passage, suite)
        if matches:
            excluded.append({"case_ids": matches})
        else:
            retained.append(passage)
    return retained, {"excluded_passages": len(excluded),
                      "case_ids": sorted({i for row in excluded for i in row["case_ids"]}),
                      "suite_sha256": suite_hash(suite),
                      "method": "normalized contiguous prompt match; not a semantic leakage detector"}


def validate_corpus_location(folder, suite_path=DEFAULT_SUITE):
    root, tests = Path(folder).resolve(), Path(suite_path).resolve()
    if root == ROOT or root == tests.parent or root in tests.parents or tests.parent in root.parents:
        raise ValueError("Keep CORPUS_FOLDER separate from the project root and evals/. Use corpus/.")


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(path):
    from nanogpt_model import GPT, GPTConfig
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**saved["model_args"]))
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved["vocabulary"], saved


@torch.inference_mode()
def generate_reply(model, vocabulary, prompt, *, seed=2026, temperature=0.8, max_tokens=24):
    if not math.isfinite(temperature) or temperature <= 0 or max_tokens < 1:
        raise ValueError("Use a positive temperature and output limit.")
    stoi = {word: i for i, word in enumerate(vocabulary)}
    tokens = word_tokens(prompt)
    ids = [stoi["<BOS>"]] + [stoi.get(t, stoi["<UNK>"]) for t in tokens]
    unknown = sorted(set(t for t in tokens if t not in stoi))
    limit = model.config.block_size
    truncated = len(ids) > limit
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    was_training = model.training
    output = []
    try:
        model.eval()
        for _ in range(max_tokens):
            logits = model(torch.tensor([ids[-limit:]], device=device))[0][0, -1].float().cpu().clone()
            # BOS is structural, not a generated word; EOS still ends generation.
            logits[stoi["<BOS>"]] = -float("inf")
            next_id = torch.multinomial(torch.softmax(logits / temperature, -1), 1,
                                       generator=generator).item()
            if next_id == stoi["<EOS>"]:
                break
            ids.append(next_id)
            output.append(vocabulary[next_id])
    finally:
        model.train(was_training)
    return {"response": " ".join(output), "unknown_prompt_words": unknown,
            "prompt_truncated": truncated}


@torch.inference_mode()
def evaluate_suite(model, vocabulary, suite, output_dir, *, stage="final", seed=2026,
                   temperature=0.8, max_tokens=24):
    """Score the model's next-word probabilities; separately save unconstrained text."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stoi = {word: i for i, word in enumerate(vocabulary)}
    device = next(model.parameters()).device
    identity = model_hash(model)
    rows = []
    was_training = model.training
    try:
        model.eval()
        for index, case in enumerate(suite["cases"]):
            prompt_tokens = word_tokens(case["prompt"])
            unknown_prompt = sorted(set(t for t in prompt_tokens if t not in stoi))
            unknown_choices = [c for c in case["choices"] if word_tokens(c)[0] not in stoi]
            status, prediction, probabilities, score = "scored", None, {}, 0
            if len(prompt_tokens) + 1 > model.config.block_size:
                status = "context_too_long"
            elif unknown_prompt or unknown_choices:
                status = "out_of_vocabulary"
            else:
                ids = [stoi["<BOS>"]] + [stoi[t] for t in prompt_tokens]
                # Only the prompt enters the model. No answer list, key, or explanation.
                logits = model(torch.tensor([ids], device=device))[0][0, -1].float().cpu()
                probs = torch.softmax(logits, -1)
                probabilities = {c: probs[stoi[word_tokens(c)[0]]].item() for c in case["choices"]}
                ranked = sorted(probabilities, key=probabilities.get, reverse=True)
                if abs(probabilities[ranked[0]] - probabilities[ranked[1]]) <= 1e-10:
                    status = "tied"
                else:
                    prediction = ranked[0]
                    score = int(prediction == case["answer"])
            sample = generate_reply(model, vocabulary, case["prompt"], seed=seed + index,
                                    temperature=temperature, max_tokens=max_tokens)
            rows.append({"id": case["id"], "group": case["group"], "category": case["category"],
                         "prompt": case["prompt"], "choices": case["choices"],
                         "expected": case["answer"], "predicted_choice": prediction,
                         "score": score, "status": status, "choice_probabilities": probabilities,
                         "unknown_prompt_words": unknown_prompt, "unknown_choices": unknown_choices,
                         "generated_text": sample["response"], "prompt_truncated": sample["prompt_truncated"],
                         "reason": case["reason"], "stage": stage, "sample_seed": seed + index})
    finally:
        model.train(was_training)
    if model_hash(model) != identity:
        raise RuntimeError("Evaluation unexpectedly changed model weights.")

    def summarize(items):
        available = [r for r in items if r["status"] in {"scored", "tied"}]
        correct = sum(r["score"] for r in items)
        return {"correct": correct, "total": len(items), "scorable": len(available),
                "success_rate_all_cases": correct / len(items),
                "accuracy_scorable_cases": correct / len(available) if available else None,
                "coverage": len(available) / len(items)}

    summary = {"suite_id": suite["suite_id"], "suite_sha256": suite_hash(suite),
               "model_sha256": identity, "stage": stage,
               "settings": {"seed": seed, "temperature": temperature, "max_tokens": max_tokens},
               "metric": "highest-probability answer among four single-word choices; ties get zero",
               "generation_note": "Generated text is saved for inspection and is not the multiple-choice score.",
               "missing_note": "Unknown-word and overlong cases get zero in all-case success, not guessed UNK credit.",
               "overall": summarize(rows),
               "by_group": {g: summarize([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})},
               "by_category": {g: summarize([r for r in rows if r["category"] == g]) for g in sorted({r["category"] for r in rows})}}
    for name, value in [("eval_cases.json", suite), ("eval_results.json", rows), ("eval_summary.json", summary)]:
        (output_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "eval_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    print(f"Language evals ({stage}): {summary['overall']['correct']}/{len(rows)}; "
          f"{summary['overall']['scorable']}/{len(rows)} cases have usable vocabulary/context.")
    for group, values in summary["by_group"].items():
        print(f"  {group}: {values['correct']}/{values['total']} correct; {values['scorable']} scorable")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Your run's model.pt")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True, help="A fresh results directory")
    parser.add_argument("--stage", default="final")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory so an earlier result is not overwritten.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, vocabulary, _ = load_model(args.model)
    evaluate_suite(model, vocabulary, load_suite(args.suite), args.output, stage=args.stage)


if __name__ == "__main__":
    main()
