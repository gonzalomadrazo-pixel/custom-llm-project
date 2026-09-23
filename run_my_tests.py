"""Run the personal hold-out tests (evals/my_holdout.json) on every saved model.

Inference only: no weights change, and the tests never enter any corpus.

Each choice is scored by the log-probability of its whole word sequence after the
prompt (sum over its tokens), so multi-word answers work. Two scoring rules:

  strict   the course rule: any unknown word in the prompt or any choice makes
           the case unscorable, and it counts as 0.
  lenient  unknown prompt words are fed in as <UNK>; choices containing unknown
           words are dropped; the case counts as 0 if the answer itself is
           unknown or fewer than two choices remain.

Negation cases with "controls" also get a flip test and a no-"not" test that
separate reading "not" from just predicting the opposite of the last adjective.

    python run_my_tests.py                # all runs in llm_runs/, both stages
"""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from run_evals import load_model

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "evals" / "my_holdout.json"
OUT = ROOT / "results" / "my_holdout"


def tokenize(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


class Scorer:
    def __init__(self, model_path):
        self.model, self.vocab = load_model(model_path)[:2]
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.bos, self.unk = self.stoi["<BOS>"], self.stoi["<UNK>"]
        self.block = self.model.config.block_size

    def unknown(self, text):
        return [t for t in tokenize(text) if t not in self.stoi]

    def ids(self, text):
        return [self.stoi.get(t, self.unk) for t in tokenize(text)]

    @torch.no_grad()
    def next_probs(self, ids):
        logits, _ = self.model(torch.tensor([ids[-self.block:]]))
        return torch.softmax(logits[0, -1], -1)

    def logprob(self, prompt, continuation):
        ctx, total = [self.bos] + self.ids(prompt), 0.0
        for t in self.ids(continuation):
            total += math.log(max(float(self.next_probs(ctx)[t]), 1e-30))
            ctx.append(t)
        return total


def pick(scores):
    """Highest-scoring choice; a tie for first place picks nothing."""
    best = max(scores.values())
    top = [c for c, s in scores.items() if s == best]
    return top[0] if len(top) == 1 else None


def score_case(sc, case):
    prompt, choices, answer = case["prompt"], case["choices"], case["answer"]
    unk_prompt = sorted(set(sc.unknown(prompt)))
    unk_choice = {c: sc.unknown(c) for c in choices}
    lp = {c: sc.logprob(prompt, c) for c in choices}
    row = {"id": case["id"], "kind": case["kind"], "skill": case.get("skill"), "prompt": prompt,
           "answer": answer, "unknown_prompt_words": unk_prompt,
           "unknown_choice_words": sorted({w for ws in unk_choice.values() for w in ws}),
           "choice_logprobs": {c: round(v, 4) for c, v in lp.items()}}
    # strict (course rule)
    if unk_prompt or any(unk_choice.values()):
        row.update(strict_status="unscorable", strict_pick=None, strict_score=0)
    else:
        p = pick(lp)
        row.update(strict_status="scored" if p else "tie", strict_pick=p, strict_score=int(p == answer))
    # lenient
    known = {c: v for c, v in lp.items() if not unk_choice[c]}
    if unk_choice[answer]:
        row.update(lenient_status="answer_unknown", lenient_pick=None, lenient_score=0)
    elif len(known) < 2:
        row.update(lenient_status="too_few_choices", lenient_pick=None, lenient_score=0)
    else:
        p = pick(known)
        row.update(lenient_status="scored" if p else "tie", lenient_pick=p, lenient_score=int(p == answer),
                   lenient_choices_used=len(known))
    return row


def control_case(sc, case):
    """Does the prediction follow 'not'? Compares the answer word A with its opposite B."""
    ctl = case["controls"]
    a, b = case["answer"], ctl["flip"]["expect"]
    if sc.unknown(a) or sc.unknown(b):
        return {"id": case["id"], "status": "word_unknown", "A": a, "B": b}
    def prefer(prompt):
        p = sc.next_probs([sc.bos] + sc.ids(prompt))
        return {a: round(float(p[sc.stoi[a]]), 4), b: round(float(p[sc.stoi[b]]), 4)}
    orig, flip, no_not = prefer(case["prompt"]), prefer(ctl["flip"]["prompt"]), prefer(ctl["no_not"]["prompt"])
    ok_orig, ok_flip, ok_nonot = orig[a] > orig[b], flip[b] > flip[a], no_not[b] > no_not[a]
    winners = {max(d, key=d.get) for d in (orig, flip, no_not)}
    if ok_orig and ok_flip and ok_nonot:
        verdict = "reads 'not'"
    elif ok_orig and ok_flip:
        verdict = "uses the earlier adjective but ignores 'not' (answers its opposite either way)"
    elif len(winners) == 1:
        verdict = "same answer whatever the story says (ignores the context)"
    else:
        verdict = "inconsistent"
    return {"id": case["id"], "status": "scored", "A": a, "B": b,
            "noun_unknown": sorted(set(sc.unknown(case["prompt"]))),
            "original": {"prompt": case["prompt"], "probs": orig, "prefers_expected": ok_orig},
            "flip": {"prompt": ctl["flip"]["prompt"], "probs": flip, "prefers_expected": ok_flip},
            "no_not": {"prompt": ctl["no_not"]["prompt"], "probs": no_not, "prefers_expected": ok_nonot},
            "verdict": verdict}


def main():
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    runs = sorted(p for p in (ROOT / "llm_runs").iterdir() if p.is_dir() and (p / "model.pt").exists())
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, controls = [], [], []
    for run in runs:
        for stage, fname in (("untrained", "model_untrained.pt"), ("trained", "model.pt")):
            if not (run / fname).exists():
                continue
            sc = Scorer(run / fname)
            rows = [score_case(sc, c) for c in suite["cases"]]
            for r in rows:
                r.update(run=run.name, stage=stage)
            all_rows += rows
            kinds = sorted({r["kind"] for r in rows})
            chance = sum(1 / r["lenient_choices_used"] for r in rows if r.get("lenient_choices_used"))
            summaries.append({
                "run": run.name, "stage": stage, "total": len(rows),
                "strict_correct": sum(r["strict_score"] for r in rows),
                "strict_scorable": sum(r["strict_status"] != "unscorable" for r in rows),
                "lenient_correct": sum(r["lenient_score"] for r in rows),
                "lenient_scorable": sum(r["lenient_status"] in ("scored", "tie") for r in rows),
                "lenient_chance_expected": round(chance, 2),
                "by_kind": {k: {"total": sum(r["kind"] == k for r in rows),
                                "strict": sum(r["strict_score"] for r in rows if r["kind"] == k),
                                "lenient": sum(r["lenient_score"] for r in rows if r["kind"] == k)} for k in kinds}})
            for c in suite["cases"]:
                if "controls" in c:
                    controls.append({"run": run.name, "stage": stage, **control_case(sc, c)})
    meta = {"suite": str(SUITE.relative_to(ROOT)), "suite_sha256": hashlib.sha256(SUITE.read_bytes()).hexdigest(),
            "scoring": __doc__.split("\n\n")[1].strip()}
    (OUT / "results.json").write_text(json.dumps({**meta, "rows": all_rows}, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({**meta, "summaries": summaries}, indent=2), encoding="utf-8")
    (OUT / "negation_controls.json").write_text(json.dumps({**meta, "controls": controls}, indent=2), encoding="utf-8")
    cols = ["run", "stage", "id", "kind", "skill", "prompt", "answer", "strict_status", "strict_pick", "strict_score",
            "lenient_status", "lenient_pick", "lenient_score", "unknown_prompt_words", "unknown_choice_words", "choice_logprobs"]
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    print(f"{'run':26} {'stage':9} strict  lenient (chance)   " + "  ".join(f"{k}(lenient)" for k in summaries[0]["by_kind"]))
    for s in summaries:
        print(f"{s['run']:26} {s['stage']:9} {s['strict_correct']:2}/{s['total']} ({s['strict_scorable']} sc)  "
              f"{s['lenient_correct']:2}/{s['total']} ({s['lenient_scorable']} sc, chance {s['lenient_chance_expected']})  "
              + "  ".join(f"{v['lenient']}/{v['total']:<9}" for v in s["by_kind"].values()))
    print(f"\nwrote {OUT.relative_to(ROOT)}/results.json, results.csv, summary.json, negation_controls.json")


if __name__ == "__main__":
    main()
