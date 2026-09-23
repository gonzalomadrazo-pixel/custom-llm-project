"""Build dashboard.html: one self-contained page over every run in llm_runs/.

Everything shown in the dashboard is read from saved artifacts (run folders,
evals/, corpus/, README.md, ASSIGNMENT.md). Nothing is invented or retyped.
The dashboard is an output, never a training input: it lives at the project
root, outside corpus/, and the notebook refuses to use the project root as a
corpus folder.

    python build_dashboard.py            # writes dashboard.html
    python build_dashboard.py --fragment out.html   # body-only copy for hosting

Rerun it after every new training run; new llm_runs/ folders are picked up
automatically.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "llm_runs"
SRC = ROOT / "dashboard_src"

# SHA-256 of evals/language_evals.json recorded when the suite was first copied in.
SUITE_FILE_SHA256 = "e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7"

# Optional friendly names; runs not listed are labelled from their corpus files.
RUN_LABELS = {
    "20260922T060103_576381Z": "Starter corpus",
    "20260922T060146_232475Z": "Expanded: opposites + negation",
    "20260923T033633_820623Z": "Run 3: expanded + Tom Sawyer ch. I-III",
}
# Run the dashboard opens on (the graded expanded-corpus experiment).
DEFAULT_RUN = "20260922T060146_232475Z"

REQUIRED_RUN_FILES = ["config.json", "history.json", "model.pt", "model_untrained.pt",
                      "inspection.json", "split.json", "corpus_manifest.json"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def tokenize(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def pack_weights(model_path):
    """All parameters as one float32 blob plus an index of (name, offset, shape)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    index, chunks, offset = [], [], 0
    for name, tensor in state.items():
        if name == "lm_head.weight":
            continue  # tied to transformer.wte.weight
        arr = tensor.detach().float().numpy().astype("<f4").ravel()
        index.append({"name": name, "offset": offset, "shape": list(tensor.shape)})
        chunks.append(arr)
        offset += arr.size
    blob = np.concatenate(chunks).tobytes()
    return {
        "index": index,
        "b64": base64.b64encode(blob).decode("ascii"),
        "count": offset,
        "sha256": sha256(model_path),
        "completed_steps": ckpt.get("completed_steps"),
        "args": ckpt.get("model_args"),
    }


def compact_results(rows):
    keep = ["id", "predicted_choice", "score", "status", "choice_probabilities",
            "unknown_prompt_words", "unknown_choices", "generated_text", "prompt_truncated"]
    return [{k: r.get(k) for k in keep} for r in rows]


def read_samples(run):
    out = {}
    for f in sorted((run / "samples").glob("step_*.txt")):
        step = int(f.stem.split("_")[1])
        out[step] = f.read_text(encoding="utf-8").splitlines()
    return out


def label_for(run_id, manifest):
    if run_id in RUN_LABELS:
        return RUN_LABELS[run_id]
    files = [f["file"] for f in manifest.get("files", [])]
    if not files:
        return "Starter corpus"
    names = ", ".join(Path(f).stem for f in files[:3]) + ("…" if len(files) > 3 else "")
    return ("Folder only: " if manifest.get("mode") == "folder" else "Expanded: ") + names


def collect_run(run):
    cfg = load(run / "config.json")
    manifest = load(run / "corpus_manifest.json")
    split = load(run / "split.json")
    ckpt = load(run / "checkpoint.json") if (run / "checkpoint.json").exists() else {}
    evals = {}
    for stage in ("untrained", "final"):
        d = run / "language_evals" / stage
        if (d / "eval_results.json").exists():
            evals[stage] = {
                "summary": load(d / "eval_summary.json"),
                "results": compact_results(load(d / "eval_results.json")),
                "files": {k: rel(d / f) for k, f in [("csv", "eval_results.csv"),
                          ("json", "eval_results.json"), ("summary", "eval_summary.json")]},
            }
    chat = load(run / "chat_transcript.json") if (run / "chat_transcript.json").exists() else None
    optional = lambda f: load(run / f) if (run / f).exists() else None
    files = sorted(p for p in run.rglob("*") if p.is_file())
    return {
        "id": run.name,
        "label": label_for(run.name, manifest),
        "kind": "starter" if not manifest.get("files") else "expanded",
        "config": cfg,
        "summary": optional("training_summary.json"),
        "history": load(run / "history.json"),
        "manifest": manifest,
        "vocab_report": optional("vocabulary_report.json"),
        "eval_separation": optional("eval_separation.json"),
        "temperature": optional("temperature_comparison.json"),
        "inspection": load(run / "inspection.json"),
        "tokenization": optional("tokenization.json"),
        "samples": read_samples(run),
        "vocab": ckpt.get("vocabulary") or load(run / "tokenization.json")["vocabulary"],
        "token_counts": ckpt.get("token_counts"),
        "split": split,
        "evals": evals,
        "chat": chat,
        "weights": {"trained": pack_weights(run / "model.pt"),
                    "untrained": pack_weights(run / "model_untrained.pt")},
        "files": [{"path": rel(p), "bytes": p.stat().st_size} for p in files],
        "zip": rel(run.with_suffix(".zip")) if run.with_suffix(".zip").exists() else None,
    }


def split_markdown(path, level=r"#{1,3}"):
    """Split a markdown file into (heading, body) sections for search."""
    text = Path(path).read_text(encoding="utf-8")
    parts, current, body = [], None, []
    for line in text.splitlines():
        m = re.match(rf"^({level})\s+(.*)", line)
        if m:
            if current is not None or body:
                parts.append({"title": current or Path(path).name, "text": "\n".join(body).strip()})
            current, body = m.group(2).strip(), []
        else:
            body.append(line)
    parts.append({"title": current or Path(path).name, "text": "\n".join(body).strip()})
    return [p for p in parts if p["text"] or p["title"]]


def notebook_status(path):
    if not path.exists():
        return None
    nb = load(path)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ran = sum(1 for c in code if c.get("execution_count"))
    with_outputs = sum(1 for c in code if c.get("outputs"))
    errors = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
    return {"path": rel(path), "code_cells": len(code), "ran": ran, "with_outputs": with_outputs, "errors": errors}


def leakage_scan(suite, runs):
    """Normalized contiguous match of every eval prompt against every training input."""
    prompts = [(c["id"], " ".join(tokenize(c["prompt"]))) for c in suite["cases"]]
    sources = []
    for f in sorted((ROOT / "corpus").rglob("*")):
        if f.is_file() and f.suffix.lower() in {".txt", ".md"} and f.name != "README.md":
            sources.append((rel(f), f.read_text(encoding="utf-8", errors="ignore")))
    for r in runs:
        sources.append((f"llm_runs/{r['id']} training split", "\n".join(r["split"]["train"])))
    hits = []
    for name, text in sources:
        norm = " " + " ".join(tokenize(text)) + " "
        for cid, p in prompts:
            if " " + p + " " in norm:
                hits.append({"case": cid, "source": name})
    return {"sources_checked": [s for s, _ in sources], "prompts": len(prompts), "hits": hits}


def build_checks(suite, runs):
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    notebooks = {n: notebook_status(ROOT / n) for n in
                 ["custom_llm_starter.executed.ipynb", "custom_llm_expanded.executed.ipynb"]}
    chat_results = sorted((ROOT / "results").glob("chat*.json"))
    chat_turns = max([len(load(p).get("turns", [])) for p in chat_results] or [0])
    screenshots = [rel(p) for p in (ROOT / "results").glob("*.png")]
    ext_files = [f.name for f in (ROOT / "corpus").rglob("*")
                 if f.is_file() and f.name != "README.md" and not f.name.startswith(".")]
    return {
        "suite_sha256": sha256(ROOT / "evals/language_evals.json"),
        "suite_sha256_expected": SUITE_FILE_SHA256,
        "notebooks": notebooks,
        "chat_turns": chat_turns,
        "chat_files": [rel(p) for p in chat_results],
        "screenshots": screenshots,
        "readme_placeholders": len(re.findall(r"✍️", readme)),
        "readme_has_four_rows": "Four-row comparison" in readme,
        "git": (ROOT / ".git").exists() or (ROOT.parent / ".git").exists(),
        "extension_files": sorted(ext_files),
        "leakage": leakage_scan(suite, runs),
        "complete_eval_sets": sum(len(r["evals"].get(s, {}).get("results", [])) == 48
                                  for r in runs for s in ("untrained", "final")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--fragment", help="also write a body-only copy (no doctype/head) here")
    ap.add_argument("--repo-url", help="GitHub file base, e.g. https://github.com/USER/REPO/blob/BRANCH; evidence links then open on GitHub")
    args = ap.parse_args()

    run_dirs = sorted(p for p in RUNS.iterdir()
                      if p.is_dir() and all((p / f).exists() for f in REQUIRED_RUN_FILES))
    runs = [collect_run(p) for p in run_dirs]
    suite = load(ROOT / "evals/language_evals.json")
    probes = {k: load(ROOT / f"results/{k}_control_probes.json")
              for k in ("opposites", "negation") if (ROOT / f"results/{k}_control_probes.json").exists()}
    data = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_url": args.repo_url,
        "default_run": DEFAULT_RUN,
        "runs": runs,
        "suite": {"id": suite.get("suite_id"), "groups": suite.get("groups"),
                  "scoring": suite.get("scoring"), "cases": suite["cases"]},
        "probes": probes,
        "assignment": split_markdown(ROOT / "ASSIGNMENT.md", r"#{1,2}"),
        "readme": split_markdown(ROOT / "README.md"),
        "checks": build_checks(suite, runs),
        "chat_record": load(ROOT / "results/chat_expanded.json")
        if (ROOT / "results/chat_expanded.json").exists() else None,
    }
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    body = (SRC / "template.html").read_text(encoding="utf-8")
    body = body.replace("/*STYLE*/", (SRC / "style.css").read_text(encoding="utf-8"))
    body = body.replace("/*APP*/", (SRC / "app.js").read_text(encoding="utf-8"))
    body = body.replace("/*DATA*/", payload)
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + "\n</body>\n</html>\n")
    Path(ROOT / args.out).write_text(full, encoding="utf-8")
    if args.fragment:
        Path(args.fragment).write_text(body, encoding="utf-8")
    leaks = data["checks"]["leakage"]["hits"]
    print(f"wrote {args.out}: {len(runs)} runs, {len(full) / 1e6:.2f} MB, "
          f"leakage hits: {len(leaks)}, complete eval sets: {data['checks']['complete_eval_sets']}")


if __name__ == "__main__":
    main()
