# CLAUDE.md — working notes for this repository

Handoff for continuing this project in Claude Code. Read this before changing anything.

## What this is

Class 4 assignment for *From Zero to AI Agents* (Fall 26), submitted by Gonzalo. A tiny nanoGPT (2 blocks, 4 heads, 64-number embeddings, 48-token context, word tokens) is trained twice on a synthetic corpus, evaluated against a fixed 48-case suite before and after training, and connected to a chat interface. The full brief is in `ASSIGNMENT.md`; `README.md` is the graded entry point.

The model is a next-word predictor with a 283-word vocabulary. It is **not** a question-answering system and cannot become one at this scale. Continuation-style output and unknown-word failures are expected results to document, not bugs to fix.

## Ground rules

1. **Never invent results.** Every number in `README.md` was generated from files in `llm_runs/` or `results/`. If a claim changes, regenerate it from artifacts rather than editing prose.
2. **Never put eval material in the corpus.** `evals/language_evals.json`, its answers and any eval output must stay out of `corpus/`, out of vocabulary building and out of training. The course penalty extends beyond the 3-point testing category. `make_extension_corpus.py` runs the course's own `matching_cases` check and refuses to write on a match; keep that check in place.
3. **The ✍️ sections in `README.md` belong to Gonzalo.** §2 (prediction) and §8 (six "what I learned" answers) must be written by him, not drafted for him. The assignment's starter prompt instructs the AI assistant to *ask* him to explain these in his own words. Helping him tighten prose he wrote is fine; supplying the understanding is not.
4. **Keep the eval suite unchanged.** Its SHA-256 is `e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7`. Verify with `sha256sum evals/language_evals.json`.
5. **Disclose, don't hide, failures.** The negation result is a documented failure (§6.4). Keep it.

## Current state

Both required experiments are complete, deterministic and reproducible.

| Experiment | Run folder | Corpus | All-case eval, untrained → trained |
|---|---|---|---|
| Starter | `llm_runs/20260922T060103_576381Z` | classroom only | 9/48 → 20/48 |
| Expanded | `llm_runs/20260922T060146_232475Z` | classroom + `corpus/negation.txt` + `corpus/opposites.txt` | 5/48 → 27/48 |

Both at 3,000 steps, learning rate 0.001, seed 42. Executed notebooks are `custom_llm_starter.executed.ipynb` and `custom_llm_expanded.executed.ipynb` (outputs intact, zero errors; do not clear outputs).

**Key finding.** Opposites improved for real (control probes: the true antonym ranks first for 21 of 23 stems, including stems never trained in that sentence frame). Negation did **not**: the model answers *blue* or *closed* regardless of the story, because the notebook split each three-sentence teaching example into separate passages, and because answer words were over-represented. See README §6.4 and `results/negation_control_probes.json`.

## Outstanding tasks

1. **Gonzalo writes the three ✍️ sections** in `README.md` (§2 prediction, §8 six answers, confirming the §9 prediction is his). `STUDY_NOTES.md` — kept outside this repo, in his downloads — holds the concepts and his numbers as raw material.
2. **Paste his prediction into the "My prediction" markdown cell** of both executed notebooks. Edit markdown only; do not re-execute, or the outputs are lost.
3. **Optional:** he runs `chat.py` locally and takes a live screenshot to accompany `results/chat_expanded.png` (currently a faithful render of a real captured session, labelled as such).
4. **Review the AI-assistance paragraph** in README §12 against course policy.
5. **Push to a new public GitHub repo and submit the URL on bCourses.** Nothing has been pushed yet; there is no git history in this folder.

## Dashboard

`python build_dashboard.py` bundles every complete folder in `llm_runs/` (weights included) into one self-contained `dashboard.html`. Page source lives in `dashboard_src/` (template.html, style.css, app.js). The JS forward pass returns attention and MLP activations and was checked against PyTorch: next-token probabilities match to about 1e-5. The builder also re-scans all 48 eval prompts against `corpus/` text files and each run's training split, and derives checklist status from the files on disk. Rerun it after every new training run. `RUN_LABELS` in the builder gives runs friendly names. The dashboard is an output: never put it in `corpus/`.

## Commands

```bash
# environment
pip install -r requirements.txt

# rerun evals from saved weights (output folder must not already exist)
python run_evals.py --model llm_runs/20260922T060146_232475Z/model.pt --output results/rerun-check
python run_evals.py --model llm_runs/20260922T060146_232475Z/model_untrained.pt --stage untrained --output results/rerun-check-untrained

# terminal chat against the trained model
python chat.py --model llm_runs/20260922T060146_232475Z/model.pt --transcript results/my-chat.json

# regenerate the extension corpus (deterministic, includes the leakage pre-check)
python make_extension_corpus.py

# integrity checks
python -m unittest test_language_evals test_corpus
sha256sum evals/language_evals.json
```

To reproduce a full experiment, open `custom_llm.ipynb` and Run All. For the starter run, `corpus/` must contain only `README.md`. For the expanded run, both TXT files must be present. Each Run All writes a new timestamped folder to `llm_runs/`.

## Environment gotchas

- On Linux, the PyPI `torch` wheel preloads CUDA libraries at import even for CPU-only work. If `import torch` fails with a missing `libcudnn`/`libcublas`, either install the matching `nvidia-*-cu12` packages or use the CPU wheel from `download.pytorch.org`. On macOS this is not an issue.
- Executing the notebooks headlessly needs `nbconvert` and `ipykernel`:
  `jupyter nbconvert --to notebook --execute custom_llm.ipynb --output custom_llm_expanded.executed.ipynb --ExecutePreprocessor.timeout=900`
- Training takes roughly 30 seconds per run on one CPU core. No GPU is needed.
- `chat.py` refuses to overwrite an existing transcript; pass a new `--transcript` path each session.

## Layout notes

- `corpus/` holds only teaching text. Never point `CORPUS_FOLDER` at the project root or at `evals/`.
- `results/` holds eval reruns, the chat transcript and image, next-token diagnostics and the control probes.
- `chat-with-your-llm.html` is an optional browser interface: the forward pass reimplemented in JavaScript (`nanogpt_engine.js`) over the exported weights, verified to match PyTorch's greedy output exactly. `model-viewer.html` is an optional visual viewer (loss curves, embedding map with neighbor lookup, samples, scores). Neither is required by the assignment; the terminal interface and the run artifacts are the primary evidence.
- `.gitignore` deliberately does **not** exclude `llm_runs/` or `corpus/`; the grader needs to see both.

## Push

Create an empty public repo on GitHub first (no README), then:

```bash
git init -b main
git add .
git commit -m "Class 4: custom nanoGPT, two corpus experiments, 48 evals, chat"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Afterwards, open the repository signed out and confirm the notebooks render with their outputs, `training_curves.svg` displays in README §5.1, and the evidence links resolve.
