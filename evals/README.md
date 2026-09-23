# The exam stays outside the textbook

This folder contains **48 fixed, synthetic language evals authored for this assignment**.
They are deliberately small and varied. They are not a published benchmark, a measure
of general intelligence, or a requirement that this tiny model become a capable chatbot.

**Use the supplied tests. You do not need to invent the eval suite.**
Open [language_evals.json](language_evals.json) to read every prompt, answer choice,
correct answer, category, and explanation. Keep the complete suite unchanged across runs.

## What the tests cover

| Group | Cases | What the starter can teach |
|---|---:|---|
| `starter_patterns` | 16 | Familiar sentence patterns and associations across shopping, products, banking, fruit, transport, software, health, and education. Exact test prefixes are reserved before training. |
| `starter_transfer` | 8 | The same words and associations in new sentence structures. These probe transfer beyond memorized templates; success is not guaranteed. |
| `extend_corpus` | 24 | Eight different skills absent from the starter, with many unknown words. Students need different teaching examples and retraining, not just more steps on the original corpus. |

Examples from the suite (answers shown here for explanation, never supplied to the model):

| Skill | Model sees this prefix | Intended next word |
|---|---|---|
| Starter domain association | the report about the surgeon explains the | patient |
| Starter place association | the team discussed the mango and the juice at the | kitchen |
| New wording | yesterday our office discussed the platform and the | update |
| Grammar | yesterday she | walked |
| Opposites | the opposite of empty is | full |
| Negation | ava did not buy tea . she bought milk . ava bought | milk |
| References and roles | ella gave finn a pencil . the person who received the pencil was | finn |
| Sequence | first wash the cup . then dry it . the last action is | dry |
| Spatial relations | the lamp is above the desk . the desk is | below |
| Everyday knowledge | water freezes into | ice |
| Categories and analogies | a puppy grows into a dog . a kitten grows into a | cat |

The full extension group has three different examples per skill. These require
different language patterns, not merely changing the topic of the same sentence.

## Run them

**Colab or Jupyter:** open `custom_llm.ipynb` and Run All. Section 6b runs all 48
tests on the untrained model. Section 8b runs them on the trained model. Setup
downloads the exact helper files and suite outside `corpus/` when they are absent.

The results ZIP includes:

```text
eval_separation.json
model_untrained.pt
model.pt
language_eval_comparison.json
language_evals/
  untrained/
    eval_cases.json
    eval_results.json
    eval_results.csv
    eval_summary.json
  final/
    eval_cases.json
    eval_results.json
    eval_results.csv
    eval_summary.json
```

**Terminal, after training:** install `requirements.txt` and run from this repository:

```sh
python run_evals.py --model llm_runs/YOUR_RUN/model.pt --output results/my-final-evals
python run_evals.py --model llm_runs/YOUR_RUN/model_untrained.pt --stage untrained --output results/my-untrained-evals
```

Replace `YOUR_RUN` with the actual folder name. Use a fresh output folder for each
run. The saved summaries include the suite and model hashes so you can identify
which tests and weights produced the results. CPU is sufficient.

## How scoring works

1. Feed **only the prefix** into the student's trained nanoGPT and read its
   next-token probabilities. No answers or answer choices appear in the prompt.
2. Compare the probabilities of the four candidate words. The highest one is the
   selected answer. Correct earns 1; incorrect or tied earns 0. Random selection
   would average 25% on these four-choice questions.
3. Generate a separate, unconstrained continuation at temperature 0.8, with a fixed
   per-case seed and a 24-token output limit. Save the actual text, even if empty
   or nonsensical. **The multiple-choice score does not grade this free text.**
4. If the prompt or any answer choice contains a word outside the learned vocabulary,
   mark the case `out_of_vocabulary`. Do not compare several identical `<UNK>` tokens
   and award a lucky correct answer. Overlong prompts are also not scored.

Report **both** the success rate over all 48 cases and the accuracy among scorable
cases, plus vocabulary/context coverage and the group/category breakdown. Missing
coverage receives zero in the all-case success rate, so dropping difficult cases
cannot inflate that rate. Scorable accuracy can rise or fall when coverage changes;
read the per-case results before comparing it. Scores across different vocabularies
are useful teaching evidence but do not isolate changes in the model's reasoning.

These are narrow next-word tests. Even perfect multiple-choice results would not
demonstrate free-form question answering, reliable reasoning, or broad understanding.
There is **no minimum pass rate**. Honest failures are useful evidence.

## How evals affect your grade

The assignment uses **deliverable quality 4 + testing & evaluation 3 + working
result 3 = 10 points**. The eval percentage is a model measurement, not your grade.
All four result sets (starter untrained/trained and expanded-corpus untrained/trained),
valid separation from training, and an explanation of scores, coverage, free text,
and failure cases support the **3-point testing & evaluation category**.

Missing runs, omitted cases, a missing extension comparison, or contaminated tests
reduce the evaluation credit supported by your evidence. Partial credit is available.
A complete, valid experiment with a thoughtful explanation can earn full evaluation
credit even when model scores are low or the added corpus does not improve them.
Unknown-word cases receiving zero in the model metric do not trigger the same
percentage deduction from your assignment grade. Code, explanations, and a working
model/chat interface also matter in the other two categories.
See [the full grading guidance](../ASSIGNMENT.md#how-evals-affect-your-assignment-grade).

## Extend the corpus without copying the exam

For the assignment, save a starter-corpus run, then choose **at least two extension
categories**, add focused teaching material in `corpus/`, and train a fresh model.
Run the same 48 tests again and explain changes in scores, coverage, and free text.
More examples can help; a tiny model and short training budget can still fail.

Possible teaching material:

- Grammar: varied sentences with singular/plural subjects and present/past actions.
- Opposites: contextual contrasts involving temperatures, amounts, sounds, sizes,
  and textures. Teach the vocabulary in sentences rather than pasting answer lists.
- Negation: varied corrections that explicitly distinguish what happened from what
  did not, using different people, objects, and wording from the tests.
- References: short stories with different names, givers, recipients, callers,
  and responders. Include vocabulary naturally; do not reproduce the test stories.
- Sequence and space: new event orders, instructions, locations, and inverse relations.
- Knowledge and categories: simple factual descriptions and category relationships,
  written differently from the eval prompts. The underlying facts may overlap.

Aim for varied practice rather than one copied answer per test. Check the printed
unknown-word reports: the tokenizer keeps only the 509 most frequent training token
types. The vocabulary must still come **only from training text**. Never insert eval
words directly into the vocabulary just to make the tests scorable.

Keep these paths separate:

```text
corpus/                       teaching sources only
evals/language_evals.json      fixed exam and answer key
run_evals.py                  inference and scoring, no training
llm_runs/                     saved weights, eval outputs, and chat evidence
```

The starter reserves any generated classroom sentence containing a test prefix
before the train/validation split and vocabulary building. It rejects exact test
prefixes in imported training files and rejects a corpus folder that includes
`evals/` or the repository root. `eval_separation.json` records the reservation.
Checks normalize case, punctuation spacing, and whitespace, but **do not detect all
paraphrases, leaked answer lists, or semantic contamination**. Inspect your sources.
Do not copy this README, the test JSON, eval outputs, or chat transcripts into training.

These tests are public and students use them to guide improvements. Describe the
comparison as a **fixed development benchmark**, not an untouched final test. For
a claim about unseen generalization, reserve additional new examples that have not
guided any corpus or model choices.

## Chat with the same model

Section 10 of the notebook lets you edit a prompt, run the cell, and receive a new
response from the trained model. Each message starts fresh. It saves a transcript
and refreshes the results ZIP after every interaction.

The terminal version is:

```sh
python chat.py --model llm_runs/YOUR_RUN/model.pt --transcript results/my-chat.json
```

Type `/quit` to finish. Unknown words and context truncation are shown. The script
records the model hash and actual prompts/replies. Neither interface trains on the
conversation. A custom HTML/web interface is also acceptable if it actually uses
your trained model. Include at least three real interactions and a screenshot or
short recording; a frontend showing canned text is not a working model interface.

## Maintainer checks

```sh
python -m unittest test_language_evals test_corpus
```

These check held-out prompt separation, supported starter vocabulary, unknown-answer
handling, answer-key isolation, and unchanged weights/random state during inference.
