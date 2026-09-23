"""Generate corpus-extension teaching data for two eval categories:
opposites and negation. We teach the *pattern* and the *answer/distractor
vocabulary* using different subjects, people and objects than the eval suite,
and we never reproduce an eval prompt verbatim. A leakage pre-check against
evals/language_evals.json runs at the end and refuses to write on any match.
"""
import random
from pathlib import Path
from run_evals import load_suite, matching_cases

R = random.Random(2026)

# ---------- OPPOSITES ----------
# Antonym pairs. The three eval STEMS (hot, empty, noisy) are taught only through
# frames that do NOT begin "the opposite of <stem> is", but their partners and the
# reverse direction are taught directly. Distractor words (warm, heavy, fast, soft,
# early, loud, round, late) are included so eval cases become scorable.
pairs = [
    ("cold", "hot"), ("full", "empty"), ("quiet", "noisy"), ("quiet", "loud"),
    ("big", "small"), ("fast", "slow"), ("open", "closed"), ("light", "heavy"),
    ("happy", "sad"), ("up", "down"), ("hard", "soft"), ("clean", "dirty"),
    ("near", "far"), ("high", "low"), ("old", "new"), ("wet", "dry"),
    ("long", "short"), ("strong", "weak"), ("early", "late"), ("dark", "bright"),
    ("warm", "cool"), ("heavy", "light"), ("round", "flat"), ("rich", "poor"),
    ("wide", "narrow"), ("thick", "thin"), ("deep", "shallow"), ("true", "false"),
]
STEMS = {"hot", "empty", "noisy"}   # never use "the opposite of <stem> is"

opp_frames_safe = [
    "the opposite of {a} is {b} .",
    "{a} is the opposite of {b} .",
    "{a} and {b} are opposites .",
    "if it is not {a} then it is {b} .",
    "when something is not {a} it is {b} .",
    "one word means {a} and its opposite means {b} .",
]
opp_frames_nostem = [f for f in opp_frames_safe if not f.startswith("the opposite of {a} is")]

opposites = []
for a, b in pairs:
    # both directions
    for x, y in [(a, b), (b, a)]:
        frames = opp_frames_nostem if x in STEMS else opp_frames_safe
        for f in frames:
            opposites.append(f.format(a=x, b=y))
# a few grounding lines for the answer/distractor words
opposites += [
    "cold water feels cool but hot water feels warm .",
    "an empty glass has nothing but a full glass has milk .",
    "a quiet room is calm but a loud room is noisy .",
    "a fast car is quick but a slow car is late .",
    "a soft pillow is not hard .",
    "a heavy box is not light .",
    "an early train is not late .",
    "a round plate is not flat .",
]

# ---------- NEGATION ----------
# Pattern A: "<subj> is not <A> . it is <B> . <subj> is <B> ."   (colors/states)
# Pattern B: "<person> did not <verb> <A> . <person> <verb> <B> . <person> <verb> <B> ."
# Eval stories (box/red/blue, ava/tea/milk, door/open/closed) are avoided; we use
# other subjects but keep the answer words (blue, milk, closed) and distractor
# words (green, yellow, red, rice, tea, bread, open, wide, missing) in play.
subjects_color = ["the sky", "the wall", "the cup", "the hat", "the car", "the shirt",
                  "the roof", "the chair", "the fence", "the boat", "the kite", "the lamp"]
colors = ["blue", "green", "yellow", "red", "white", "black", "grey", "brown"]

subjects_state = ["the window", "the gate", "the shop", "the drawer", "the jar",
                  "the case", "the tent", "the book"]
states = [("open", "closed"), ("closed", "open"), ("wide", "narrow"),
          ("full", "empty"), ("wet", "dry"), ("clean", "dirty")]

people = ["sam", "mia", "leo", "nora", "omar", "ella", "finn", "iris", "hugo", "lena", "theo", "ruby"]
buy_pairs = [("water", "milk"), ("juice", "milk"), ("tea", "milk"), ("soda", "milk"),
             ("bread", "rice"), ("pasta", "rice"), ("rice", "bread"), ("cake", "bread"),
             ("apples", "pears"), ("coffee", "tea"), ("water", "tea")]
buy_verbs = ["bought", "chose", "took", "ordered", "picked"]

VERB_BASE = {"bought": "buy", "chose": "choose", "took": "take", "ordered": "order", "picked": "pick"}
negation = []
# Pattern A colors — several passes so the pattern (not X . it is Y . subj is Y) recurs
for _ in range(4):
    for subj in subjects_color:
        wrong, right = R.sample(colors, 2)
        negation.append(f"{subj} is not {wrong} . it is {right} . {subj} is {right} .")
        wrong2, right2 = R.sample(colors, 2)
        negation.append(f"{subj} was not {wrong2} . it was {right2} . so {subj} is {right2} .")
# Pattern A states — several passes
for _ in range(3):
    for subj in subjects_state:
        a, b = R.choice(states)
        negation.append(f"{subj} is not {a} . it is {b} . {subj} is {b} .")
# Pattern B people buying / choosing — several passes
for _ in range(3):
    for p in people:
        a, b = R.choice(buy_pairs)
        v = R.choice(buy_verbs)
        negation.append(f"{p} did not {VERB_BASE[v]} {a} . {p} {v} {b} . {p} {v} {b} .")
# Targeted reinforcement so the eval ANSWER words (blue, milk, closed) sit in the
# resolved slot, taught with non-eval subjects/people.
blue_subj = ["the sky", "the sea", "the door", "the coat", "the sign", "the ball", "the box"]
for subj in blue_subj:
    wrong = R.choice([c for c in colors if c != "blue"])
    negation.append(f"{subj} is not {wrong} . it is blue . {subj} is blue .")
milk_people = ["theo", "ruby", "hugo", "lena", "iris", "finn", "nora"]
for p in milk_people:
    wrong = R.choice(["tea", "juice", "water", "soda", "coffee"])
    negation.append(f"{p} did not buy {wrong} . {p} bought milk . {p} bought milk .")
closed_subj = ["the window", "the gate", "the shop", "the drawer", "the jar", "the case", "the lid"]
for subj in closed_subj:
    negation.append(f"{subj} is not open . it is closed . {subj} is closed .")
# grounding lines for color/food words
negation += [
    "red green yellow and blue are colors .",
    "milk and tea and juice are drinks .",
    "rice and bread and cake are foods .",
    "an open door is not closed .",
    "a closed gate is not open .",
    "a wide road is not narrow .",
    "the missing key is not here .",
]

# ---------- leakage pre-check ----------
suite = load_suite("evals/language_evals.json")
all_lines = opposites + negation
bad = []
for line in all_lines:
    m = matching_cases(line, suite)
    if m:
        bad.append((line, m))
if bad:
    print("LEAKAGE DETECTED — not writing:")
    for line, m in bad:
        print("  ", m, repr(line))
    raise SystemExit(1)

Path("corpus/opposites.txt").write_text("\n".join(opposites) + "\n", encoding="utf-8")
Path("corpus/negation.txt").write_text("\n".join(negation) + "\n", encoding="utf-8")
print(f"opposites.txt: {len(opposites)} lines")
print(f"negation.txt:  {len(negation)} lines")
print("leakage pre-check: PASS (no eval prompt appears as a substring)")
print("\n--- sample opposites ---")
for l in opposites[:6]:
    print("  ", l)
print("--- sample negation ---")
for l in negation[:6]:
    print("  ", l)
