# Run 3 hypothesis, recorded before training

Recorded: 2026-09-23T03:35:49Z (before any Run 3 training started)

Gonzalo's stated question, verbatim from the chat:

> "Let's do that and see whether adding some more information without full context and without accompanying the text with teaching or something like that will actually do anything to the training of this model."

Setup fixed before training: Run 2's corpus (classroom + corpus/opposites.txt + corpus/negation.txt) plus The Adventures of Tom Sawyer, chapters I-III only (Project Gutenberg #74, public domain in the USA). Same 3,000 steps, learning rate 0.001, seed 42, 48-token context, 509-type vocabulary cap.

Pre-training vocabulary simulation (computed from Run 2's training split, not a model result): adding about the first 10% of the book keeps 28 of 48 eval cases scorable (Run 2: 29), pushes ball, missing and pillow out of the vocabulary, and brings in before, from, gave, into, last, right, see, she, that, to, turn, were, who.
