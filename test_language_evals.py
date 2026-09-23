"""Verify leakage protection, meaningful scoring, and inference-only execution."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from nanogpt_model import GPT, GPTConfig
from run_evals import (evaluate_suite, generate_reply, load_suite, matching_cases,
                       model_hash, reject_eval_leakage, reserve_classroom_passages,
                       validate_corpus_location, word_tokens)

ROOT = Path(__file__).resolve().parent


class FixedModel(torch.nn.Module):
    def __init__(self, size, preferred):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(size))
        with torch.no_grad():
            self.logits[preferred] = 5
        self.config = SimpleNamespace(block_size=48)
        self.inputs = []

    def forward(self, ids):
        self.inputs.append(ids.detach().clone())
        return self.logits[None, None, :], None


class LanguageEvalTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_suite()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_suite_is_diverse_and_starter_cases_have_supported_vocabulary(self):
        self.assertEqual(len(self.suite["cases"]), 48)
        self.assertEqual(len({c["category"] for c in self.suite["cases"] if c["group"] == "extend_corpus"}), 8)
        tree = ast.parse((ROOT / "custom_llm.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "classroom_corpus")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "corpus-generator", "exec"), namespace)
        original = namespace["classroom_corpus"]().splitlines()
        retained, audit = reserve_classroom_passages(original, self.suite)
        self.assertGreater(audit["excluded_passages"], 0)
        self.assertGreater(len(retained), 100)
        self.assertFalse(any(matching_cases(p, self.suite) for p in retained))
        vocabulary = set(word_tokens(" ".join(retained)))
        for c in self.suite["cases"]:
            words = set(word_tokens(c["prompt"] + " " + " ".join(c["choices"])))
            if c["group"].startswith("starter"):
                self.assertFalse(words - vocabulary, (c["id"], words - vocabulary))
            else:
                self.assertTrue(words - vocabulary, c["id"])

    def test_leakage_checks_normalization_multisentence_and_folder_scope(self):
        case = next(c for c in self.suite["cases"] if c["category"] == "negation")
        contaminated = "\n  ".join(case["prompt"].upper().split()) + " blue ."
        with self.assertRaisesRegex(ValueError, "Eval leakage"):
            reject_eval_leakage(contaminated, self.suite, "practice.pdf")
        reject_eval_leakage("A yellow bowl sits on a wooden table.", self.suite, "practice.txt")
        for bad in [ROOT, ROOT / "evals", ROOT / "evals/nested", ROOT.parent]:
            with self.assertRaisesRegex(ValueError, "separate"):
                validate_corpus_location(bad)
        validate_corpus_location(Path(self.temp.name) / "training")

    def test_unknown_answers_do_not_receive_accidental_unk_credit(self):
        vocab = ["<UNK>", "<BOS>", "<EOS>", "one", "bird", "is"]
        case = next(c for c in self.suite["cases"] if c["category"] == "grammar")
        suite = {"suite_id": "test", "cases": [case]}
        summary = evaluate_suite(FixedModel(len(vocab), 0), vocab, suite, self.temp.name, max_tokens=1)
        result = json.loads((Path(self.temp.name) / "eval_results.json").read_text())[0]
        self.assertEqual(result["status"], "out_of_vocabulary")
        self.assertIsNone(result["predicted_choice"])
        self.assertEqual(summary["overall"]["correct"], 0)
        self.assertIsNone(summary["overall"]["accuracy_scorable_cases"])

    def test_gold_answer_and_choices_are_not_in_model_prompt(self):
        case = next(c for c in self.suite["cases"] if c["category"] == "grammar")
        vocab = ["<UNK>", "<BOS>", "<EOS>"] + sorted(set(word_tokens(case["prompt"]) + case["choices"]))
        model = FixedModel(len(vocab), vocab.index(case["answer"]))
        suite = {"suite_id": "test", "cases": [case]}
        summary = evaluate_suite(model, vocab, suite, self.temp.name, max_tokens=1)
        self.assertEqual(summary["overall"]["correct"], 1)
        expected = [vocab.index("<BOS>")] + [vocab.index(w) for w in word_tokens(case["prompt"])]
        self.assertTrue(all(t.tolist() == [expected] for t in model.inputs))
        # Relabeling the answer changes the score, never the model input/prediction.
        changed = copy.deepcopy(suite)
        changed["cases"][0]["answer"] = next(c for c in case["choices"] if c != case["answer"])
        result = evaluate_suite(model, vocab, changed, self.temp.name, max_tokens=1)
        self.assertEqual(result["overall"]["correct"], 0)

    def test_real_model_weights_mode_rng_and_repeatability(self):
        vocab = ["<UNK>", "<BOS>", "<EOS>"] + sorted({t for c in self.suite["cases"]
            for t in word_tokens(c["prompt"] + " " + " ".join(c["choices"]))})
        model = GPT(GPTConfig(vocab_size=len(vocab), block_size=48, n_layer=1, n_head=2, n_embd=16, dropout=0.1))
        model.train()
        before, rng = model_hash(model), torch.get_rng_state().clone()
        a = generate_reply(model, vocab, "the customer", seed=10, max_tokens=3)
        b = generate_reply(model, vocab, "the customer", seed=10, max_tokens=3)
        self.assertEqual(a, b)
        evaluate_suite(model, vocab, {**self.suite, "cases":self.suite["cases"][:2]}, self.temp.name, max_tokens=2)
        self.assertEqual(before, model_hash(model))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
