import unittest
from pathlib import Path
import tempfile

from ollama_client import cost_usd, result_from_response, run_call
from runner import read_dataset


class FakeBackend:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "ঠিক আছে"}, "prompt_eval_count": 12,
                "eval_count": 4}


class AccountingTests(unittest.TestCase):
    def test_cost_uses_both_ollama_counts(self):
        self.assertAlmostEqual(cost_usd(1_000_000, 1_000_000), 0.75)

    def test_response_preserves_counts_and_calculates_cost(self):
        result = result_from_response(
            {"message": {"content": "answer"}, "prompt_eval_count": 10, "eval_count": 2},
            0.25)
        self.assertEqual((result.input_tokens, result.output_tokens), (10, 2))
        self.assertEqual(result.output, "answer")
        self.assertAlmostEqual(result.cost_usd, 0.0000027)

    def test_runner_uses_fixed_settings_and_fresh_state(self):
        backend = FakeBackend()
        result = run_call(backend, model="fake", system_prompt="system",
                          prompt="প্রশ্ন", max_tokens=32)
        self.assertEqual(result.input_tokens, 12)
        call = backend.calls[0]
        self.assertEqual(call["keep_alive"], 0)
        self.assertEqual(call["options"], {"temperature": 0.0, "seed": 42, "num_predict": 32})
        self.assertEqual(call["messages"][0]["content"], "system")

    def test_dataset_has_fifty_aligned_rows(self):
        rows = read_dataset(Path(__file__).parents[1] / "dataset.tsv")
        self.assertEqual(len(rows), 50)
        self.assertEqual(rows[0]["id"], "1")
        self.assertEqual(rows[-1]["id"], "50")


if __name__ == "__main__":
    unittest.main()
