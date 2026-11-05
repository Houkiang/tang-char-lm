from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from tang_char_lm.data import prepare_data
from tang_char_lm.evaluation import _summary_statistics, evaluate_baselines
from tang_char_lm.generation import analyze_text, generate_text
from tang_char_lm.model import CharLanguageModel, ModelConfig


class EvaluationGenerationTests(unittest.TestCase):
    def test_summary_statistics_uses_sample_standard_deviation(self) -> None:
        result = _summary_statistics([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_standard_deviation"], 1.0)

    def test_statistical_baselines_select_on_validation_and_count_test_tokens(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "poems.txt"
            source.write_text("\n\n".join(f"春山{index}。" for index in range(30)), encoding="utf-8")
            prepared = root / "prepared"
            manifest = prepare_data(source, prepared)
            output = root / "baselines.json"
            result = evaluate_baselines(prepared, source, output, alphas=(0.01, 0.1))
            self.assertTrue(output.is_file())
            for name in ("unigram", "bigram"):
                self.assertIn(result["models"][name]["selected_alpha"], (0.01, 0.1))
                self.assertEqual(
                    result["models"][name]["test"]["tokens"],
                    manifest["splits"]["test"]["target_tokens_including_eos"],
                )
            self.assertEqual(evaluate_baselines(prepared, source, output), result)

    def test_five_character_constraint_has_exact_surface_form(self) -> None:
        tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "春", "山", "月", "水", "，", "。"]
        model = CharLanguageModel(ModelConfig(len(tokens), "gru", 8, 12, 1, 0.0))
        result = generate_text(
            model,
            tokens,
            "春",
            device=torch.device("cpu"),
            method="top_p",
            seed=42,
            max_new_tokens=40,
            line_length=5,
        )
        analysis = analyze_text(result["text"], set())
        self.assertTrue(result["stopped_on_eos"])
        self.assertTrue(analysis["four_line_five_char_complete"])
        self.assertEqual(analysis["line_lengths"], [5, 5, 5, 5])


if __name__ == "__main__":
    unittest.main()
