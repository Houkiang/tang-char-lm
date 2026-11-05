from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tang_char_lm.data import prepare_data
from tang_char_lm.evaluation import evaluate_test
from tang_char_lm.model import ModelConfig
from tang_char_lm.training import TrainConfig, train


class TrainingTests(unittest.TestCase):
    def test_checkpoint_and_resume_completed_run(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "poems.txt"
            source.write_text("\n\n".join(f"诗{index}。" for index in range(20)), encoding="utf-8")
            prepared = root / "prepared"
            manifest = prepare_data(source, prepared)
            run_dir = root / "run"
            config = TrainConfig(
                model=ModelConfig(manifest["vocab_size"], "gru", 8, 12, 1, 0.0),
                prepared_dir=str(prepared),
                source_path=str(source),
                run_dir=str(run_dir),
                batch_size=4,
                epochs=2,
                learning_rate=1e-2,
                patience=2,
                device="cpu",
            )
            result = train(config)
            self.assertEqual(result["epochs_completed"], 2)
            for name in ("config.json", "environment.json", "metrics.json", "best.pt", "last.pt", "summary.json"):
                self.assertTrue((run_dir / name).is_file(), name)
            resumed = train(config, resume=True)
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(resumed["best_validation_perplexity"], result["best_validation_perplexity"])
            test_result = evaluate_test(run_dir, "cpu")
            self.assertEqual(test_result["metrics"]["tokens"], manifest["splits"]["test"]["target_tokens_including_eos"])
            self.assertEqual(evaluate_test(run_dir, "cpu"), test_result)


if __name__ == "__main__":
    unittest.main()
