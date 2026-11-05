import math
import unittest

import torch

from tang_char_lm.model import CharLanguageModel, ModelConfig
from tang_char_lm.training import _metrics, run_epoch
from tang_char_lm.torch_data import collate_poems


class ModelTests(unittest.TestCase):
    def test_lstm_and_gru_shapes_and_packed_count(self) -> None:
        inputs, _, lengths = collate_poems([[1, 4, 2], [1, 5, 6, 2]])
        for rnn_type in ("lstm", "gru"):
            model = CharLanguageModel(
                ModelConfig(10, rnn_type, embedding_dim=4, hidden_dim=6, num_layers=2, dropout=0.1)
            )
            logits, _ = model(inputs)
            self.assertEqual(tuple(logits.shape), (2, 3, 10))
            packed_logits = model.forward_packed(inputs, lengths)
            self.assertEqual(tuple(packed_logits.shape), (5, 10))
            self.assertGreater(model.parameter_count, 0)

    def test_perplexity_uses_global_token_average(self) -> None:
        result = _metrics(6.0, 3)
        self.assertEqual(result["cross_entropy"], 2.0)
        self.assertAlmostEqual(result["perplexity"], math.exp(2.0))

    def test_epoch_counts_only_non_padding_targets(self) -> None:
        model = CharLanguageModel(ModelConfig(10, "gru", 4, 6, 1, 0.0))
        batch = collate_poems([[1, 4, 2], [1, 5, 6, 2]])
        result = run_epoch(model, [batch], torch.device("cpu"))
        self.assertEqual(result["tokens"], 5)


if __name__ == "__main__":
    unittest.main()
