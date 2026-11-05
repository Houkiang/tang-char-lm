import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tang_char_lm.data import prepare_data


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchDataTests(unittest.TestCase):
    def test_batches_count_every_target_once(self) -> None:
        import torch

        from tang_char_lm.torch_data import collate_poems, make_loader

        inputs, targets, lengths = collate_poems([[1, 4, 2], [1, 5, 6, 2]])
        self.assertEqual(inputs.tolist(), [[1, 4, 0], [1, 5, 6]])
        self.assertEqual(targets.tolist(), [[4, 2, 0], [5, 6, 2]])
        self.assertEqual(lengths.tolist(), [2, 3])

        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "poems.txt"
            source.write_text("\n\n".join(f"诗{index}。" for index in range(20)), encoding="utf-8")
            output = root / "prepared"
            manifest = prepare_data(source, output)
            for split in ("train", "validation", "test"):
                loader = make_loader(output, split, batch_size=3)
                count = sum(int(torch.count_nonzero(targets).item()) for _, targets, _ in loader)
                self.assertEqual(count, manifest["splits"][split]["target_tokens_including_eos"])


if __name__ == "__main__":
    unittest.main()
