import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tang_char_lm.data import SPECIAL_TOKENS, encode_poem, load_prepared, prepare_data


class PreparedDataTests(unittest.TestCase):
    def test_reproducible_split_and_train_only_vocab(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "poems.txt"
            poems = ["诗" + chr(0x4E00 + index) + "。" for index in range(30)]
            source.write_text("\n\n".join(poems + [poems[0]]), encoding="utf-8")
            output = root / "prepared"
            manifest = prepare_data(source, output, seed=42)
            self.assertEqual(prepare_data(source, output, seed=42), manifest)
            second = prepare_data(source, root / "prepared-again", seed=42)
            self.assertEqual(second["splits_sha256"], manifest["splits_sha256"])
            self.assertEqual(second["vocab_sha256"], manifest["vocab_sha256"])
            self.assertEqual(manifest["poems_raw"], 31)
            self.assertEqual(manifest["exact_duplicate_poems_removed"], 1)
            self.assertEqual([manifest["splits"][name]["poems"] for name in ("train", "validation", "test")], [24, 3, 3])

            artifact = load_prepared(output, source)
            hashes = [{row["sha256"] for row in artifact["splits"][name]} for name in ("train", "validation", "test")]
            self.assertFalse(hashes[0] & hashes[1] | hashes[0] & hashes[2] | hashes[1] & hashes[2])
            tokens = artifact["vocab"]["tokens"]
            self.assertEqual(tokens[:4], list(SPECIAL_TOKENS))
            self.assertGreater(manifest["splits"]["validation"]["oov_rate"], 0)
            self.assertGreater(manifest["splits"]["test"]["oov_rate"], 0)
            self.assertEqual(encode_poem("未知", tokens)[0], 1)
            self.assertEqual(encode_poem("未知", tokens)[-1], 2)

            with self.assertRaises(FileExistsError):
                prepare_data(source, output, seed=43)

            split_file = output / "splits.json"
            split_file.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(output, source)


if __name__ == "__main__":
    unittest.main()
