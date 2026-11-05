"""Deterministic poem-level splits and train-only character vocabulary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
SPLIT_NAMES = ("train", "validation", "test")
SOURCE_URL = "https://github.com/nndl/llm-beginner/blob/master/poetryFromTang.txt"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_poems(path: Path) -> list[str]:
    """Read blank-line-separated poems; preserve poem-internal line breaks and punctuation."""
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return [block.strip() for block in re.split(r"\n(?:[ \t]*\n)+", text) if block.strip()]


def poem_hash(poem: str) -> str:
    return sha256_bytes(poem.encode("utf-8"))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _split_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("At least three unique poems are required for train/validation/test")
    train_count = int(total * 0.8)
    validation_count = int(total * 0.1)
    test_count = total - train_count - validation_count
    if min(train_count, validation_count, test_count) == 0:
        raise ValueError("Corpus is too small for an 80/10/10 poem-level split")
    return train_count, validation_count, test_count


def prepare_data(source: Path, output: Path, seed: int = 42) -> dict[str, object]:
    """Create immutable split/vocabulary artifacts and an audit manifest."""
    source = source.resolve()
    output = output.resolve()
    raw_bytes = source.read_bytes()
    poems = read_poems(source)
    if not poems:
        raise ValueError("Corpus has no poems")

    unique: dict[str, dict[str, object]] = {}
    for index, poem in enumerate(poems):
        digest = poem_hash(poem)
        unique.setdefault(digest, {"source_index": index, "sha256": digest})
    records = list(unique.values())
    train_count, validation_count, _ = _split_counts(len(records))
    random.Random(seed).shuffle(records)
    splits = {
        "train": records[:train_count],
        "validation": records[train_count : train_count + validation_count],
        "test": records[train_count + validation_count :],
    }
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda record: record["source_index"])

    train_text = "".join(poems[record["source_index"]] for record in splits["train"])
    vocab = {"tokens": list(SPECIAL_TOKENS) + sorted(set(train_text))}
    known_chars = set(vocab["tokens"][len(SPECIAL_TOKENS) :])
    split_summary: dict[str, dict[str, object]] = {}
    for name in SPLIT_NAMES:
        subset = [poems[record["source_index"]] for record in splits[name]]
        char_count = sum(len(poem) for poem in subset)
        unk_count = sum(char not in known_chars for poem in subset for char in poem)
        split_summary[name] = {
            "poems": len(subset),
            "characters_including_line_breaks": char_count,
            "target_tokens_including_eos": char_count + len(subset),
            "unknown_characters": unk_count,
            "oov_rate": unk_count / char_count if char_count else 0.0,
        }

    split_bytes = _json_bytes(splits)
    vocab_bytes = _json_bytes(vocab)
    manifest: dict[str, object] = {
        "format_version": 1,
        "source_url": SOURCE_URL,
        "source_path": os.path.relpath(source, output),
        "source_sha256": sha256_bytes(raw_bytes),
        "source_bytes": len(raw_bytes),
        "seed": seed,
        "split_rule": "shuffle unique poems with random.Random(seed); floor(80%), floor(10%), remainder",
        "poems_raw": len(poems),
        "poems_unique": len(records),
        "exact_duplicate_poems_removed": len(poems) - len(records),
        "vocab_size": len(vocab["tokens"]),
        "special_token_ids": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "splits_sha256": sha256_bytes(split_bytes),
        "vocab_sha256": sha256_bytes(vocab_bytes),
        "splits": split_summary,
    }

    if output.exists():
        existing = load_prepared(output, source)
        if existing["manifest"] == manifest and existing["splits"] == splits:
            return manifest
        raise FileExistsError(f"Prepared output differs from current data or seed: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{output.name}.tmp-", dir=output.parent) as temp:
        temp_path = Path(temp)
        (temp_path / "splits.json").write_bytes(split_bytes)
        (temp_path / "vocab.json").write_bytes(vocab_bytes)
        (temp_path / "manifest.json").write_bytes(_json_bytes(manifest))
        temp_path.rename(output)
    return manifest


def load_prepared(output: Path, source: Path | None = None) -> dict[str, object]:
    """Verify artifacts and source before any training or evaluation uses them."""
    output = output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_bytes = (output / "splits.json").read_bytes()
    vocab_bytes = (output / "vocab.json").read_bytes()
    if sha256_bytes(split_bytes) != manifest["splits_sha256"]:
        raise ValueError("splits.json checksum differs from manifest")
    if sha256_bytes(vocab_bytes) != manifest["vocab_sha256"]:
        raise ValueError("vocab.json checksum differs from manifest")
    splits = json.loads(split_bytes)
    vocab = json.loads(vocab_bytes)
    if vocab["tokens"][: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
        raise ValueError("Special-token IDs are invalid")
    source = Path(source).resolve() if source is not None else (output / manifest["source_path"]).resolve()
    if sha256_bytes(source.read_bytes()) != manifest["source_sha256"]:
        raise ValueError("Source corpus checksum differs from manifest")
    poems = read_poems(source)
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        for record in splits[name]:
            index = record["source_index"]
            digest = record["sha256"]
            if not 0 <= index < len(poems) or poem_hash(poems[index]) != digest:
                raise ValueError(f"Invalid poem record in {name}")
            if digest in seen:
                raise ValueError(f"Poem leakage or duplicate in {name}")
            seen.add(digest)
    if len(seen) != manifest["poems_unique"]:
        raise ValueError("Split records do not cover all unique poems")
    return {"manifest": manifest, "splits": splits, "vocab": vocab, "poems": poems}


def encode_poem(poem: str, tokens: list[str]) -> list[int]:
    token_to_id = {token: index for index, token in enumerate(tokens)}
    return [1] + [token_to_id.get(char, 3) for char in poem] + [2]
