"""PyTorch adapter for the verified poem-level corpus."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .data import SPLIT_NAMES, encode_poem, load_prepared


class PoemDataset(Dataset[list[int]]):
    def __init__(self, prepared_dir: Path, split: str, source: Path | None = None) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        artifact = load_prepared(prepared_dir, source)
        tokens = artifact["vocab"]["tokens"]
        poems = artifact["poems"]
        records = artifact["splits"][split]
        self.sequences = [encode_poem(poems[record["source_index"]], tokens) for record in records]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


def collate_poems(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return input, next-token targets, and target lengths; padding ID is zero."""
    if not sequences:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(sequence) - 1 for sequence in sequences], dtype=torch.long)
    width = int(lengths.max().item())
    inputs = torch.full((len(sequences), width), 0, dtype=torch.long)
    targets = torch.full((len(sequences), width), 0, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        length = len(sequence) - 1
        inputs[row, :length] = torch.tensor(sequence[:-1], dtype=torch.long)
        targets[row, :length] = torch.tensor(sequence[1:], dtype=torch.long)
    return inputs, targets, lengths


def make_loader(
    prepared_dir: Path,
    split: str,
    batch_size: int,
    *,
    source: Path | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = PoemDataset(prepared_dir, split, source)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_poems,
        generator=generator,
        num_workers=0,
    )
