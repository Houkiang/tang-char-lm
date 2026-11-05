"""LSTM and GRU character language models with one shared interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    rnn_type: str
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if min(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CharLanguageModel(nn.Module):
    """Embedding -> unidirectional LSTM/GRU -> vocabulary projection."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        rnn_class = nn.LSTM if config.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input_ids)
        recurrent, next_state = self.rnn(embedded, state)
        return self.output(recurrent), next_state

    def forward_packed(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return logits for valid positions only, in PackedSequence data order."""
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        recurrent, _ = self.rnn(packed)
        return self.output(recurrent.data)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
