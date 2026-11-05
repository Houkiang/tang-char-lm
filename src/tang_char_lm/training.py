"""Reproducible training, validation, checkpointing, and overfit diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import load_prepared
from .model import CharLanguageModel, ModelConfig
from .torch_data import PoemDataset, collate_poems, make_loader


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    prepared_dir: str = "data/processed/poetryFromTang-seed42"
    source_path: str = "data/raw/poetryFromTang.txt"
    run_dir: str = "runs/model"
    seed: int = 42
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = 8
    device: str = "auto"

    def __post_init__(self) -> None:
        if min(self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("batch_size, epochs, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("optimizer settings are invalid")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["model"] = self.model.to_dict()
        return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _targets_in_packed_order(targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return pack_padded_sequence(
        targets,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    ).data


def _metrics(total_nll: float, tokens: int) -> dict[str, float | int]:
    if tokens < 1:
        raise ValueError("Cannot compute language-model metrics with zero tokens")
    cross_entropy = total_nll / tokens
    return {
        "nll": total_nll,
        "tokens": tokens,
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
    }


def run_epoch(
    model: CharLanguageModel,
    loader: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    for inputs, targets, lengths in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        packed_targets = _targets_in_packed_order(targets, lengths)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model.forward_packed(inputs, lengths)
            loss = criterion(logits, packed_targets)
            if training:
                (loss / packed_targets.numel()).backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        total_nll += float(loss.detach().item())
        total_tokens += packed_targets.numel()
    return _metrics(total_nll, total_tokens)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_checkpoint(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _environment(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result.update(
            gpu_name=torch.cuda.get_device_name(device),
            gpu_capability=list(torch.cuda.get_device_capability(device)),
        )
    return result


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(config: TrainConfig, resume: bool = False) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = Path(config.run_dir).resolve()
    prepared_dir = Path(config.prepared_dir).resolve()
    source_path = Path(config.source_path).resolve()
    artifact = load_prepared(prepared_dir, source_path)
    if artifact["manifest"]["vocab_size"] != config.model.vocab_size:
        raise ValueError("Model vocabulary size differs from prepared vocabulary")

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config.to_dict():
            raise ValueError("Existing run configuration differs from requested configuration")
        if not resume:
            raise FileExistsError(f"Run already exists; pass --resume: {run_dir}")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume missing run: {run_dir}")
        _write_json(config_path, config.to_dict())
        _write_json(run_dir / "environment.json", _environment(device))

    model = CharLanguageModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    best_validation = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    elapsed_before = 0.0
    if resume:
        # Keep RNG tensors on CPU; module/optimizer loaders move parameter state as needed.
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]
        elapsed_before = checkpoint["elapsed_seconds"]
        _restore_rng(checkpoint["rng_state"])

    validation_loader = make_loader(
        prepared_dir,
        "validation",
        config.batch_size,
        source=source_path,
        shuffle=False,
        seed=config.seed,
    )
    started = time.perf_counter()
    can_continue = epochs_without_improvement < config.patience
    for epoch in range(start_epoch, config.epochs + 1) if can_continue else ():
        train_loader = make_loader(
            prepared_dir,
            "train",
            config.batch_size,
            source=source_path,
            shuffle=True,
            seed=config.seed + epoch,
        )
        train_metrics = run_epoch(model, train_loader, device, optimizer, config.gradient_clip)
        with torch.inference_mode():
            validation_metrics = run_epoch(model, validation_loader, device)
        improved = validation_metrics["cross_entropy"] < best_validation
        if improved:
            best_validation = float(validation_metrics["cross_entropy"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed = elapsed_before + time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "improved": improved,
            "elapsed_seconds": elapsed,
        }
        history.append(record)
        state = {
            "format_version": 1,
            "epoch": epoch,
            "model_config": config.model.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "elapsed_seconds": elapsed,
            "rng_state": _rng_state(),
        }
        _save_checkpoint(last_path, state)
        if improved:
            _save_checkpoint(best_path, state)
        _write_json(run_dir / "metrics.json", history)
        print(
            f"epoch={epoch:03d} "
            f"train_ppl={train_metrics['perplexity']:.4f} "
            f"val_ppl={validation_metrics['perplexity']:.4f} "
            f"best_val_ppl={math.exp(best_validation):.4f}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            break

    parameter_count = model.parameter_count
    summary = {
        "run_dir": str(run_dir),
        "model": config.model.rnn_type,
        "parameters": parameter_count,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "best_validation_cross_entropy": best_validation,
        "best_validation_perplexity": math.exp(best_validation),
        "elapsed_seconds": history[-1]["elapsed_seconds"] if history else elapsed_before,
        "stopped_early": epochs_without_improvement >= config.patience,
    }
    _write_json(run_dir / "summary.json", summary)
    # Release CUDA tensors before interpreter teardown on Windows.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        model.to("cpu")
        del optimizer
        if "state" in locals():
            del state
        torch.cuda.empty_cache()
    return summary


def overfit_check(
    rnn_type: str,
    prepared_dir: Path,
    source_path: Path,
    device_name: str = "auto",
    seed: int = 42,
    max_steps: int = 500,
    target_perplexity: float = 1.15,
) -> dict[str, object]:
    """Verify that the model can memorize one shortest training poem."""
    seed_everything(seed)
    device = resolve_device(device_name)
    artifact = load_prepared(prepared_dir, source_path)
    dataset = PoemDataset(prepared_dir, "train", source_path)
    sequence = min(dataset.sequences, key=len)
    inputs, targets, lengths = collate_poems([sequence])
    inputs, targets = inputs.to(device), targets.to(device)
    packed_targets = _targets_in_packed_order(targets, lengths)
    config = ModelConfig(
        vocab_size=artifact["manifest"]["vocab_size"],
        rnn_type=rnn_type,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
    )
    model = CharLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    initial_perplexity = math.inf
    perplexity = math.inf
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_packed(inputs, lengths)
        loss = criterion(logits, packed_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        perplexity = math.exp(float(loss.detach().item()))
        if step == 1:
            initial_perplexity = perplexity
        if perplexity <= target_perplexity:
            break
    passed = perplexity <= target_perplexity
    result = {
        "model": rnn_type,
        "device": str(device),
        "tokens": packed_targets.numel(),
        "steps": step,
        "initial_perplexity": initial_perplexity,
        "final_perplexity": perplexity,
        "target_perplexity": target_perplexity,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Overfit check failed: {result}")
    return result
