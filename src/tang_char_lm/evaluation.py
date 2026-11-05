"""Held-out neural evaluation and smoothed statistical language-model baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

import torch

from .data import encode_poem, load_prepared, sha256_bytes
from .model import CharLanguageModel, ModelConfig
from .torch_data import make_loader
from .training import _write_json, resolve_device, run_epoch


def load_run_model(
    run_dir: Path,
    device_name: str = "cpu",
) -> tuple[CharLanguageModel, dict[str, object], dict[str, object], torch.device]:
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_config"] != config["model"]:
        raise ValueError("Checkpoint model configuration differs from run configuration")
    device = resolve_device(device_name)
    model = CharLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, config, checkpoint, device


def evaluate_test(run_dir: Path, device_name: str = "cpu") -> dict[str, object]:
    """Evaluate the selected best checkpoint once; repeated calls return the saved result."""
    run_dir = run_dir.resolve()
    output = run_dir / "test_metrics.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    model, config, checkpoint, device = load_run_model(run_dir, device_name)
    prepared = Path(config["prepared_dir"]).resolve()
    source = Path(config["source_path"]).resolve()
    artifact = load_prepared(prepared, source)
    loader = make_loader(
        prepared,
        "test",
        int(config["batch_size"]),
        source=source,
        shuffle=False,
        seed=int(config["seed"]),
    )
    with torch.inference_mode():
        metrics = run_epoch(model, loader, device)
    checkpoint_path = run_dir / "best.pt"
    result: dict[str, object] = {
        "format_version": 1,
        "model": config["model"]["rnn_type"],
        "seed": config["seed"],
        "best_epoch": checkpoint["epoch"],
        "checkpoint": "best.pt",
        "checkpoint_sha256": sha256_bytes(checkpoint_path.read_bytes()),
        "source_sha256": artifact["manifest"]["source_sha256"],
        "splits_sha256": artifact["manifest"]["splits_sha256"],
        "vocab_sha256": artifact["manifest"]["vocab_sha256"],
        "oov_rate": artifact["manifest"]["splits"]["test"]["oov_rate"],
        "metrics": metrics,
        "evaluation_device": str(device),
    }
    _write_json(output, result)
    return result


def _encoded_split(artifact: dict[str, object], split: str) -> list[list[int]]:
    tokens = artifact["vocab"]["tokens"]
    poems = artifact["poems"]
    return [
        encode_poem(poems[record["source_index"]], tokens)
        for record in artifact["splits"][split]
    ]


def _statistical_metrics(
    sequences: Iterable[list[int]],
    order: int,
    alpha: float,
    target_support: int,
    unigram_counts: Counter[int],
    context_counts: Counter[int],
    bigram_counts: dict[int, Counter[int]],
) -> dict[str, float | int]:
    total_nll = 0.0
    total_tokens = 0
    unigram_total = sum(unigram_counts.values())
    for sequence in sequences:
        for context, target in zip(sequence[:-1], sequence[1:]):
            if order == 1:
                probability = (unigram_counts[target] + alpha) / (
                    unigram_total + alpha * target_support
                )
            else:
                probability = (bigram_counts[context][target] + alpha) / (
                    context_counts[context] + alpha * target_support
                )
            total_nll -= math.log(probability)
            total_tokens += 1
    cross_entropy = total_nll / total_tokens
    return {
        "nll": total_nll,
        "tokens": total_tokens,
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
    }


def evaluate_baselines(
    prepared_dir: Path,
    source_path: Path,
    output: Path,
    alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0),
) -> dict[str, object]:
    """Select additive smoothing on validation, then evaluate test exactly once."""
    output = output.resolve()
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("All smoothing coefficients must be positive")
    artifact = load_prepared(prepared_dir, source_path)
    train_sequences = _encoded_split(artifact, "train")
    validation_sequences = _encoded_split(artifact, "validation")
    test_sequences = _encoded_split(artifact, "test")
    unigram_counts: Counter[int] = Counter()
    context_counts: Counter[int] = Counter()
    bigram_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for sequence in train_sequences:
        for context, target in zip(sequence[:-1], sequence[1:]):
            unigram_counts[target] += 1
            context_counts[context] += 1
            bigram_counts[context][target] += 1
    target_support = artifact["manifest"]["vocab_size"] - 2  # exclude pad and bos
    models: dict[str, object] = {}
    for order, name in ((1, "unigram"), (2, "bigram")):
        validation = {
            str(alpha): _statistical_metrics(
                validation_sequences,
                order,
                alpha,
                target_support,
                unigram_counts,
                context_counts,
                bigram_counts,
            )
            for alpha in alphas
        }
        selected = min(alphas, key=lambda alpha: validation[str(alpha)]["cross_entropy"])
        test = _statistical_metrics(
            test_sequences,
            order,
            selected,
            target_support,
            unigram_counts,
            context_counts,
            bigram_counts,
        )
        models[name] = {
            "selected_alpha": selected,
            "validation_grid": validation,
            "test": test,
        }
    result = {
        "format_version": 1,
        "selection": "minimum validation cross-entropy; test evaluated after alpha selection",
        "target_support": target_support,
        "source_sha256": artifact["manifest"]["source_sha256"],
        "splits_sha256": artifact["manifest"]["splits_sha256"],
        "vocab_sha256": artifact["manifest"]["vocab_sha256"],
        "test_oov_rate": artifact["manifest"]["splits"]["test"]["oov_rate"],
        "models": models,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    return result


def _summary_statistics(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_runs(
    run_dirs: list[Path],
    protocol_path: Path,
    output: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Audit protocol compliance and aggregate already-saved test results."""
    output = output.resolve()
    if output.exists() and not overwrite:
        return json.loads(output.read_text(encoding="utf-8"))
    protocol_bytes = protocol_path.resolve().read_bytes()
    protocol = json.loads(protocol_bytes)
    expected_seeds = set(protocol["random_seeds"])
    grouped: dict[str, list[dict[str, object]]] = {"lstm": [], "gru": []}
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        model = config["model"]["rnn_type"]
        seed = config["seed"]
        if model not in grouped or seed not in expected_seeds:
            raise ValueError(f"Unexpected model or seed in {run_dir}")
        expected_model = protocol["model_config"]
        for key in ("embedding_dim", "hidden_dim", "num_layers", "dropout"):
            if config["model"][key] != expected_model[key]:
                raise ValueError(f"Protocol model mismatch for {key} in {run_dir}")
        expected_training = protocol["training"]
        comparisons = {
            "batch_size": "batch_size",
            "epochs": "epochs_max",
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "gradient_clip": "gradient_clip",
            "patience": "early_stopping_patience",
        }
        for config_key, protocol_key in comparisons.items():
            if config[config_key] != expected_training[protocol_key]:
                raise ValueError(f"Protocol training mismatch for {config_key} in {run_dir}")
        for hash_key in ("source_sha256", "splits_sha256", "vocab_sha256"):
            if test[hash_key] != protocol["data"][hash_key]:
                raise ValueError(f"Protocol data mismatch for {hash_key} in {run_dir}")
        grouped[model].append(
            {
                "seed": seed,
                "run_dir": run_dir.name,
                "parameters": summary["parameters"],
                "best_epoch": test["best_epoch"],
                "training_seconds": summary["elapsed_seconds"],
                "test_cross_entropy": test["metrics"]["cross_entropy"],
                "test_perplexity": test["metrics"]["perplexity"],
                "checkpoint_sha256": test["checkpoint_sha256"],
            }
        )
    for model, rows in grouped.items():
        seeds = [row["seed"] for row in rows]
        if set(seeds) != expected_seeds or len(seeds) != len(expected_seeds):
            raise ValueError(f"Runs for {model} do not contain each protocol seed exactly once")
        rows.sort(key=lambda row: row["seed"])
    models: dict[str, object] = {}
    for model, rows in grouped.items():
        models[model] = {
            "runs": rows,
            "parameters": sorted({row["parameters"] for row in rows}),
            "best_epoch": _summary_statistics([float(row["best_epoch"]) for row in rows]),
            "training_seconds": _summary_statistics([float(row["training_seconds"]) for row in rows]),
            "test_cross_entropy": _summary_statistics([float(row["test_cross_entropy"]) for row in rows]),
            "test_perplexity": _summary_statistics([float(row["test_perplexity"]) for row in rows]),
        }
    lstm_by_seed = {row["seed"]: row for row in grouped["lstm"]}
    gru_by_seed = {row["seed"]: row for row in grouped["gru"]}
    paired = []
    for seed in sorted(expected_seeds):
        lstm_ppl = float(lstm_by_seed[seed]["test_perplexity"])
        gru_ppl = float(gru_by_seed[seed]["test_perplexity"])
        paired.append(
            {
                "seed": seed,
                "lstm_perplexity": lstm_ppl,
                "gru_perplexity": gru_ppl,
                "lstm_minus_gru": lstm_ppl - gru_ppl,
                "lstm_relative_reduction_from_gru": (gru_ppl - lstm_ppl) / gru_ppl,
            }
        )
    result = {
        "format_version": 1,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "seeds": sorted(expected_seeds),
        "models": models,
        "paired_comparison": paired,
        "paired_lstm_minus_gru": _summary_statistics([row["lstm_minus_gru"] for row in paired]),
        "paired_lstm_relative_reduction_from_gru": _summary_statistics(
            [row["lstm_relative_reduction_from_gru"] for row in paired]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    return result


def export_public_results(run_root: Path, output_dir: Path) -> dict[str, object]:
    """Export lightweight, path-safe JSON snapshots without checkpoints or host metadata."""
    run_root = run_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []

    def export_json(source: Path, destination: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Missing result artifact: {source}")
        value = json.loads(source.read_text(encoding="utf-8"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, value)
        exported.append(destination.relative_to(output_dir).as_posix())

    export_json(run_root / "baselines-seed42.json", output_dir / "baselines-seed42.json")
    export_json(run_root / "multiseed_summary.json", output_dir / "multiseed_summary.json")
    for model in ("lstm", "gru"):
        for seed in (42, 43, 44):
            name = f"{model}-seed{seed}"
            source_dir = run_root / name
            destination_dir = output_dir / name
            for filename in ("config.json", "metrics.json", "test_metrics.json"):
                export_json(source_dir / filename, destination_dir / filename)
            generation = source_dir / "generation.json"
            if generation.is_file():
                export_json(generation, destination_dir / "generation.json")
    manifest = {
        "format_version": 1,
        "author": "Houkiang",
        "description": "Lightweight public experiment snapshots; model checkpoints are excluded.",
        "files": sorted(exported),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
