"""Small standard-library commands for checking the research workspace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

from .data import load_prepared, prepare_data, read_poems

DATA_URL = "https://raw.githubusercontent.com/nndl/llm-beginner/master/poetryFromTang.txt"
DEFAULT_DATA = Path("data/raw/poetryFromTang.txt")


def data_stats(path: Path) -> dict[str, object]:
    poems = read_poems(path)
    if not poems:
        raise ValueError(f"No poems found in {path}")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": sha256,
        "poems": len(poems),
        "exact_duplicate_poems": len(poems) - len(set(poems)),
        "characters_including_line_breaks": sum(len(poem) for poem in poems),
        "unique_characters": len(set("".join(poems))),
    }


def fetch_data(path: Path, force: bool) -> dict[str, object]:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; use --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATA_URL, headers={"User-Agent": "tang-char-lm/0.1"})
    with urlopen(request, timeout=30) as response:
        content = response.read()
    content.decode("utf-8-sig")
    if not content.strip():
        raise ValueError("Downloaded dataset is empty")
    descriptor, temporary_name = tempfile.mkstemp(prefix="poetry-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        temporary_path = Path(temporary_name)
        data_stats(temporary_path)
        temporary_path.replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return data_stats(path)


def doctor() -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "dataset_present": DEFAULT_DATA.exists(),
        "xelatex_available": shutil.which("xelatex") is not None,
    }
    if report["torch_installed"]:
        try:
            import torch

            report["torch_version"] = torch.__version__
            report["torch_cuda_build"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                report["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as error:
            report["torch_import_error"] = f"{type(error).__name__}: {error}"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tang-lm")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show available local prerequisites")
    data = commands.add_parser("data", help="manage the original corpus")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="download the upstream corpus")
    fetch.add_argument("--path", type=Path, default=DEFAULT_DATA)
    fetch.add_argument("--force", action="store_true")
    stats = data_commands.add_parser("stats", help="inspect corpus size and checksum")
    stats.add_argument("--path", type=Path, default=DEFAULT_DATA)
    prepare = data_commands.add_parser("prepare", help="create poem-level splits and train-only vocabulary")
    prepare.add_argument("--path", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--output", type=Path, default=Path("data/processed/poetryFromTang-seed42"))
    prepare.add_argument("--seed", type=int, default=42)
    audit = data_commands.add_parser("audit", help="verify prepared data and report split statistics")
    audit.add_argument("--path", type=Path, default=DEFAULT_DATA)
    audit.add_argument("--output", type=Path, default=Path("data/processed/poetryFromTang-seed42"))
    overfit = commands.add_parser("overfit-check", help="verify that LSTM/GRU can memorize a tiny sample")
    overfit.add_argument("--model", choices=("lstm", "gru", "both"), default="both")
    overfit.add_argument("--path", type=Path, default=DEFAULT_DATA)
    overfit.add_argument("--prepared", type=Path, default=Path("data/processed/poetryFromTang-seed42"))
    overfit.add_argument("--device", default="auto")
    overfit.add_argument("--seed", type=int, default=42)
    overfit.add_argument("--max-steps", type=int, default=500)
    overfit.add_argument("--target-perplexity", type=float, default=1.15)
    train_parser = commands.add_parser("train", help="train an LSTM or GRU language model")
    train_parser.add_argument("--model", choices=("lstm", "gru"), required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--path", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--prepared", type=Path, default=Path("data/processed/poetryFromTang-seed42"))
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--embedding-dim", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="evaluate one selected best checkpoint on test")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    baseline = commands.add_parser("baseline", help="tune and evaluate unigram/bigram baselines")
    baseline.add_argument("--path", type=Path, default=DEFAULT_DATA)
    baseline.add_argument("--prepared", type=Path, default=Path("data/processed/poetryFromTang-seed42"))
    baseline.add_argument("--output", type=Path, default=Path("runs/baselines-seed42.json"))
    generate = commands.add_parser("generate", help="generate text from one best checkpoint")
    generate.add_argument("--run-dir", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--method", choices=("greedy", "sample", "top_k", "top_p"), default="top_p")
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--max-new-tokens", type=int, default=80)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--line-length", type=int, choices=(5, 7))
    suite = commands.add_parser("generate-suite", help="generate the fixed reproducible sample suite")
    suite.add_argument("--run-dir", type=Path, required=True)
    suite.add_argument("--device", default="cpu")
    aggregate = commands.add_parser("aggregate", help="audit and aggregate protocol test runs")
    aggregate.add_argument("--run-dir", type=Path, action="append", required=True)
    aggregate.add_argument("--protocol", type=Path, default=Path("configs/experiment_protocol_v1.json"))
    aggregate.add_argument("--output", type=Path, default=Path("runs/multiseed_summary.json"))
    aggregate.add_argument("--force", action="store_true")
    export_results = commands.add_parser("export-results", help="export lightweight public result snapshots")
    export_results.add_argument("--run-root", type=Path, default=Path("runs"))
    export_results.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "overfit-check":
            from .training import overfit_check

            models = ("lstm", "gru") if args.model == "both" else (args.model,)
            result = [
                overfit_check(
                    model,
                    args.prepared,
                    args.path,
                    args.device,
                    args.seed,
                    args.max_steps,
                    args.target_perplexity,
                )
                for model in models
            ]
        elif args.command == "train":
            from .model import ModelConfig
            from .training import TrainConfig, train

            artifact = load_prepared(args.prepared, args.path)
            model_config = ModelConfig(
                vocab_size=artifact["manifest"]["vocab_size"],
                rnn_type=args.model,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            result = train(
                TrainConfig(
                    model=model_config,
                    prepared_dir=str(args.prepared),
                    source_path=str(args.path),
                    run_dir=str(args.run_dir),
                    seed=args.seed,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    gradient_clip=args.gradient_clip,
                    patience=args.patience,
                    device=args.device,
                ),
                resume=args.resume,
            )
        elif args.command == "evaluate":
            from .evaluation import evaluate_test

            result = evaluate_test(args.run_dir, args.device)
        elif args.command == "baseline":
            from .evaluation import evaluate_baselines

            result = evaluate_baselines(args.prepared, args.path, args.output)
        elif args.command == "generate":
            from .evaluation import load_run_model
            from .generation import analyze_text, generate_text

            model, config, _, device = load_run_model(args.run_dir, args.device)
            artifact = load_prepared(Path(config["prepared_dir"]), Path(config["source_path"]))
            tokens = artifact["vocab"]["tokens"]
            poems = artifact["poems"]
            train_poems = {
                poems[record["source_index"]]
                for record in artifact["splits"]["train"]
            }
            generated = generate_text(
                model,
                tokens,
                args.prompt,
                device=device,
                method=args.method,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                line_length=args.line_length,
            )
            result = {**generated, "analysis": analyze_text(generated["text"], train_poems)}
        elif args.command == "generate-suite":
            from .generation import generate_suite

            result = generate_suite(args.run_dir, args.device)
        elif args.command == "aggregate":
            from .evaluation import aggregate_runs

            result = aggregate_runs(args.run_dir, args.protocol, args.output, args.force)
        elif args.command == "export-results":
            from .evaluation import export_public_results

            result = export_public_results(args.run_root, args.output)
        elif args.data_command == "fetch":
            result = fetch_data(args.path, args.force)
        elif args.data_command == "prepare":
            result = prepare_data(args.path, args.output, args.seed)
        elif args.data_command == "audit":
            result = load_prepared(args.output, args.path)["manifest"]
        else:
            result = data_stats(args.path)
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
