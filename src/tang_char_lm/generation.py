"""Reproducible decoding and surface-form analysis for generated Tang-style text."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import torch

from .data import load_prepared
from .evaluation import load_run_model
from .training import _write_json


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and "\u3400" <= token <= "\u9fff"


def _constraint_state(text: str, line_length: int) -> tuple[int, int]:
    punctuation = ("，", "。", "，", "。")
    line = 0
    characters = 0
    for token in text:
        if _is_cjk(token):
            if line >= 4 or characters >= line_length:
                raise ValueError("Prompt does not fit the requested regulated form")
            characters += 1
        elif token in {"，", "。"}:
            if line >= 4 or characters != line_length or token != punctuation[line]:
                raise ValueError("Prompt punctuation does not fit the requested regulated form")
            line += 1
            characters = 0
        elif token.isspace():
            continue
        else:
            raise ValueError(f"Constraint prompt contains unsupported character: {token!r}")
    return line, characters


def _constrained_ids(tokens: list[str], text: str, line_length: int) -> list[int]:
    line, characters = _constraint_state(text, line_length)
    if line == 4:
        return [2]
    if characters == line_length:
        expected = ("，", "。", "，", "。")[line]
        if expected not in tokens:
            raise ValueError(f"Vocabulary does not contain required punctuation {expected!r}")
        return [tokens.index(expected)]
    allowed = [index for index, token in enumerate(tokens) if _is_cjk(token)]
    if not allowed:
        raise ValueError("Vocabulary has no CJK characters for constrained decoding")
    return allowed


def _sample_id(
    logits: torch.Tensor,
    method: str,
    generator: torch.Generator,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if method == "greedy":
        return int(torch.argmax(logits).item())
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / temperature
    if method == "top_k":
        if top_k < 1:
            raise ValueError("top_k must be positive")
        count = min(top_k, logits.numel())
        threshold = torch.topk(logits, count).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    elif method == "top_p":
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.clone()
        logits[sorted_indices[remove]] = float("-inf")
    elif method != "sample":
        raise ValueError(f"Unknown decoding method: {method}")
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise RuntimeError("No valid token remains after decoding filters")
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate_text(
    model: torch.nn.Module,
    tokens: list[str],
    prompt: str,
    *,
    device: torch.device,
    method: str = "top_p",
    seed: int = 42,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int = 20,
    top_p: float = 0.9,
    line_length: int | None = None,
) -> dict[str, object]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if line_length not in (None, 5, 7):
        raise ValueError("line_length must be 5, 7, or None")
    token_to_id = {token: index for index, token in enumerate(tokens)}
    if line_length is not None:
        _constraint_state(prompt, line_length)
    prefix = [1] + [token_to_id.get(character, 3) for character in prompt]
    generator = torch.Generator(device=device.type).manual_seed(seed)
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    model.eval()
    text = prompt
    generated_ids: list[int] = []
    stopped_on_eos = False
    with torch.inference_mode():
        logits, state = model(input_ids)
        next_logits = logits[0, -1].clone()
        for _ in range(max_new_tokens):
            allowed = (
                _constrained_ids(tokens, text, line_length)
                if line_length is not None
                else [index for index in range(len(tokens)) if index not in (0, 1, 3)]
            )
            mask = torch.ones_like(next_logits, dtype=torch.bool)
            mask[torch.tensor(allowed, device=device)] = False
            filtered = next_logits.masked_fill(mask, float("-inf"))
            next_id = _sample_id(filtered, method, generator, temperature, top_k, top_p)
            generated_ids.append(next_id)
            if next_id == 2:
                stopped_on_eos = True
                break
            text += tokens[next_id]
            one = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, state = model(one, state)
            next_logits = logits[0, -1].clone()
    return {
        "text": text,
        "generated_token_ids": generated_ids,
        "stopped_on_eos": stopped_on_eos,
        "generated_tokens": len(generated_ids),
    }


def analyze_text(text: str, train_poems: set[str]) -> dict[str, object]:
    cjk = [character for character in text if _is_cjk(character)]
    bigrams = list(zip(cjk, cjk[1:]))
    segments = [segment for segment in re.split(r"[，。]", text) if segment]
    lengths = [sum(_is_cjk(character) for character in segment) for segment in segments]
    repeated_chars = len(cjk) - len(set(cjk))
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    return {
        "cjk_characters": len(cjk),
        "line_lengths": lengths,
        "five_char_line_ratio": sum(length == 5 for length in lengths) / len(lengths) if lengths else 0.0,
        "seven_char_line_ratio": sum(length == 7 for length in lengths) / len(lengths) if lengths else 0.0,
        "four_line_five_char_complete": lengths == [5, 5, 5, 5] and text.count("，") == 2 and text.count("。") == 2,
        "four_line_seven_char_complete": lengths == [7, 7, 7, 7] and text.count("，") == 2 and text.count("。") == 2,
        "repeated_character_ratio": repeated_chars / len(cjk) if cjk else 0.0,
        "repeated_bigram_ratio": repeated_bigrams / len(bigrams) if bigrams else 0.0,
        "exact_training_match": text in train_poems,
    }


def generate_suite(run_dir: Path, device_name: str = "cpu") -> dict[str, object]:
    run_dir = run_dir.resolve()
    output = run_dir / "generation.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    model, config, checkpoint, device = load_run_model(run_dir, device_name)
    artifact = load_prepared(Path(config["prepared_dir"]), Path(config["source_path"]))
    tokens = artifact["vocab"]["tokens"]
    poems = artifact["poems"]
    train_poems = {
        poems[record["source_index"]]
        for record in artifact["splits"]["train"]
    }
    decoders = (
        {"name": "greedy", "method": "greedy"},
        {"name": "temperature", "method": "sample", "temperature": 0.8},
        {"name": "top_k", "method": "top_k", "temperature": 0.9, "top_k": 20},
        {"name": "top_p", "method": "top_p", "temperature": 0.9, "top_p": 0.9},
        {"name": "constrained_5", "method": "top_p", "temperature": 0.9, "top_p": 0.9, "line_length": 5},
        {"name": "constrained_7", "method": "top_p", "temperature": 0.9, "top_p": 0.9, "line_length": 7},
    )
    samples: list[dict[str, object]] = []
    for prompt_index, prompt in enumerate(("春", "月", "山")):
        for decoder_index, decoder in enumerate(decoders):
            settings = dict(decoder)
            name = settings.pop("name")
            seed = 42 + prompt_index * 100 + decoder_index
            generated = generate_text(
                model,
                tokens,
                prompt,
                device=device,
                seed=seed,
                max_new_tokens=80,
                **settings,
            )
            samples.append(
                {
                    "prompt": prompt,
                    "decoder": name,
                    "seed": seed,
                    "settings": settings,
                    **generated,
                    "analysis": analyze_text(generated["text"], train_poems),
                }
            )
    aggregate: dict[str, object] = {}
    for decoder in decoders:
        name = decoder["name"]
        subset = [sample for sample in samples if sample["decoder"] == name]
        aggregate[name] = {
            "samples": len(subset),
            "mean_repeated_character_ratio": sum(sample["analysis"]["repeated_character_ratio"] for sample in subset) / len(subset),
            "mean_repeated_bigram_ratio": sum(sample["analysis"]["repeated_bigram_ratio"] for sample in subset) / len(subset),
            "exact_training_matches": sum(sample["analysis"]["exact_training_match"] for sample in subset),
            "complete_requested_form": sum(
                sample["analysis"]["four_line_five_char_complete"]
                or sample["analysis"]["four_line_seven_char_complete"]
                for sample in subset
            ),
        }
    result = {
        "format_version": 1,
        "model": config["model"]["rnn_type"],
        "seed": config["seed"],
        "best_epoch": checkpoint["epoch"],
        "evaluation_device": str(device),
        "prompts": ["春", "月", "山"],
        "aggregate": aggregate,
        "samples": samples,
    }
    _write_json(output, result)
    return result
