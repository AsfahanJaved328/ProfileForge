from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from src.config import TrainingConfig
from src.data import CharacterVocabulary
from src.model import CharTransformer
from src.train_utils import load_checkpoint, select_device


ROOT = Path(__file__).resolve().parent


def console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured profile-writing samples.")
    parser.add_argument("--checkpoint-path", default="checkpoints/profile_best.pt")
    parser.add_argument("--prompts-path", default="data/profile_prompts.json")
    parser.add_argument("--notes-path", default="")
    parser.add_argument("--output-path", default="outputs/profile_samples.txt")
    parser.add_argument("--temperature-multiplier", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="Override top-k for all sections when > 0.")
    parser.add_argument("--top-p", type=float, default=0.0, help="Override top-p for all sections when > 0.")
    parser.add_argument("--repetition-penalty", type=float, default=0.0, help="Override repetition penalty when > 0.")
    parser.add_argument("--pool-factor", type=int, default=3, help="Generate more candidates per section and keep the best.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CharTransformer, CharacterVocabulary]:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = TrainingConfig.from_dict(checkpoint["config"])
    config.validate()
    vocab = CharacterVocabulary.from_chars(checkpoint["chars"])
    model = CharTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


def sanitize_for_vocab(text: str, vocab: CharacterVocabulary) -> str:
    sanitized: list[str] = []
    for ch in text:
        if ch in vocab.stoi:
            sanitized.append(ch)
        elif ch in "\r\t":
            sanitized.append(" ")
        else:
            sanitized.append(" ")
    return "".join(sanitized)


def clean_candidate(text: str, section: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    if section == "headline":
        candidate = re.sub(r"\s+", " ", lines[0]).strip(" -")
        return candidate[:120]

    if section == "project_bullet":
        candidate = re.sub(r"\s+", " ", lines[0]).strip()
        if not candidate.startswith("- "):
            candidate = "- " + candidate.lstrip("- ").strip()
        return candidate[:180]

    candidate = " ".join(lines[:3])
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate[:360]


def score_candidate(candidate: str, section: str) -> float:
    if not candidate:
        return float("-inf")

    total = max(1, len(candidate))
    alpha_space_ratio = sum(ch.isalpha() or ch == " " for ch in candidate) / total
    digit_penalty = sum(ch.isdigit() for ch in candidate) / total
    punctuation_penalty = sum(ch in "@%&/+" for ch in candidate) / total
    repeated_penalty = len(re.findall(r"(.)\1{2,}", candidate)) * 0.12
    short_penalty = 0.0
    length_bonus = 0.0

    if section == "headline":
        if 40 <= len(candidate) <= 95:
            length_bonus += 0.25
        else:
            short_penalty += 0.15
        if "|" in candidate:
            length_bonus += 0.15
    elif section == "project_bullet":
        if candidate.startswith("- "):
            length_bonus += 0.15
        if 70 <= len(candidate) <= 170:
            length_bonus += 0.25
        else:
            short_penalty += 0.15
    elif section == "about":
        if 160 <= len(candidate) <= 320:
            length_bonus += 0.25
        else:
            short_penalty += 0.15
    elif section == "skills_summary":
        if 80 <= len(candidate) <= 180:
            length_bonus += 0.2
        if "," in candidate:
            length_bonus += 0.1

    keyword_bonus = 0.0
    lower = candidate.lower()
    for keyword in ("python", "project", "software", "machine", "learning", "development"):
        if keyword in lower:
            keyword_bonus += 0.04

    return alpha_space_ratio + length_bonus + keyword_bonus - digit_penalty - punctuation_penalty - repeated_penalty - short_penalty


def build_prefix(section_prompt: str, notes: str, vocab: CharacterVocabulary) -> str:
    prefix_parts: list[str] = []
    if notes.strip():
        prefix_parts.append("Personal notes:\n")
        prefix_parts.append(notes.strip())
        prefix_parts.append("\n\n")
    prefix_parts.append(section_prompt)
    return sanitize_for_vocab("".join(prefix_parts), vocab)


def generate_completion(
    model: CharTransformer,
    vocab: CharacterVocabulary,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prompt_ids = vocab.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    decoded = vocab.decode(generated[0])
    return decoded[len(prompt) :]


def main() -> None:
    args = parse_args()
    device = select_device()
    model, vocab = load_model(ROOT / args.checkpoint_path, device=device)

    prompts = json.loads((ROOT / args.prompts_path).read_text(encoding="utf-8"))
    notes = ""
    if args.notes_path:
        notes_path = ROOT / args.notes_path
        if notes_path.exists():
            notes = notes_path.read_text(encoding="utf-8")

    sections_output: list[str] = []

    for section_name, section_config in prompts.items():
        header = section_name.replace("_", " ").title()
        sections_output.append(f"{header}")
        sections_output.append("-" * len(header))

        prefix = build_prefix(section_config["prompt"], notes, vocab)
        section_top_k = args.top_k if args.top_k > 0 else int(section_config["top_k"])
        section_top_p = args.top_p if args.top_p > 0 else float(section_config.get("top_p", 0.92))
        section_repetition_penalty = (
            args.repetition_penalty
            if args.repetition_penalty > 0
            else float(section_config.get("repetition_penalty", 1.08))
        )
        section_temp = float(section_config["temperature"]) * args.temperature_multiplier
        samples = int(section_config["samples"])
        pool_size = max(samples, samples * args.pool_factor)
        ranked_candidates: list[tuple[float, str]] = []
        seen_candidates: set[str] = set()

        for _ in range(pool_size):
            completion = generate_completion(
                model=model,
                vocab=vocab,
                device=device,
                prompt=prefix,
                max_new_tokens=int(section_config["max_new_tokens"]),
                temperature=section_temp,
                top_k=section_top_k,
                top_p=section_top_p,
                repetition_penalty=section_repetition_penalty,
            )
            cleaned = clean_candidate(completion, section_name)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            ranked_candidates.append((score_candidate(cleaned, section_name), cleaned))

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)

        if not ranked_candidates:
            sections_output.append("1. [empty sample]")
        else:
            for index, (_, candidate) in enumerate(ranked_candidates[:samples], start=1):
                sections_output.append(f"{index}. {candidate}")

        sections_output.append("")

    output_text = "\n".join(sections_output).rstrip() + "\n"
    output_path = ROOT / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")

    print(console_text(output_text))
    print(f"saved samples: {output_path}")


if __name__ == "__main__":
    main()
