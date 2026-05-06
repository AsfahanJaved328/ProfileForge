from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint-path", default="checkpoints/best.pt")
    parser.add_argument("--prompt", default="A small model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = load_checkpoint(ROOT / args.checkpoint_path, device=device)

    config = TrainingConfig.from_dict(checkpoint["config"])
    config.validate()

    vocab = CharacterVocabulary.from_chars(checkpoint["chars"])
    model = CharTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_ids = vocab.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(console_text(vocab.decode(generated[0])))


if __name__ == "__main__":
    main()
