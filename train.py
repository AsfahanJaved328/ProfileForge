from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

from src.config import TrainingConfig
from src.data import TextDataset
from src.model import CharTransformer
from src.train_utils import (
    estimate_loss,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
)


ROOT = Path(__file__).resolve().parent


def console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train a tiny character-level transformer.")
    parser.add_argument("--data-path", default="data/input.txt", help="Path to the training corpus.")
    parser.add_argument(
        "--charset-path",
        default="data/charset.txt",
        help="Path to the allowed character set file. Pass an empty string to infer chars from the dataset.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/best.pt",
        help="Where to save the best checkpoint.",
    )
    parser.add_argument(
        "--latest-checkpoint-path",
        default="checkpoints/last.pt",
        help="Where to save the latest training state for automatic resume.",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint path to continue training from.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore saved checkpoints and start from random weights.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--max-iters", type=int, default=defaults.max_iters)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--lr-decay-iters", type=int, default=defaults.lr_decay_iters)
    parser.add_argument("--warmup-iters", type=int, default=defaults.warmup_iters)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--n-embd", type=int, default=defaults.n_embd)
    parser.add_argument("--n-head", type=int, default=defaults.n_head)
    parser.add_argument("--n-layer", type=int, default=defaults.n_layer)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--train-split", type=float, default=defaults.train_split)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience-evals", type=int, default=defaults.patience_evals)
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=160,
        help="How many characters to sample during evaluation snapshots.",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for progress snapshots.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        lr_decay_iters=args.lr_decay_iters,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        train_split=args.train_split,
        seed=args.seed,
        patience_evals=args.patience_evals,
    )
    config.validate()
    return config


def sample_text(
    model: CharTransformer,
    dataset: TextDataset,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = dataset.train_data[: min(16, dataset.train_data.size(0))].unsqueeze(0).to(device)
    generated = model.generate(
        idx=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=20,
    )
    return dataset.vocab.decode(generated[0])


def learning_rate_for_step(step: int, config: TrainingConfig) -> float:
    if config.warmup_iters > 0 and step < config.warmup_iters:
        return config.learning_rate * float(step + 1) / float(config.warmup_iters)
    if step >= config.lr_decay_iters:
        return config.min_learning_rate

    decay_ratio = (step - config.warmup_iters) / max(1, config.lr_decay_iters - config.warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    data_path = ROOT / args.data_path
    checkpoint_path = ROOT / args.checkpoint_path
    latest_checkpoint_path = ROOT / args.latest_checkpoint_path
    charset_path = ROOT / args.charset_path if args.charset_path else None
    resume_path = ROOT / args.resume_from if args.resume_from else None

    set_seed(config.seed)
    dataset = TextDataset.from_file(
        path=data_path,
        train_split=config.train_split,
        charset_path=charset_path,
    )
    config.vocab_size = dataset.vocab.size
    config.validate()

    device = select_device()
    model = CharTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_lr = learning_rate_for_step(step=0, config=config)
    for param_group in optimizer.param_groups:
        param_group["lr"] = initial_lr
    start_step = 0
    best_val_loss = float("inf")
    auto_resume = False

    if args.fresh_start:
        resume_path = None
    elif resume_path is None and latest_checkpoint_path.exists():
        resume_path = latest_checkpoint_path
        auto_resume = True

    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path, device=device)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            if auto_resume:
                print(
                    "latest checkpoint is incompatible with the current model shape; "
                    "starting a fresh training run instead."
                )
                resume_path = None
            else:
                raise RuntimeError(
                    "could not resume from the checkpoint. This usually means the model "
                    "shape changed, often because the charset or architecture changed."
                ) from exc
        if resume_path is not None:
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = int(checkpoint.get("step", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            if auto_resume:
                print(f"auto-resumed from: {resume_path}")
            else:
                print(f"resumed from: {resume_path}")
            print(f"resume step: {start_step}")

    print(f"device: {device}")
    print(f"vocab size: {dataset.vocab.size}")
    print(f"charset source: {charset_path if charset_path else 'dataset characters only'}")
    print(f"train characters: {dataset.train_data.size(0)}")
    print(f"val characters: {dataset.val_data.size(0)}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"session training steps: {config.max_iters}")
    print(f"best checkpoint path: {checkpoint_path}")
    print(f"latest checkpoint path: {latest_checkpoint_path}")

    def report(step: int) -> bool:
        nonlocal best_val_loss
        losses = estimate_loss(model=model, dataset=dataset, config=config, device=device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"step {step:4d} | train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | lr {current_lr:.6f}"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                config=config,
                dataset=dataset,
                step=step,
                best_val_loss=best_val_loss,
                optimizer=optimizer,
            )
            print(f"saved checkpoint: {checkpoint_path}")
            improved = True
        else:
            improved = False
        save_checkpoint(
            path=latest_checkpoint_path,
            model=model,
            config=config,
            dataset=dataset,
            step=step,
            best_val_loss=best_val_loss,
            optimizer=optimizer,
        )
        print(f"saved latest checkpoint: {latest_checkpoint_path}")
        print("sample:")
        print(console_text(sample_text(model, dataset, device, args.sample_tokens, args.sample_temperature)))
        print("-" * 60)
        return improved

    report(step=start_step)
    stale_evals = 0

    for session_step in range(1, config.max_iters + 1):
        step = start_step + session_step
        current_lr = learning_rate_for_step(step=step, config=config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        x, y = dataset.get_batch(
            split="train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
        )
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("loss should not be None during training")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        if session_step % config.eval_interval == 0 or session_step == config.max_iters:
            improved = report(step=step)
            if improved:
                stale_evals = 0
            else:
                stale_evals += 1
                if config.patience_evals and stale_evals >= config.patience_evals:
                    print(
                        f"early stopping at step {step} after "
                        f"{stale_evals} evaluation(s) without validation improvement"
                    )
                    break


if __name__ == "__main__":
    main()
