from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import TrainingConfig
from src.data import TextDataset
from src.model import CharTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def estimate_loss(
    model: CharTransformer,
    dataset: TextDataset,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}

    for split in ("train", "val"):
        split_losses = torch.zeros(config.eval_batches)
        for batch_index in range(config.eval_batches):
            x, y = dataset.get_batch(
                split=split,
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
            )
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss should not be None during evaluation")
            split_losses[batch_index] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses


def save_checkpoint(
    path: str | Path,
    model: CharTransformer,
    config: TrainingConfig,
    dataset: TextDataset,
    step: int,
    best_val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "chars": dataset.vocab.chars,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)
