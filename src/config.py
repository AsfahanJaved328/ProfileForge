from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 1500
    eval_interval: int = 150
    eval_batches: int = 40
    learning_rate: float = 3e-4
    min_learning_rate: float = 8e-5
    lr_decay_iters: int = 3000
    warmup_iters: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    train_split: float = 0.9
    seed: int = 1337
    patience_evals: int = 6
    vocab_size: int | None = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be less than or equal to learning_rate")
        if self.lr_decay_iters <= 0:
            raise ValueError("lr_decay_iters must be positive")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be 0 or greater")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be 0 or greater")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if self.patience_evals < 0:
            raise ValueError("patience_evals must be 0 or greater")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive when provided")

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | float | None]) -> "TrainingConfig":
        defaults = cls().to_dict()
        defaults.update(data)
        return cls(**defaults)
