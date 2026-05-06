from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.config import TrainingConfig


class AttentionHead(nn.Module):
    def __init__(self, head_size: int, config: TrainingConfig) -> None:
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-2, -1)
        weights = weights * (key.size(-1) ** -0.5)
        weights = weights.masked_fill(self.mask[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        value = self.value(x)
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList(
            AttentionHead(head_size=head_size, config=config) for _ in range(config.n_head)
        )
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("config.vocab_size must be set before creating the model")

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.n_layer))
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time_steps = idx.shape
        if time_steps > self.config.block_size:
            raise ValueError(
                f"sequence length {time_steps} exceeds block size {self.config.block_size}"
            )

        token_embeddings = self.token_embedding(idx)
        positions = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            batch_size, time_steps, channels = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * time_steps, channels),
                targets.view(batch_size * time_steps),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1 when provided")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for batch_index in range(idx_cond.size(0)):
                    seen_token_ids = torch.unique(idx_cond[batch_index])
                    selected = logits[batch_index, seen_token_ids]
                    adjusted = torch.where(
                        selected > 0,
                        selected / repetition_penalty,
                        selected * repetition_penalty,
                    )
                    logits[batch_index, seen_token_ids] = adjusted

            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                cutoff = values[:, [-1]]
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
                remove_mask.scatter_(1, sorted_indices, sorted_remove)
                logits = logits.masked_fill(remove_mask, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
