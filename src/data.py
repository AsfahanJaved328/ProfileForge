from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

SPECIAL_CHAR_TOKENS = {
    "<SPACE>": " ",
    "<TAB>": "\t",
    "<NEWLINE>": "\n",
    "<CR>": "\r",
}


@dataclass(slots=True)
class CharacterVocabulary:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharacterVocabulary":
        chars = sorted(set(text))
        return cls.from_chars(chars)

    @classmethod
    def from_charset_file(cls, path: str | Path) -> "CharacterVocabulary":
        raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
        chars: list[str] = []
        for line in raw_lines:
            if line in SPECIAL_CHAR_TOKENS:
                chars.append(SPECIAL_CHAR_TOKENS[line])
            elif len(line) == 1:
                chars.append(line)
            else:
                raise ValueError(
                    f"invalid charset entry {line!r}; use one literal character per line "
                    "or one of <SPACE>, <TAB>, <NEWLINE>, <CR>"
                )
        return cls.from_chars(chars)

    @classmethod
    def from_chars(cls, chars: list[str]) -> "CharacterVocabulary":
        if not chars:
            raise ValueError("character vocabulary cannot be empty")
        deduped = list(dict.fromkeys(chars))
        stoi = {ch: i for i, ch in enumerate(deduped)}
        itos = {i: ch for i, ch in enumerate(deduped)}
        return cls(chars=deduped, stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(
                    f"character {ch!r} is not in the active vocabulary; "
                    "add it to the current charset file or remove it from the dataset"
                )
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)


@dataclass(slots=True)
class TextDataset:
    vocab: CharacterVocabulary
    train_data: torch.Tensor
    val_data: torch.Tensor

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        train_split: float,
        charset_path: str | Path | None = None,
    ) -> "TextDataset":
        text = Path(path).read_text(encoding="utf-8")
        vocab: CharacterVocabulary | None = None
        if charset_path:
            vocab = CharacterVocabulary.from_charset_file(charset_path)
        return cls.from_text(text=text, train_split=train_split, vocab=vocab)

    @classmethod
    def from_text(
        cls,
        text: str,
        train_split: float,
        vocab: CharacterVocabulary | None = None,
    ) -> "TextDataset":
        if len(text) < 4:
            raise ValueError("training text must contain at least 4 characters")
        if vocab is None:
            vocab = CharacterVocabulary.from_text(text)
        encoded = torch.tensor(vocab.encode(text), dtype=torch.long)
        split_index = max(2, int(len(encoded) * train_split))
        split_index = min(split_index, len(encoded) - 2)
        train_data = encoded[:split_index]
        val_data = encoded[split_index:]
        return cls(vocab=vocab, train_data=train_data, val_data=val_data)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_data if split == "train" else self.val_data
        if len(source) <= block_size:
            raise ValueError(
                f"{split} split is too short for block_size={block_size}; "
                "use a longer dataset or a smaller block size"
            )
        starts = torch.randint(0, len(source) - block_size, (batch_size,))
        x = torch.stack([source[i : i + block_size] for i in starts])
        y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
        return x.to(device), y.to(device)
