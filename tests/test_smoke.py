from __future__ import annotations

import string
import unittest

import torch

from src.config import TrainingConfig
from src.data import CharacterVocabulary, TextDataset
from src.model import CharTransformer


class SmokeTests(unittest.TestCase):
    def test_forward_loss_and_generation(self) -> None:
        corpus = ("hello world\n" * 40) + ("tiny model\n" * 40)
        charset = CharacterVocabulary.from_chars(["\n", "\t", " "] + list(string.ascii_lowercase))
        dataset = TextDataset.from_text(text=corpus, train_split=0.9, vocab=charset)

        config = TrainingConfig(
            batch_size=4,
            block_size=8,
            max_iters=2,
            eval_interval=1,
            eval_batches=2,
            learning_rate=1e-3,
            n_embd=32,
            n_head=4,
            n_layer=2,
            dropout=0.0,
            vocab_size=dataset.vocab.size,
        )
        config.validate()
        self.assertGreater(dataset.vocab.size, len(set(corpus)))

        model = CharTransformer(config)
        x, y = dataset.get_batch(
            split="train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=torch.device("cpu"),
        )
        logits, loss = model(x, y)

        self.assertEqual(
            logits.shape,
            (config.batch_size, config.block_size, dataset.vocab.size),
        )
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss).item())

        seed = x[:1, :3]
        generated = model.generate(
            seed,
            max_new_tokens=5,
            temperature=1.0,
            top_k=5,
            top_p=0.95,
            repetition_penalty=1.05,
        )
        self.assertEqual(generated.shape[1], seed.shape[1] + 5)

    def test_config_from_dict_compatibility(self) -> None:
        legacy = {
            "batch_size": 8,
            "block_size": 16,
            "max_iters": 10,
            "eval_interval": 2,
            "eval_batches": 2,
            "learning_rate": 1e-3,
            "n_embd": 32,
            "n_head": 4,
            "n_layer": 2,
            "dropout": 0.1,
            "train_split": 0.9,
            "seed": 42,
            "vocab_size": 20,
        }
        config = TrainingConfig.from_dict(legacy)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.vocab_size, 20)
        self.assertGreater(config.lr_decay_iters, 0)
        config.validate()

    def test_multilingual_vocabulary_round_trip(self) -> None:
        chars = ["\n", " ", "A", "é", "ñ", "۔", "؟", "ی", "ہ", "ک"]
        vocab = CharacterVocabulary.from_chars(chars)
        text = "Aé\nیہ؟"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, text)


if __name__ == "__main__":
    unittest.main()
