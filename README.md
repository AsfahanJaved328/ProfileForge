# ProfileForge

ProfileForge is a personal profile-writing project built in Python and
PyTorch. It trains a small character-level autoregressive language model that
learns from structured profile text and then generates profile-related writing
such as headlines, about sections, and project bullets.

It is intentionally small and readable:

- `src/data.py` builds the character vocabulary and training batches.
- `src/model.py` implements a decoder-only transformer from scratch.
- `train.py` runs gradient descent with cross-entropy loss.
- `generate.py` loads a checkpoint and samples new text.
- `data/charset.txt` defines the full allowed vocabulary.
- `media/Flonnect Recording - May 07, 2026.mp4` contains the demo video.

## Project Layout

```text
tiny-char-llm/
  .venv/
  checkpoints/
  data/
    charset.txt
    input.txt
    profile_charset.txt
    profile_input.txt
    profile_personal_notes.txt
    profile_prompts.json
  media/
    Flonnect Recording - May 07, 2026.mp4
  src/
    __init__.py
    config.py
    data.py
    model.py
    train_utils.py
  tests/
    test_smoke.py
  .gitignore
  generate.py
  profile_writer.py
  README.md
  requirements.txt
  train.py
```

## Setup

The local environment already exists at `D:\tiny-char-llm\.venv`.

Install dependencies:

```powershell
D:\tiny-char-llm\.venv\Scripts\python.exe -m pip install -r D:\tiny-char-llm\requirements.txt
```

## Train

```powershell
cd D:\tiny-char-llm
.\.venv\Scripts\python.exe .\train.py
```

Simple workflow:

```cmd
cd /d D:\tiny-char-llm
step1_test.bat
step2_train_short.bat
step3_train_more.bat
step6_profile_writer.bat
```

There is also a short guide in `SIMPLE_STEPS.md`.

By default, `train.py` auto-resumes from `checkpoints/last.pt` when that file
exists. This means repeated runs continue from the most recent saved training
state instead of starting from random weights.

Useful overrides:

```powershell
.\.venv\Scripts\python.exe .\train.py --max-iters 800 --batch-size 16 --block-size 48
```

Resume from a saved checkpoint:

```powershell
.\.venv\Scripts\python.exe .\train.py --resume-from checkpoints/best.pt --max-iters 300
```

Force a fresh run that ignores existing checkpoints:

```powershell
.\.venv\Scripts\python.exe .\train.py --fresh-start
```

Train using only characters found in the dataset instead of `data/charset.txt`:

```powershell
.\.venv\Scripts\python.exe .\train.py --charset-path ""
```

## Generate Text

```powershell
cd D:\tiny-char-llm
.\.venv\Scripts\python.exe .\generate.py --prompt "A small model" --max-new-tokens 200
```

Windows `cmd.exe` launchers:

```cmd
D:\tiny-char-llm\run_train_utf8.bat
D:\tiny-char-llm\run_generate_utf8.bat --prompt "A small model"
```

## Profile Writer

For LinkedIn-style output, use the structured profile generator:

```cmd
cd /d D:\tiny-char-llm
step6_profile_writer.bat
```

This produces section-based candidates for:

- headline
- about section
- project bullets
- skills summary

The output is also saved to `outputs/profile_samples.txt`.

## Smoke Test

```powershell
cd D:\tiny-char-llm
.\.venv\Scripts\python.exe -m unittest tests.test_smoke -v
```

## How The Model Learns

1. The text in `data/input.txt` is split into unique characters.
2. Each character is mapped to an integer id.
3. The model reads fixed-length windows of ids.
4. For every position, it predicts the id of the next character.
5. Cross-entropy loss measures how wrong the predictions are.
6. AdamW updates the model weights to reduce that loss.

The project uses a tiny decoder-only transformer with causal masking. Causal
masking means each character can only look at characters to its left when it
makes a prediction.

By default, the model uses a predefined multilingual charset from
`data/charset.txt`. It includes ASCII, accented Latin characters, and a base
Urdu character set with Urdu punctuation. That means the vocabulary size can be
larger than the set of characters that happen to appear in `data/input.txt`.
This expands what the model can represent, but it still learns best from
characters that are actually present in the training corpus.

If you change the charset, old checkpoints may become incompatible because the
embedding table and output layer dimensions depend on vocabulary size. After a
charset expansion, train a fresh checkpoint or use a new checkpoint path.

## Replacing The Dataset

You can replace `data/input.txt` with any plain text file. Better training data
usually improves the generated text more than making the model slightly bigger.
