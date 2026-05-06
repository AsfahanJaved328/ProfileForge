# Training Explained

This file explains what happens when you run:

```cmd
.\.venv\Scripts\python.exe .\train.py
```

## 1. Load Text And Charset

The training script reads:

- `data/input.txt`: the text the model will learn from
- `data/charset.txt`: the full list of characters the model is allowed to use

This means the model vocabulary is no longer limited to only the characters
that happen to appear in `input.txt`.

In the current project, the charset includes:

- ASCII symbols and digits
- accented Latin characters
- Urdu letters
- Urdu punctuation such as `،`, `؟`, and `۔`

Example:

- If `input.txt` contains 57 unique characters
- but `charset.txt` contains 97 characters

then the model will use a vocabulary size of 97.

## 2. Convert Characters To Integer Ids

Neural networks do not read raw text directly. Every character is mapped to an
integer.

Example:

- `A -> 35`
- `m -> 79`
- `space -> 2`

The whole text becomes a long list of ids.

## 3. Build Training Pairs

The model learns from fixed windows of text.

If the window is 8 characters:

```text
input:  "model si"
target: "odel sit"
```

The target is just the same sequence shifted one character to the left. So for
every position, the model must predict the next character.

## 4. Turn Ids Into Vectors

The model uses an embedding table.

This means each character id is turned into a learned vector of numbers. Similar
characters or commonly related characters can end up with useful internal
representations.

## 5. Add Position Information

The same character can mean different things in different places. So the model
also adds a position embedding that tells it where each character appears inside
the current context window.

## 6. Run Masked Self-Attention

This is the core transformer step.

For each position, the model asks:

- Which earlier characters matter most?
- How strongly should I use each one?

The causal mask blocks future positions. That means the model can look left, but
never right.

So when predicting the next character after:

```text
"the rain"
```

the model can use `"the rain"` as context, but it cannot peek at the correct
next character.

## 7. Score Every Possible Next Character

The final linear layer produces one score for every character in the vocabulary.

If the vocabulary size is 97, then each prediction step outputs 97 scores.

These scores are called logits.

## 8. Measure Error With Cross-Entropy

Cross-entropy compares:

- the model's predicted distribution
- the true next character

If the model puts high probability on the correct next character, loss is low.
If it spreads probability badly or favors the wrong character, loss is high.

## 9. Backpropagate And Update Weights

PyTorch computes gradients for every trainable parameter.

Then `AdamW` updates the weights to reduce future error.

This is the actual learning step.

The model structure does not grow during training. What changes are the numeric
weights inside the existing layers.

## 10. Save The Best Checkpoint

At evaluation intervals, the script checks training and validation loss.

If validation loss improves, it saves:

- model weights
- optimizer state
- config
- vocabulary characters
- best validation score
- current step

to `checkpoints/best.pt`.

It also saves the latest training state, including optimizer progress, to
`checkpoints/last.pt`. That file is what the training script uses for automatic
resume.

## 11. Resume Training Later

You can continue from the saved checkpoint:

```cmd
.\.venv\Scripts\python.exe .\train.py --resume-from checkpoints/best.pt --max-iters 300
```

This continues training from the saved weights instead of starting from random
weights.

If `checkpoints/last.pt` exists, plain `train.py` now resumes from it
automatically. Use `--fresh-start` only when you intentionally want to discard
that continuation path for a new run.

## 12. Why Validation Loss Can Get Worse

If training loss keeps dropping but validation loss rises, the model is
overfitting.

That means it is memorizing the training text too closely instead of learning
general patterns that transfer to unseen text.

The script now supports early stopping with `--patience-evals` to stop after too
many evaluation rounds without improvement.

## 13. What Charset Expansion Really Changes

When you expand the charset:

- the embedding table gets more rows
- the final output layer predicts more possible characters
- the total parameter count increases slightly

What does not change:

- the overall transformer design
- the training algorithm
- the fact that the model only learns well from characters it actually sees in
  the corpus
