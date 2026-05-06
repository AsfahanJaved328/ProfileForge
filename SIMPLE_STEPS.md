# Simple Steps

Use these steps in order.

## Step 1: Check That Everything Works

```cmd
cd /d D:\tiny-char-llm
step1_test.bat
```

This runs the smoke test.

## Step 2: Do A Small Training Run

```cmd
cd /d D:\tiny-char-llm
step2_train_short.bat
```

This is a short run for quick checking.
It uses the safer profile-writing dataset and a modestly larger model.

## Step 3: Continue Training

```cmd
cd /d D:\tiny-char-llm
step3_train_more.bat
```

This continues from the last checkpoint automatically.
It keeps using the profile-specific checkpoints:

- `checkpoints/profile_last.pt`
- `checkpoints/profile_best.pt`

## Step 4: Generate Text

```cmd
cd /d D:\tiny-char-llm
step4_generate.bat
```

This loads the best saved model and prints generated text.

## Step 5: Generate Profile Candidates

```cmd
cd /d D:\tiny-char-llm
step6_profile_writer.bat
```

This creates separate candidates for:

- headline
- about section
- project bullets
- skills summary

The samples are also saved to `outputs/profile_samples.txt`.

## Step 6: Add New Data

Open `data/input.txt` and add more text at the end.
For the profile workflow, the better file to extend is `data/profile_input.txt`.

Then run:

```cmd
cd /d D:\tiny-char-llm
step3_train_more.bat
```

The model will keep its old learning and continue from the latest checkpoint.

## Step 7: Start Fresh Only If You Really Want To

```cmd
cd /d D:\tiny-char-llm
step5_fresh_start.bat
```

Use this only when you want a brand new model.

## What Each Checkpoint Means

- `checkpoints/last.pt`: the newest training state, used for auto-resume
- `checkpoints/best.pt`: the best validation checkpoint, best choice for generation
- `checkpoints/profile_last.pt`: the newest training state for the profile-writing model
- `checkpoints/profile_best.pt`: the best validation checkpoint for the profile-writing model
