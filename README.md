# ProfileForge

ProfileForge is a personal profile-writing project built with Python and
PyTorch. It trains a small character-level transformer on structured
professional writing and generates LinkedIn-style profile text such as:

- headlines
- about sections
- project bullets
- profile summaries

## Demo Video

Watch the project demo here:

- [Flonnect Recording - May 07, 2026.mp4](./Flonnect%20Recording%20-%20May%2007,%202026.mp4)

The project was built as a practical learning exercise in model training,
checkpointing, prompt design, and local text generation.

## What It Does

- trains a character-level autoregressive transformer
- saves both best and latest checkpoints
- resumes training automatically
- generates structured profile-writing candidates
- includes a local demo video in `media/`

## Main Files

- `train.py`: training entry point
- `generate.py`: raw text generation
- `profile_writer.py`: structured profile generation
- `src/model.py`: transformer model
- `src/data.py`: dataset and vocabulary handling
- `data/profile_input.txt`: profile-focused training corpus
- `data/profile_personal_notes.txt`: personal notes used to steer output
- `data/profile_prompts.json`: prompt presets for each profile section

## Demo Video

The demo recording is included in the repository root so it appears directly on
the GitHub front page:

- `Flonnect Recording - May 07, 2026.mp4`

## Project Structure

```text
ProfileForge/
  checkpoints/
  data/
    profile_charset.txt
    profile_input.txt
    profile_personal_notes.txt
    profile_prompts.json
  outputs/
  src/
  tests/
  Flonnect Recording - May 07, 2026.mp4
  generate.py
  profile_writer.py
  train.py
  README.md
  SIMPLE_STEPS.md
```

## Quick Start

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Simple Workflow

Run these from the project folder:

```cmd
step1_test.bat
step2_train_short.bat
step3_train_more.bat
step6_profile_writer.bat
```

What each step does:

- `step1_test.bat`: smoke tests
- `step2_train_short.bat`: short training run
- `step3_train_more.bat`: continue profile-model training
- `step6_profile_writer.bat`: generate structured profile candidates

Generated output is saved to:

- `outputs/profile_samples.txt`

## Current Model Direction

The current profile-writing model is trained mainly for English professional
writing and is intended for:

- LinkedIn profile improvement
- project summaries
- resume-style bullet drafting
- portfolio presentation experiments

## Notes

- this is a local educational project, not a production LLM
- it is intentionally small enough to run on CPU
- output quality depends heavily on the quality of the profile dataset
- better personal data in `data/profile_input.txt` improves results

## Why This Project Matters

ProfileForge shows:

- practical Python and PyTorch work
- model training from scratch
- local checkpoint management
- dataset shaping for a specific use case
- structured generation workflows
- debugging and iteration on Windows
