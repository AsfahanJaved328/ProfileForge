@echo off
cd /d D:\tiny-char-llm
chcp 65001 >nul
.\.venv\Scripts\python.exe .\profile_writer.py --checkpoint-path checkpoints\profile_best.pt --prompts-path data\profile_prompts.json --notes-path data\profile_personal_notes.txt --output-path outputs\profile_samples.txt %*
