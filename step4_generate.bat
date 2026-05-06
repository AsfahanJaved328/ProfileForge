@echo off
cd /d D:\tiny-char-llm
chcp 65001 >nul
.\.venv\Scripts\python.exe .\generate.py --checkpoint-path checkpoints\profile_best.pt --prompt "Professional summary: " --max-new-tokens 260
