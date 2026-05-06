@echo off
cd /d D:\tiny-char-llm
chcp 65001 >nul
.\.venv\Scripts\python.exe .\generate.py %*
