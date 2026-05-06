@echo off
cd /d D:\tiny-char-llm
chcp 65001 >nul
.\.venv\Scripts\python.exe .\train.py --data-path data\profile_input.txt --charset-path data\profile_charset.txt --checkpoint-path checkpoints\profile_best.pt --latest-checkpoint-path checkpoints\profile_last.pt --batch-size 16 --block-size 128 --n-embd 192 --n-head 6 --n-layer 4 --dropout 0.12 --learning-rate 0.00022 --min-learning-rate 0.00007 --lr-decay-iters 2400 --warmup-iters 8 --weight-decay 0.02 --grad-clip 1.0 --max-iters 24 --eval-interval 8 --eval-batches 2 --sample-tokens 140 --patience-evals 0
