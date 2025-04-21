#!/usr/bin/env bash
# queue.sh — 顺序运行多个 Python 脚本，行为等同于直接运行 “python main.py”
for script in "$@"; do
  python "$script" || exit $?
done
