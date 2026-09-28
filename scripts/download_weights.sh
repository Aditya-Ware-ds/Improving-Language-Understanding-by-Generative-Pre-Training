#!/bin/bash
# Resumable download of the GPT-1 files into weights/openai-gpt/ (curl -C - resumes after any drop).
# Use this if `from_pretrained` downloads are slow or keep failing. The code picks the local folder
# up automatically (see src/paths.py). Progress bar shown by curl.
set -u
cd "$(dirname "$0")/.."
DIR=weights/openai-gpt
mkdir -p "$DIR"
BASE=https://huggingface.co/openai-community/openai-gpt/resolve/main
# reuse a partial download started earlier in this project's setup, if present
if [ -f "$HOME/gpt1_dl/model.safetensors" ] && [ ! -f "$DIR/model.safetensors" ]; then
  echo "reusing partial download from ~/gpt1_dl"; cp "$HOME/gpt1_dl/model.safetensors" "$DIR/"
fi
for f in config.json vocab.json merges.txt tokenizer.json tokenizer_config.json model.safetensors; do
  for i in $(seq 1 100); do
    echo "[$f] attempt $i"
    curl -L --retry 5 --connect-timeout 30 --speed-time 60 --speed-limit 1000 -C - --progress-bar \
         -o "$DIR/$f" "$BASE/$f" && break
    sleep 5
  done
done
ls -la "$DIR"
echo "model.safetensors should be ~479 MB (compare with the size on huggingface.co/openai-community/openai-gpt/tree/main); a truncated file fails to load, then just re-run this script."
