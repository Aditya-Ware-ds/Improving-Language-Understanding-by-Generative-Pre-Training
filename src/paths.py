"""Where to load GPT-1 from: a local folder (scripts/download_weights.sh) if present, else the HF Hub."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL = os.path.join(ROOT, "weights", "openai-gpt")


def gpt_source() -> str:
    if os.path.exists(os.path.join(LOCAL, "model.safetensors")) and os.path.exists(os.path.join(LOCAL, "config.json")):
        return LOCAL
    return "openai-community/openai-gpt"
