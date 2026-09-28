"""Classify custom text with a fine-tuned checkpoint (saved by train.py --save-model).

    python src/predict.py "the service was slow but the food was wonderful"
    python src/predict.py --ckpt checkpoints/E1_mrpc_seed42.pt --pair "He left." "He departed."
    echo "great phone, terrible battery" | python src/predict.py -
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TASKS, get_tokenizer, make_collate, special_ids, transform_pair, transform_single  # noqa: E402
from model import GPT, GPTClassifier, GPTConfig  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_classifier(path, device):
    ck = torch.load(path, map_location="cpu")
    gpt = GPT(GPTConfig(**ck["config"]))
    model = GPTClassifier(gpt, ck["n_classes"])
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return model.to(device).eval(), ck["task"]


@torch.no_grad()
def predict(model, task, inputs, device):
    tok = get_tokenizer(verbose=False)
    spec = TASKS[task]
    sp = special_ids(40478)
    if spec.pair:
        items = [(transform_pair(tok.encode(a), tok.encode(b), sp, spec.max_len), 0) for a, b in inputs]
    else:
        items = [(transform_single(tok.encode(t), sp, spec.max_len), 0) for t in inputs]
    b = make_collate(sp["<pad>"], spec.pair)(items)
    logits, _ = model(b["input_ids"].to(device), b["extract_idx"].to(device))
    probs = logits.float().softmax(-1).cpu()
    return [(spec.label_names[int(p.argmax())], float(p.max())) for p in probs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="texts to classify ('-' reads lines from stdin)")
    ap.add_argument("--ckpt", default=None, help="default: newest checkpoints/*sst2*.pt")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="sentence pair (for an MRPC checkpoint)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_ck = os.path.join(ROOT, "checkpoints", "E1_sst2_seed42.pt")
    ckpt = a.ckpt or (main_ck if os.path.exists(main_ck) else
                      max(glob.glob(os.path.join(ROOT, "checkpoints", "*sst2*.pt")), key=os.path.getmtime))
    print(f"[predict] checkpoint: {os.path.relpath(ckpt, ROOT)}")
    model, task = load_classifier(ckpt, device)
    if a.pair:
        inputs = [tuple(a.pair)]
    elif a.texts == ["-"]:
        inputs = [l.strip() for l in sys.stdin if l.strip()]
    else:
        inputs = a.texts or ["an absolute joy to watch from start to finish .",
                             "the plot is thin and the jokes fall flat ."]
    for x, (lab, p) in zip(inputs, predict(model, task, inputs, device)):
        print(f"{lab:>15s}  p={p:.3f}  | {x}")


if __name__ == "__main__":
    main()
