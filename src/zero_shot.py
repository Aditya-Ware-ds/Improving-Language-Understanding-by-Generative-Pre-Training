"""E5: zero-shot SST-2 with the pre-trained LM head only (no fine-tuning).

Paper §5 "Zero-shot Behaviors": "we append the token *very* to each example and restrict the
language model's output distribution to only the words *positive* and *negative* and guess the
token it assigns higher probability to as the prediction."

We implement exactly that. As an extra diagnostic (clearly separated in the output) we also
report the same scores with the decision threshold re-centred on the median log-odds, which
shows how much of the error is a constant class bias rather than a lack of signal.

    python src/zero_shot.py            # full SST-2 validation set (872)
    python src/zero_shot.py --quick    # 100 examples
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GlueDataset, TASKS, get_tokenizer  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from load_pretrained import load_pretrained_gpt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def single_token_id(tok, word):
    ids = tok.encode(word)
    assert len(ids) == 1, f"'{word}' is not a single BPE token: {tok.convert_ids_to_tokens(ids)}"
    return ids[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()
    very, pos_id, neg_id = (single_token_id(tok, w) for w in ("very", "positive", "negative"))
    model = load_pretrained_gpt().to(device).eval()
    ds = GlueDataset("sst2", "validation", subset=100 if a.quick else None)

    # sequence = BPE(sentence) + [very]; next-token distribution read at the "very" position.
    seqs = [r["bpe"][0][:510] + [very] for r in ds.rows]
    labels = [r["label"] for r in ds.rows]
    t0 = time.time()
    logodds, p_pos_restricted, top_is_label = [], [], 0
    for i in tqdm(range(0, len(seqs), a.batch), desc="zero-shot SST-2", unit="batch", dynamic_ncols=True):
        chunk = seqs[i:i + a.batch]
        T = max(map(len, chunk))
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
        last = torch.tensor([len(s) - 1 for s in chunk])
        h = model(ids.to(device))
        logits = model.lm_logits(h[torch.arange(len(chunk)), last.to(device)]).float()
        lp = logits.log_softmax(-1)
        d = (lp[:, pos_id] - lp[:, neg_id]).cpu()
        logodds += d.tolist()
        p_pos_restricted += torch.sigmoid(d).tolist()
    elapsed = time.time() - t0
    logodds = np.array(logodds)
    pred = (logodds > 0).astype(int)                      # paper's rule: higher probability wins
    m = compute_metrics(labels, pred)
    thr = float(np.median(logodds))
    m_cal = compute_metrics(labels, (logodds > thr).astype(int))

    run = "E5_zeroshot_sst2" + ("_quick" if a.quick else "")
    res = {"run_name": run, "task": "sst2", "label_names": list(TASKS["sst2"].label_names),
           "setting": {"method": "append 'very'; compare P('positive') vs P('negative') from the LM head",
                       "token_ids": {"very": very, "positive": pos_id, "negative": neg_id},
                       "fine_tuned": False},
           "metrics": m,
           "diagnostic_median_threshold": {"threshold_logodds": thr, "metrics": m_cal,
                                           "note": "NOT the paper's method; threshold uses validation "
                                                   "statistics (label-free) to remove class bias"},
           "frac_pred_positive": float(pred.mean()), "frac_true_positive": float(np.mean(labels)),
           "chance_majority_acc": float(max(np.mean(labels), 1 - np.mean(labels))),
           "compute": {"time_s": elapsed, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
           "quick": a.quick}
    out = os.path.join(ROOT, "results")
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    with open(os.path.join(out, f"{run}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, "preds", f"{run}.json"), "w", encoding="utf-8") as f:
        json.dump([{"idx": r["idx"], "text": r["text"], "label": y, "pred": int(p), "logodds": float(l)}
                   for r, y, p, l in zip(ds.rows, labels, pred, logodds)], f)
    print(f"[{run}] paper rule acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"pred_pos={pred.mean():.2f} | median-threshold diag acc={m_cal['accuracy']:.4f}")


if __name__ == "__main__":
    main()
