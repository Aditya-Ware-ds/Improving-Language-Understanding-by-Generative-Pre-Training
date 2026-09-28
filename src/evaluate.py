"""Metrics and plots shared by train.py, baselines.py, zero_shot.py and make_report_assets.py.

CLI:  python src/evaluate.py results/E1_sst2_seed42.json     -> prints metrics, writes plots to results/figures/
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, positive_label: int = 1) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pp, pr, pf, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                    pos_label=positive_label, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "precision_pos": float(pp), "recall_pos": float(pr), "f1_pos": float(pf),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ----------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def plot_confusion(cm, labels, title, path):
    plt = _plt()
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xticks([0, 1], labels, fontsize=7)
    ax.set_yticks([0, 1], labels, fontsize=7, rotation=90, va="center")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2 = "#0b0b0b", "#52514e"


def plot_curves(curves: dict, title, path):
    """curves: {'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_lm_loss'}.
    One y-axis per panel; the aux LM loss (different scale) gets its own panel."""
    plt = _plt()
    x = curves["epoch"]
    has_lm = bool(curves.get("train_lm_loss")) and any(v is not None for v in curves["train_lm_loss"])
    fig, axes = plt.subplots(1, 3 if has_lm else 2, figsize=(9.6 if has_lm else 7.2, 2.8))
    a1, a2 = axes[0], axes[1]
    for ax, key, lab in ((a1, "loss", "classification loss (CE)"), (a2, "acc", "accuracy")):
        ax.plot(x, curves[f"train_{key}"], color=SERIES[0], lw=2, marker="o", ms=3, label="train")
        ax.plot(x, curves[f"val_{key}"], color=SERIES[1], lw=2, marker="o", ms=3, label="validation")
        ax.set_xlabel("epoch", color=INK2)
        ax.set_title(lab, fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False)
    if has_lm:
        a3 = axes[2]
        a3.plot(x, curves["train_lm_loss"], color=SERIES[2], lw=2, marker="o", ms=3)
        a3.set_xlabel("epoch", color=INK2)
        a3.set_title("auxiliary LM loss (train, CE/token)", fontsize=9, color=INK)
    fig.suptitle(title, fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        m = r["metrics"]
        print(os.path.basename(p), {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in m.items()})
        fig_dir = os.path.join(os.path.dirname(p), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(p))[0]
        plot_confusion(m["confusion_matrix"], r.get("label_names", ["0", "1"]), stem,
                       os.path.join(fig_dir, f"cm_{stem}.png"))
        if r.get("curves"):
            plot_curves(r["curves"], stem, os.path.join(fig_dir, f"curves_{stem}.png"))
