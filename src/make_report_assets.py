"""Read results/*.json and write every table (results/tables/*.md) and figure (results/figures/*.png)
used in the report. No number in the report is typed by hand -- they all come from here.

    python src/make_report_assets.py                 # tables + figures from existing results
    python src/make_report_assets.py --attention     # also render the attention map (needs a checkpoint)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import INK, INK2, SERIES, _plt, plot_confusion, plot_curves  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")

# Paper numbers, transcribed from the PDF (Table 4 and Table 5; GLUE benchmark evaluation).
PAPER = {
    "sst2_acc": {"full (w/ aux LM)": 91.3, "w/o aux LM": 92.0, "w/o pre-training": 84.0, "LSTM w/ aux LM": 90.5},
    "mrpc_f1": {"full (w/ aux LM)": 82.3, "w/o aux LM": 84.9, "w/o pre-training": 79.4, "LSTM w/ aux LM": 83.2},
}


def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if not r.get("quick"):
            out.append(r)
    return out


def pct(x):
    return f"{100 * x:.2f}"


def mean_std(vals):
    vals = np.asarray(vals, dtype=float) * 100
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"


def write(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"  wrote tables/{name}")


# ----------------------------------------------------------------------------- tables
def exp_runs(exp, task):
    return load(f"{exp}_{task}_seed*.json")


def main_table(task):
    rows = [("E1", "GPT-1 pre-trained, fine-tuned, λ = 0.5", exp_runs("E1", task)),
            ("E2", "GPT-1 pre-trained, fine-tuned, λ = 0 (no aux LM)", exp_runs("E2", task)),
            ("E3", "Same Transformer, random init (no pre-training)", exp_runs("E3", task)),
            ("E4", "TF-IDF + Logistic Regression", load(f"E4_tfidf_lr_{task}.json"))]
    if task == "sst2":
        rows.append(("E5", "Zero-shot LM heuristic (no fine-tuning)", load("E5_zeroshot_sst2.json")))
    head = ("| Exp | Model | Seeds | Accuracy (%) | Precision macro (%) | Recall macro (%) | "
            "F1 macro (%) | F1 positive class (%) |\n|---|---|---|---|---|---|---|---|\n")
    body = ""
    for tag, name, runs in rows:
        if not runs:
            body += f"| {tag} | {name} | – | not run | | | | |\n"
            continue
        ms = [r["metrics"] for r in runs]
        seeds = ",".join(str(r["hyperparams"]["seed"]) for r in runs) if "hyperparams" in runs[0] else "–"
        body += (f"| {tag} | {name} | {seeds} | {mean_std([m['accuracy'] for m in ms])} | "
                 f"{mean_std([m['precision_macro'] for m in ms])} | {mean_std([m['recall_macro'] for m in ms])} | "
                 f"{mean_std([m['f1_macro'] for m in ms])} | {mean_std([m['f1_pos'] for m in ms])} |\n")
    n_val = next((r[2][0]["metrics"]["n"] for r in rows if r[2]), None)
    note = (f"\n*{task.upper()} GLUE validation split (n = {n_val}). Mean ± sample std over seeds where "
            f"more than one seed was run.*\n")
    write(f"main_{task}.md", head + body + note)
    return rows


def compute_table():
    runs = load("E[1236]_*.json")
    head = ("| Run | Task | Micro-batch × accum (= effective) | Max len | Train time (min) | "
            "Peak VRAM allocated (MiB) | OOM retries |\n|---|---|---|---|---|---|---|\n")
    body = ""
    for r in runs:
        h, c = r["hyperparams"], r["compute"]
        body += (f"| {r['run_name']} | {r['task']} | {h['micro_batch']} × {h['grad_accum']} (= {h['effective_batch']}) | "
                 f"{h['max_len']} | {c['train_time_s'] / 60:.1f} | {c['peak_vram_mib']:.0f} | {len(r['oom_events'])} |\n")
    for r in load("E4_*.json"):
        body += f"| {r['run_name']} | {r['task']} | – (CPU) | – | {r['compute']['train_time_s'] / 60:.1f} | – | – |\n"
    for r in load("E5_*.json"):
        body += f"| {r['run_name']} | sst2 | – (inference only) | – | {r['compute']['time_s'] / 60:.1f} | – | – |\n"
    if runs:
        c = runs[0]["compute"]
        body += (f"\n*Hardware: {c['gpu']}; torch {c['torch']}, transformers {c.get('transformers')}; "
                 f"{c['platform']}. Classifier parameter count: {c['n_parameters']:,}.*\n")
    write("compute.md", head + body)


def layer_table():
    pts = []
    for r in load("E6_mrpc_k*_seed*.json"):
        pts.append((r["setting"]["transfer_layers"], r))
    for r in load("E1_mrpc_seed42.json"):
        pts.append((12, r))
    pts.sort(key=lambda t: t[0])
    body = "| Pre-trained layers transferred (k) | Accuracy (%) | F1 positive (%) | F1 macro (%) |\n|---|---|---|---|\n"
    for k, r in pts:
        m = r["metrics"]
        body += f"| {k} | {pct(m['accuracy'])} | {pct(m['f1_pos'])} | {pct(m['f1_macro'])} |\n"
    none = load("E3_mrpc_seed42.json")
    if none:
        m = none[0]["metrics"]
        body += f"| none (random embeddings too; = E3) | {pct(m['accuracy'])} | {pct(m['f1_pos'])} | {pct(m['f1_macro'])} |\n"
    body += ("\n*MRPC validation, seed 42, λ = 0.5. k = 0 transfers only the token/position embeddings; "
             "k = 12 is the full model (E1, seed 42).*\n")
    write("layer_transfer_mrpc.md", body)
    return pts, none


def paper_table():
    def ours(exp, task, key):
        rs = exp_runs(exp, task)
        return mean_std([r["metrics"][key] for r in rs]) if rs else "not run"
    body = ("| Setting | Paper SST-2 acc (GLUE test) | Ours SST-2 acc (dev) | Paper MRPC F1 (GLUE test) | "
            "Ours MRPC F1 (dev) |\n|---|---|---|---|---|\n")
    for label, exp in (("full (w/ aux LM)", "E1"), ("w/o aux LM", "E2"), ("w/o pre-training", "E3")):
        body += (f"| {label} | {PAPER['sst2_acc'][label]} | {ours(exp, 'sst2', 'accuracy')} | "
                 f"{PAPER['mrpc_f1'][label]} | {ours(exp, 'mrpc', 'f1_pos')} |\n")
    body += f"| LSTM w/ aux LM | {PAPER['sst2_acc']['LSTM w/ aux LM']} | not run | {PAPER['mrpc_f1']['LSTM w/ aux LM']} | not run |\n"
    body += ("\n*Paper values: Radford et al. (2018), Table 4 and Table 5 (GLUE benchmark). Ours: GLUE "
             "validation split. The two are not directly comparable (different split, and see caveats).*\n")
    write("paper_comparison.md", body)


NEGATION = {"not", "n't", "no", "never", "nothing", "none", "nor", "neither", "without", "hardly", "barely"}
CONTRAST = {"but", "though", "although", "yet", "however", "despite", "while", "even", "still"}


CAUSE = {"negation": "negation word: polarity flip or scope misjudged",
         "contrast": "contrast connective: clauses of opposite polarity",
         "very short": "very little context",
         "long": "long sentence with mixed content",
         "rhetorical/punct.": "rhetorical question / exclamation, possibly ironic"}


def likely_cause(sentence: str) -> str:
    c = [k for k in cues(sentence).split(", ") if k in CAUSE]
    return "; ".join(CAUSE[k] for k in c) if c else "no surface cue: irony, world knowledge or debatable gold label"


def cues(sentence: str) -> str:
    """Heuristic linguistic cues for error analysis (automatic, not a human judgement)."""
    toks = sentence.lower().replace("n't", " n't").split()
    c = []
    if NEGATION & set(toks):
        c.append("negation")
    if CONTRAST & set(toks):
        c.append("contrast")
    if len(toks) <= 6:
        c.append("very short")
    if len(toks) >= 35:
        c.append("long")
    if any(t in toks for t in ("?", "!")) or "..." in sentence:
        c.append("rhetorical/punct.")
    return ", ".join(c) or "–"


def misclassified_table(n=15):
    p = os.path.join(RES, "preds", "E1_sst2_seed42.json")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        preds = json.load(f)
    wrong = [x for x in preds if x["label"] != x["pred"]]
    conf = lambda x: x["prob_pos"] if x["pred"] == 1 else 1 - x["prob_pos"]
    wrong.sort(key=conf, reverse=True)
    names = ["negative", "positive"]
    body = ("| # | GLUE idx | Sentence | True | Pred | Confidence | Likely cause (automatic heuristic) |\n"
            "|---|---|---|---|---|---|---|\n")
    for i, x in enumerate(wrong[:n], 1):
        s = x["text"][0].replace("|", "\\|").strip()
        x["cues"] = cues(x["text"][0])
        body += (f"| {i} | {x['idx']} | {s} | {names[x['label']]} | {names[x['pred']]} | {conf(x):.3f} | "
                 f"{likely_cause(x['text'][0])} |\n")
    # cue frequencies: all errors vs all correct predictions
    right = [x for x in preds if x["label"] == x["pred"]]
    freq = lambda xs, c: sum(c in cues(x["text"][0]) for x in xs) / max(1, len(xs))
    cue_stats = {c: {"errors": freq(wrong, c), "correct": freq(right, c)}
                 for c in ("negation", "contrast", "very short", "long")}
    with open(os.path.join(TAB, "error_cues_sst2.json"), "w", encoding="utf-8") as f:
        json.dump({"n_errors": len(wrong), "n_correct": len(right), "cue_rates": cue_stats}, f, indent=2)
    body += (f"\n*The {n} most confident errors of E1 (seed 42) out of {len(wrong)} errors on "
             f"{len(preds)} SST-2 validation sentences.*\n")
    write("misclassified_sst2.md", body)
    with open(os.path.join(TAB, "misclassified_sst2.json"), "w", encoding="utf-8") as f:
        json.dump(wrong[:n], f, indent=2)


def zero_shot_table():
    r = load("E5_zeroshot_sst2.json")
    if not r:
        return
    r = r[0]
    m, d = r["metrics"], r["diagnostic_median_threshold"]["metrics"]
    body = ("| Decision rule | Accuracy (%) | F1 macro (%) | Predicted positive (%) |\n|---|---|---|---|\n"
            f"| Paper rule: P(positive) > P(negative) after appending “very” | {pct(m['accuracy'])} | "
            f"{pct(m['f1_macro'])} | {pct(r['frac_pred_positive'])} |\n"
            f"| Diagnostic: threshold at median log-odds (not the paper's method) | {pct(d['accuracy'])} | "
            f"{pct(d['f1_macro'])} | 50.00 |\n"
            f"| Majority-class guess | {pct(r['chance_majority_acc'])} | – | – |\n")
    body += f"\n*SST-2 validation, n = {m['n']}; true positive rate {pct(r['frac_true_positive'])}%.*\n"
    write("zero_shot_sst2.md", body)


def length_table():
    body = "| Task | Split | n | Mean | Median | 95th pct | 99th pct | Max | Chosen max_len | % truncated |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for t in ("sst2", "mrpc"):
        p = os.path.join(RES, f"length_stats_{t}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        for sp in ("train", "validation"):
            d = s[sp]
            body += (f"| {t.upper()} | {sp} | {d['n']} | {d['mean']:.1f} | {d['median']:.0f} | {d['p95']:.0f} | "
                     f"{d['p99']:.0f} | {d['max']} | {s['chosen_max_len']} | {100 * d['frac_over_max_len']:.2f} |\n")
    body += "\n*Lengths in BPE tokens after the input transformation (including ⟨s⟩, ⟨$⟩, ⟨e⟩).*\n"
    write("lengths.md", body)


# ----------------------------------------------------------------------------- figures
def fig_comparison(sst_rows, mrpc_rows):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for ax, rows, key, title, paper_key in ((axes[0], sst_rows, "accuracy", "SST-2 accuracy (%)", "sst2_acc"),
                                            (axes[1], mrpc_rows, "f1_pos", "MRPC F1, positive class (%)", "mrpc_f1")):
        labels, vals, errs = [], [], []
        for tag, name, runs in rows:
            if not runs:
                continue
            v = np.array([r["metrics"][key] for r in runs]) * 100
            labels.append(tag)
            vals.append(v.mean())
            errs.append(v.std(ddof=1) if len(v) > 1 else 0)
        y = np.arange(len(labels))
        ax.barh(y, vals, xerr=errs, color=SERIES[0], height=0.6, error_kw={"lw": 1, "ecolor": INK2})
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
        ax.axvline(PAPER[paper_key]["full (w/ aux LM)"], color=SERIES[1], lw=1.5, ls="--")
        ax.text(PAPER[paper_key]["full (w/ aux LM)"], len(labels) - 0.4, " paper (test)",
                color=INK2, fontsize=7, va="bottom")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.02, "E1 λ=0.5 · E2 λ=0 · E3 no pre-training · E4 TF-IDF+LR · E5 zero-shot   "
             "(error bars: std over seeds)", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_layers(pts, none):
    if not pts:
        return
    plt = _plt()
    ks = [k for k, _ in pts]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for key, lab, c in (("accuracy", "accuracy", SERIES[0]), ("f1_pos", "F1 (positive)", SERIES[1])):
        v = [100 * r["metrics"][key] for _, r in pts]
        ax.plot(ks, v, marker="o", ms=5, lw=2, color=c, label=lab)
        ax.text(ks[-1] + 0.3, v[-1], lab, color=INK2, fontsize=7, va="center")
    if none:
        m = none[0]["metrics"]
        ax.axhline(100 * m["accuracy"], color=INK2, lw=1, ls=":")
        ax.text(0, 100 * m["accuracy"] + 0.4, "no pre-training at all (E3) accuracy", fontsize=7, color=INK2)
    ax.set_xticks(ks)
    ax.set_xlabel("number of pre-trained Transformer blocks transferred (k)")
    ax.set_ylabel("MRPC validation score (%)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlim(-0.5, 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "layer_transfer_mrpc.png"))
    plt.close(fig)


def fig_lengths():
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.6))
    for ax, t in zip(axes, ("sst2", "mrpc")):
        p = os.path.join(RES, f"length_stats_{t}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        d = s["train"]
        e = np.array(d["hist_edges"])
        ax.bar(e[:-1], d["hist"], width=np.diff(e), align="edge", color=SERIES[0], edgecolor="white", lw=0.5)
        ax.axvline(s["chosen_max_len"], color=SERIES[1], lw=1.5, ls="--")
        ax.text(s["chosen_max_len"], max(d["hist"]) * 0.9, f" max_len = {s['chosen_max_len']}", fontsize=7, color=INK2)
        ax.set_title(f"{t.upper()} train: transformed length (BPE tokens)", fontsize=9, color=INK)
        ax.set_ylabel("examples")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lengths.png"))
    plt.close(fig)


def fig_zero_shot():
    p = os.path.join(RES, "preds", "E5_zeroshot_sst2.json")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        z = json.load(f)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    lo = np.array([x["logodds"] for x in z])
    y = np.array([x["label"] for x in z])
    bins = np.linspace(lo.min(), lo.max(), 35)
    ax.hist(lo[y == 0], bins=bins, color=SERIES[1], alpha=0.75, label="true negative")
    ax.hist(lo[y == 1], bins=bins, color=SERIES[0], alpha=0.75, label="true positive")
    ax.axvline(0, color=INK, lw=1)
    ax.text(0, ax.get_ylim()[1] * 0.92, " paper's decision boundary", fontsize=7, color=INK2)
    ax.set_xlabel("log P(positive | x, very) − log P(negative | x, very)")
    ax.set_ylabel("sentences")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "zero_shot_logodds.png"))
    plt.close(fig)


def fig_architecture():
    """Block diagram of GPT-1 (left) and the two input transformations used here (right)."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.0, 6.6))
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.grid(False)
    FILL = {"emb": "#dbe8f8", "attn": "#fde3d8", "ln": "#eeeeec", "ffn": "#d7f1e6", "head": "#fbeccc"}

    def box(x, y, w, h, text, kind, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=FILL[kind], ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK2))

    cx, w = 0.8, 6.2
    box(cx, 0.3, w, 0.7, "token ids  U = (u₁ … u_T),  T ≤ 512", "ln", 7.5)
    box(cx, 1.4, w, 1.0, "h₀ = U·W_e + W_p\ntoken emb. (40478+4)×768  +  learned position emb. 512×768", "emb", 6)
    arrow(cx + w / 2, 1.0, cx + w / 2, 1.4)
    # transformer block (x12)
    ax.add_patch(FancyBboxPatch((cx - 0.3, 2.8), w + 0.6, 5.6, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec=INK2, lw=1, ls="--"))
    ax.text(cx + w + 0.4, 5.6, "× 12", fontsize=11, color=INK, va="center")
    box(cx, 3.1, w, 1.0, "masked multi-head self-attention\n12 heads × 64 d, causal mask", "attn")
    box(cx, 4.5, w, 0.6, "add & LayerNorm:  n = LN(x + Attn(x))", "ln", 7.5)
    box(cx, 5.5, w, 1.0, "position-wise feed-forward\n768 → 3072 → 768, GELU", "ffn")
    box(cx, 6.9, w, 0.6, "add & LayerNorm:  h = LN(n + FFN(n))", "ln", 7.5)
    arrow(cx + w / 2, 2.3, cx + w / 2, 3.1)
    arrow(cx + w / 2, 4.1, cx + w / 2, 4.5)
    arrow(cx + w / 2, 5.1, cx + w / 2, 5.5)
    arrow(cx + w / 2, 6.5, cx + w / 2, 6.9)
    arrow(cx + w / 2, 7.5, cx + w / 2, 8.2)
    arrow(cx + w / 2, 8.2, cx + 1.5, 8.9)
    arrow(cx + w / 2, 8.2, cx + w - 1.5, 8.9)
    ax.text(cx + w / 2 + 0.15, 7.75, "h_n", fontsize=8, color=INK2)
    box(cx, 8.9, 3.0, 1.1, "text prediction\nsoftmax(h_n W_eᵀ)\n(tied weights, loss L₁)", "head", 7)
    box(cx + 3.2, 8.9, 3.0, 1.1, "task classifier\nsoftmax(h_n^m W_y)\n(state at ⟨e⟩, loss L₂)", "head", 7)
    ax.text(cx + w / 2, 10.5, "fine-tuning loss  L₃ = L₂ + λ·L₁,  λ = 0.5", ha="center", fontsize=8, color=INK)
    ax.text(cx + w / 2, 12.3, "GPT-1 (decoder-only, post-LN)", ha="center", fontsize=10, color=INK, weight="bold")

    # input transformations
    ax.text(14.0, 12.3, "Task-specific input transformations", ha="center", fontsize=10, color=INK, weight="bold")

    def seq(y, parts):
        x = 8.6
        for t, kind in parts:
            ww = 0.55 + 0.13 * len(t)
            box(x, y, ww, 0.6, t, kind, 7)
            x += ww + 0.08
        return x

    ax.text(8.6, 11.3, "Classification (SST-2 sentiment)", fontsize=8, color=INK)
    xe = seq(10.5, [("⟨s⟩", "ln"), ("review text", "emb"), ("⟨e⟩", "ln")])
    box(xe + 0.4, 10.5, 2.6, 0.6, "Transformer → W_y", "head", 7)
    arrow(xe, 10.8, xe + 0.4, 10.8)

    ax.text(8.6, 9.3, "Similarity (MRPC paraphrase)", fontsize=8, color=INK)
    x1 = seq(8.5, [("⟨s⟩", "ln"), ("sentence 1", "emb"), ("⟨$⟩", "ln"), ("sentence 2", "emb"), ("⟨e⟩", "ln")])
    seq(7.6, [("⟨s⟩", "ln"), ("sentence 2", "emb"), ("⟨$⟩", "ln"), ("sentence 1", "emb"), ("⟨e⟩", "ln")])
    box(x1 + 0.35, 8.5, 1.9, 0.6, "Transformer", "attn", 7)
    box(x1 + 0.35, 7.6, 1.9, 0.6, "Transformer", "attn", 7)
    arrow(x1, 8.8, x1 + 0.35, 8.8)
    arrow(x1, 7.9, x1 + 0.35, 7.9)
    ax.text(x1 + 2.6, 8.35, "⊕", fontsize=14, color=INK, ha="center", va="center")
    box(x1 + 2.95, 8.05, 0.9, 0.6, "W_y", "head", 7)

    ax.text(8.6, 6.2, "Entailment (not run here)", fontsize=8, color=INK2)
    seq(5.4, [("⟨s⟩", "ln"), ("premise", "emb"), ("⟨$⟩", "ln"), ("hypothesis", "emb"), ("⟨e⟩", "ln")])
    ax.text(8.6, 4.4, "Multiple choice (not run here): one sequence per answer, softmax over answers", fontsize=8, color=INK2)
    seq(3.6, [("⟨s⟩", "ln"), ("context", "emb"), ("⟨$⟩", "ln"), ("answer k", "emb"), ("⟨e⟩", "ln")])
    ax.text(8.6, 2.4, "⟨s⟩, ⟨$⟩, ⟨e⟩ are new tokens (embeddings init N(0, 0.02)); the classifier reads\n"
            "the final hidden state at ⟨e⟩. ⊕ = element-wise sum of both orderings.", fontsize=7, color=INK2)
    fig.savefig(os.path.join(FIG, "architecture.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_attention(sentence="the acting is superb , but the plot is painfully thin and predictable ."):
    """Attention of the fine-tuned SST-2 model (E1): last layer, head-average, plus one head."""
    import torch
    from data import get_tokenizer, special_ids, transform_single
    from predict import load_classifier
    ck = os.path.join(ROOT, "checkpoints", "E1_sst2_seed42.pt")
    if not os.path.exists(ck):
        print("  (attention map skipped: no checkpoint)")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_classifier(ck, device)
    tok = get_tokenizer(verbose=False)
    ids = transform_single(tok.encode(sentence), special_ids(40478), 64)
    with torch.no_grad():
        _, attns = model.transformer(torch.tensor([ids], device=device), return_attn=True)
    labels = ["⟨s⟩"] + [t.replace("</w>", "") for t in tok.convert_ids_to_tokens(ids[1:-1])] + ["⟨e⟩"]
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    A = attns[-1][0].float().mean(0).cpu().numpy()
    axes[0].imshow(A, cmap="Blues", vmin=0)
    axes[0].set_title("last layer (12), mean over 12 heads", fontsize=9)
    # the row of <e> in every layer: what the classifier token looks at
    E = np.stack([a[0].float().mean(0)[-1].cpu().numpy() for a in attns])
    axes[1].imshow(E, cmap="Blues", vmin=0, aspect="auto")
    axes[1].set_title("attention from ⟨e⟩ (classifier token), per layer", fontsize=9)
    axes[1].set_yticks(range(12), [f"L{i + 1}" for i in range(12)], fontsize=7)
    for ax in axes:
        ax.grid(False)
        ax.set_xticks(range(len(labels)), labels, rotation=70, fontsize=7)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[0].set_xlabel("attended-to (key)")
    axes[0].set_ylabel("attending (query)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "attention_sst2.png"))
    plt.close(fig)
    with open(os.path.join(TAB, "attention_sentence.json"), "w", encoding="utf-8") as f:
        json.dump({"sentence": sentence, "tokens": labels,
                   "e_row_last_layer": A[-1].tolist(), "e_row_per_layer": E.tolist()}, f)
    print("  wrote figures/attention_sst2.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", action="store_true")
    a = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    sst = main_table("sst2")
    mrpc = main_table("mrpc")
    compute_table()
    pts, none = layer_table()
    paper_table()
    misclassified_table()
    zero_shot_table()
    length_table()
    fig_comparison(sst, mrpc)
    fig_layers(pts, none)
    fig_lengths()
    fig_zero_shot()
    fig_architecture()
    for r in load("E[123]_*_seed42.json"):
        plot_curves(r["curves"], f"{r['run_name']} (λ = {r['setting']['lambda']:g}, "
                    f"{'pre-trained' if r['setting']['pretrained'] else 'random init'})",
                    os.path.join(FIG, f"curves_{r['run_name']}.png"))
    for r in load("E1_*_seed42.json") + load("E4_*.json") + load("E5_*.json"):
        plot_confusion(r["metrics"]["confusion_matrix"], r["label_names"], r["run_name"],
                       os.path.join(FIG, f"cm_{r['run_name']}.png"))
    if a.attention:
        fig_attention()
    print("done")


if __name__ == "__main__":
    main()
