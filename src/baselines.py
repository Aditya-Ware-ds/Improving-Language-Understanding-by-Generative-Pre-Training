"""E4: TF-IDF + Logistic Regression baseline (classical, no pre-training).

SST-2 : word 1-2-gram TF-IDF of the sentence (+ char 2-5-grams) -> LR.
MRPC  : TF-IDF vectors a, b of both sentences; features [|a-b|, a*b, cosine(a,b), length diff]
        (symmetric, mirroring GPT's order-invariant similarity transformation) -> LR.
The regularisation strength C is chosen by 5-fold CV on the training set only.

    python src/baselines.py --task sst2
    python src/baselines.py --task mrpc --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TASKS, load_glue  # noqa: E402
from evaluate import compute_metrics  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _vec():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=200_000)),
    ])


def _pair_features(vec, s1, s2):
    A, B = vec.transform(s1), vec.transform(s2)
    cos = np.asarray(A.multiply(B).sum(1))            # rows are L2-normalised per sub-vectoriser
    ld = np.abs(np.array([len(x.split()) for x in s1]) - np.array([len(x.split()) for x in s2]))[:, None]
    return sp.hstack([abs(A - B), A.multiply(B), sp.csr_matrix(cos), sp.csr_matrix(ld / 10.0)]).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="sst2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    spec = TASKS[a.task]
    ds = load_glue(a.task)
    tr, va = ds["train"], ds["validation"]
    if a.quick:
        tr = tr.shuffle(seed=a.seed).select(range(100))
        va = va.select(range(100))
    t0 = time.time()
    print(f"[E4 {a.task}] building TF-IDF features for {len(tr)} train / {len(va)} validation examples ...", flush=True)
    vec = _vec()
    if spec.pair:
        vec.fit(list(tr["sentence1"]) + list(tr["sentence2"]))
        Xtr = _pair_features(vec, tr["sentence1"], tr["sentence2"])
        Xva = _pair_features(vec, va["sentence1"], va["sentence2"])
    else:
        Xtr = vec.fit_transform(tr["sentence"])
        Xva = vec.transform(va["sentence"])
    ytr, yva = np.array(tr["label"]), np.array(va["label"])
    print(f"[E4 {a.task}] {Xtr.shape[1]:,} features; grid-searching C with cross-validation (progress below)", flush=True)
    grid = GridSearchCV(LogisticRegression(max_iter=3000, solver="liblinear"),
                        {"C": [0.25, 1, 4, 16]}, cv=3 if a.quick else 5, scoring="accuracy", n_jobs=-1, verbose=2)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xva)
    prob = grid.predict_proba(Xva)[:, 1]
    elapsed = time.time() - t0
    m = compute_metrics(yva, pred, spec.positive_label)
    run = f"E4_tfidf_lr_{a.task}" + ("_quick" if a.quick else "")
    res = {"run_name": run, "task": a.task, "label_names": list(spec.label_names),
           "setting": {"model": "TF-IDF (word 1-2g + char 2-5g) + LogisticRegression",
                       "best_C": grid.best_params_["C"], "cv_accuracy": grid.best_score_,
                       "n_features": int(Xtr.shape[1])},
           "data": {"n_train": len(ytr), "n_val": len(yva)},
           "metrics": m, "compute": {"train_time_s": elapsed, "device": "cpu"}, "quick": a.quick}
    out = os.path.join(ROOT, "results")
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    with open(os.path.join(out, f"{run}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    fields = spec.fields
    preds = [{"idx": int(va[i]["idx"]), "text": [va[i][k] for k in fields], "label": int(yva[i]),
              "pred": int(pred[i]), "prob_pos": float(prob[i])} for i in range(len(yva))]
    with open(os.path.join(out, "preds", f"{run}.json"), "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"[{run}] C={grid.best_params_['C']} acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"f1_pos={m['f1_pos']:.4f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
