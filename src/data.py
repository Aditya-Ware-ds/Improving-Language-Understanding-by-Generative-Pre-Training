"""Datasets and the paper's task-specific input transformations (§3.3, Fig. 1).

  classification :  <s> text <e>
  similarity     :  <s> A <$> B <e>   and   <s> B <$> A <e>   (both orderings, summed later)

Special tokens are appended to the 40,478-entry BPE vocabulary as brand-new ids so they can
never collide with real text (e.g. a literal "$" in a review). Their embeddings are initialised
N(0, 0.02) by GPT.resize_token_embeddings.

Batches are right-padded to the longest sequence *in the batch* (dynamic padding). Because the
model is causal, pads after <e> cannot influence any real position; pads are also excluded from
the auxiliary LM loss via `lm_mask`.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import gpt_source  # noqa: E402

HF_NAME = gpt_source()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache")

SPECIAL_TOKENS = ["<s>", "<$>", "<e>", "<pad>"]   # start, delimiter, extract, pad


@dataclass
class TaskSpec:
    glue_name: str
    fields: tuple
    n_classes: int
    pair: bool
    max_len: int            # chosen from the length analysis (see analyze_lengths / README)
    label_names: tuple
    positive_label: int = 1


TASKS = {
    "sst2": TaskSpec("sst2", ("sentence",), 2, False, 64, ("negative", "positive")),
    "mrpc": TaskSpec("mrpc", ("sentence1", "sentence2"), 2, True, 104,
                     ("not_equivalent", "equivalent")),
}


# --------------------------------------------------------------------------- tokenizer
_TOK = None


def get_tokenizer(verbose: bool = True):
    """OpenAIGPTTokenizer (40k-merge BPE). With ftfy + spaCy installed it reproduces the original
    pre-processing (ftfy text fixing + spaCy word splitting, lower-cased). If either is missing,
    HF silently falls back to BERT's BasicTokenizer without ftfy, which splits punctuation
    slightly differently -- we detect and report that."""
    global _TOK
    if _TOK is None:
        from transformers import OpenAIGPTTokenizer
        _TOK = OpenAIGPTTokenizer.from_pretrained(HF_NAME)
        if verbose:
            print(f"[tokenizer] {tokenizer_mode(_TOK)}")
    return _TOK


def tokenizer_mode(tok=None) -> str:
    tok = tok or get_tokenizer(verbose=False)
    has_ftfy = getattr(tok, "fix_text", None) is not None
    has_spacy = getattr(tok, "nlp", None) is not None and "spacy" in type(tok.nlp).__module__
    if has_ftfy and has_spacy:
        return "ftfy+spacy (matches original GPT pre-processing)"
    return "FALLBACK BasicTokenizer (ftfy/spacy missing: tokenization differs slightly from the paper)"


def special_ids(base_vocab: int):
    """ids for <s>, <$>, <e>, <pad> appended after the base vocabulary."""
    return {name: base_vocab + i for i, name in enumerate(SPECIAL_TOKENS)}


# --------------------------------------------------------------------------- GLUE loading
def load_glue(task: str):
    from datasets import load_dataset
    name = TASKS[task].glue_name
    try:
        return load_dataset("nyu-mll/glue", name)
    except Exception:
        return load_dataset("glue", name)


def _tokenize_split(task: str, split: str):
    """BPE-tokenise a GLUE split once and cache the ids (spaCy tokenisation is slow)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{task}_{split}_bpe.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    spec = TASKS[task]
    ds = load_glue(task)[split]
    tok = get_tokenizer()
    from tqdm.auto import tqdm
    rows = []
    for ex in tqdm(ds, desc=f"tokenising {task}/{split} (cached after first run)", unit="ex",
                   dynamic_ncols=True):
        rows.append({"idx": int(ex["idx"]),
                     "text": [ex[f] for f in spec.fields],
                     "bpe": [tok.encode(ex[f]) for f in spec.fields],
                     "label": int(ex["label"])})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tokenizer_mode": tokenizer_mode(tok), "rows": rows}, f)
    return {"tokenizer_mode": tokenizer_mode(tok), "rows": rows}


# --------------------------------------------------------------------------- transformations
def transform_single(bpe, sp, max_len):
    bpe = bpe[: max_len - 2]
    return [sp["<s>"]] + bpe + [sp["<e>"]]


def _truncate_pair(a, b, budget):
    a, b = list(a), list(b)
    while len(a) + len(b) > budget:          # trim the longer sentence first
        (a if len(a) >= len(b) else b).pop()
    return a, b


def transform_pair(bpe_a, bpe_b, sp, max_len):
    a, b = _truncate_pair(bpe_a, bpe_b, max_len - 3)
    x1 = [sp["<s>"]] + a + [sp["<$>"]] + b + [sp["<e>"]]
    x2 = [sp["<s>"]] + b + [sp["<$>"]] + a + [sp["<e>"]]
    return [x1, x2]


class GlueDataset(Dataset):
    def __init__(self, task, split, max_len=None, subset=None, seed=42, base_vocab=40478):
        spec = TASKS[task]
        self.task, self.spec = task, spec
        self.max_len = max_len or spec.max_len
        self.sp = special_ids(base_vocab)
        data = _tokenize_split(task, split)
        self.tokenizer_mode = data["tokenizer_mode"]
        rows = data["rows"]
        if subset:
            rng = random.Random(seed)
            rows = rng.sample(rows, min(subset, len(rows)))
        self.rows = rows
        self.n_truncated = 0
        self.items = []
        for r in rows:
            if spec.pair:
                seqs = transform_pair(r["bpe"][0], r["bpe"][1], self.sp, self.max_len)
                trunc = len(r["bpe"][0]) + len(r["bpe"][1]) + 3 > self.max_len
            else:
                seqs = transform_single(r["bpe"][0], self.sp, self.max_len)
                trunc = len(r["bpe"][0]) + 2 > self.max_len
            self.n_truncated += int(trunc)
            self.items.append((seqs, r["label"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int, pair: bool):
    """Right-pad to the longest sequence in the batch.

    Returns input_ids (B,T) or (B,2,T), extract_idx (B,) or (B,2), labels (B,),
    lm_mask (same shape as input_ids[..., 1:]) = True where the next-token target is real.
    """
    def collate(batch):
        seqs = [s for s, _ in batch]
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        flat = [x for s in seqs for x in s] if pair else seqs
        T = max(len(x) for x in flat)
        ids = torch.full((len(flat), T), pad_id, dtype=torch.long)
        for i, x in enumerate(flat):
            ids[i, : len(x)] = torch.tensor(x)
        extract = torch.tensor([len(x) - 1 for x in flat], dtype=torch.long)
        lm_mask = ids[:, 1:] != pad_id
        if pair:
            B = len(batch)
            ids, extract, lm_mask = ids.view(B, 2, T), extract.view(B, 2), lm_mask.view(B, 2, T - 1)
        return {"input_ids": ids, "extract_idx": extract, "labels": labels, "lm_mask": lm_mask}
    return collate


# --------------------------------------------------------------------------- length analysis
def analyze_lengths(task: str, out_dir: str | None = None):
    """Distribution of transformed sequence lengths (incl. special tokens) -> choose max_len."""
    spec = TASKS[task]
    stats = {"task": task}
    for split in ("train", "validation"):
        rows = _tokenize_split(task, split)["rows"]
        if spec.pair:
            L = np.array([len(r["bpe"][0]) + len(r["bpe"][1]) + 3 for r in rows])
        else:
            L = np.array([len(r["bpe"][0]) + 2 for r in rows])
        stats[split] = {"n": int(len(L)), "mean": float(L.mean()), "median": float(np.median(L)),
                        "p95": float(np.percentile(L, 95)), "p99": float(np.percentile(L, 99)),
                        "max": int(L.max()),
                        "frac_over_max_len": float((L > spec.max_len).mean()),
                        "hist": np.histogram(L, bins=30)[0].tolist(),
                        "hist_edges": np.histogram(L, bins=30)[1].tolist()}
    stats["chosen_max_len"] = spec.max_len
    stats["tokenizer_mode"] = tokenizer_mode()
    if out_dir:
        with open(os.path.join(out_dir, f"length_stats_{task}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="tokenise GLUE tasks and print length statistics")
    ap.add_argument("--tasks", nargs="+", default=["sst2", "mrpc"])
    a = ap.parse_args()
    for t in a.tasks:
        s = analyze_lengths(t, os.path.join(ROOT, "results"))
        for split in ("train", "validation"):
            d = s[split]
            print(f"{t:5s} {split:10s} n={d['n']:6d} mean={d['mean']:.1f} p95={d['p95']:.0f} "
                  f"p99={d['p99']:.0f} max={d['max']}  >max_len({s['chosen_max_len']}): "
                  f"{100 * d['frac_over_max_len']:.2f}%")
