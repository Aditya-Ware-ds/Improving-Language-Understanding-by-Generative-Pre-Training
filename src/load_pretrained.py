"""Load the released GPT-1 weights (Hugging Face `openai-community/openai-gpt`) into src/model.py.

HF stores projections as `Conv1D` modules whose weight has shape (in_features, out_features),
i.e. the transpose of `nn.Linear.weight` (out_features, in_features). We transpose on load.

Run as a script to verify the port:
    python src/load_pretrained.py --test
which feeds identical token ids to our model and to HF's OpenAIGPTModel and asserts
max |h_ours - h_hf| < 1e-4 on the final hidden states.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, count_parameters  # noqa: E402

from paths import gpt_source  # noqa: E402

HF_NAME = gpt_source()        # local weights/openai-gpt/ if downloaded, else the Hub id
CONV1D_KEYS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def _hf_state_dict(name: str = HF_NAME):
    """Fetch the raw pretrained tensors (uses transformers only as a downloader)."""
    from transformers import OpenAIGPTModel
    hf = OpenAIGPTModel.from_pretrained(name)
    return hf.state_dict(), hf.config


def convert_state_dict(hf_sd: dict) -> dict:
    out = {}
    for k, v in hf_sd.items():
        if re.fullmatch(r"h\.\d+\.attn\.(bias|masked_bias)", k):
            continue                                  # HF causal-mask buffers; ours is built in
        if any(k.endswith(s) for s in CONV1D_KEYS):
            v = v.t().contiguous()                    # Conv1D (in, out) -> Linear (out, in)
        out[k] = v
    return out


def load_pretrained_gpt(name: str = HF_NAME) -> GPT:
    hf_sd, hf_cfg = _hf_state_dict(name)
    cfg = GPTConfig(vocab_size=hf_cfg.vocab_size, n_positions=hf_cfg.n_positions,
                    n_embd=hf_cfg.n_embd, n_layer=hf_cfg.n_layer, n_head=hf_cfg.n_head,
                    n_inner=4 * hf_cfg.n_embd, layer_norm_eps=hf_cfg.layer_norm_epsilon)
    model = GPT(cfg)
    model.load_state_dict(convert_state_dict(hf_sd), strict=True)
    return model


@torch.no_grad()
def verify(name: str = HF_NAME, tol: float = 1e-4, out_path: str | None = None) -> dict:
    from transformers import OpenAIGPTModel, OpenAIGPTTokenizer
    torch.manual_seed(0)
    print("[weight-match] loading pretrained weights into src/model.py (first run downloads ~480 MB) ...", flush=True)
    ours = load_pretrained_gpt(name).eval()
    print("[weight-match] loading HF reference models and comparing hidden states ...", flush=True)
    hf = OpenAIGPTModel.from_pretrained(name).eval()
    tok = OpenAIGPTTokenizer.from_pretrained(name)

    texts = ["the movie was surprisingly good , and the acting was superb .",
             "a dull , lifeless and utterly forgettable film ."]
    report = {"hf_model": name, "cases": []}
    for t in texts:
        ids = torch.tensor([tok.encode(t)])
        h_ours = ours(ids)
        h_hf = hf(input_ids=ids).last_hidden_state
        diff = (h_ours - h_hf).abs().max().item()
        report["cases"].append({"text": t, "n_tokens": ids.shape[1], "max_abs_diff": diff})
    # random ids over the full 512 context as a stress test
    ids = torch.randint(0, ours.cfg.vocab_size, (2, 512))
    diff = (ours(ids) - hf(input_ids=ids).last_hidden_state).abs().max().item()
    report["cases"].append({"text": "random ids, batch 2 x 512", "n_tokens": 512, "max_abs_diff": diff})
    # tied LM head: our h We^T vs HF's OpenAIGPTLMHeadModel logits
    from transformers import OpenAIGPTLMHeadModel
    hf_lm = OpenAIGPTLMHeadModel.from_pretrained(name).eval()
    ids = torch.tensor([tok.encode(texts[0])])
    lm_diff = (ours.lm_logits(ours(ids)) - hf_lm(input_ids=ids).logits).abs().max().item()
    report["cases"].append({"text": "LM-head logits (tied W_e)", "n_tokens": ids.shape[1],
                            "max_abs_diff": lm_diff})

    report["max_abs_diff"] = max(c["max_abs_diff"] for c in report["cases"])
    report["tolerance"] = tol
    report["passed"] = report["max_abs_diff"] < tol
    report["n_parameters"] = count_parameters(ours)
    report["n_parameters_hf"] = sum(p.numel() for p in hf.parameters())
    report["config"] = ours.cfg.to_dict()
    for c in report["cases"]:
        print(f"  {c['text'][:45]:45s}  T={c['n_tokens']:3d}  max|diff|={c['max_abs_diff']:.3e}")
    print(f"params ours={report['n_parameters']:,}  hf={report['n_parameters_hf']:,}")
    print("PASSED" if report["passed"] else "FAILED", f"(tol {tol})")
    if out_path:   # only a passing test produces weight_match.json (run_all.py uses it as "done" marker)
        if not report["passed"]:
            out_path = out_path.replace(".json", "_FAILED.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    assert report["passed"], f"weight-match test failed: {report['max_abs_diff']}"
    return report


def test_weight_match():   # pytest entry point
    verify()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "weight_match.json"))
    a = ap.parse_args()
    if a.test:
        verify(out_path=a.out)
    else:
        m = load_pretrained_gpt()
        print(f"loaded GPT-1 with {count_parameters(m):,} parameters")
