"""GPT-1 (Radford et al., 2018) implemented from scratch in PyTorch.

Only torch is used here -- no `transformers` model classes.

Architecture (paper §4.1 "Model specifications"):
  * decoder-only Transformer, 12 layers, 768-d states, 12 heads, FFN inner size 3072
  * learned position embeddings, GELU activation, dropout 0.1 (embedding, attention, residual)
  * post-LayerNorm blocks:  n = LN(x + Attn(x)),  h = LN(n + MLP(n))
  * output projection tied to the token embedding:  P(u) = softmax(h_n We^T)   (Eq. 2)

Fine-tuning head (Eq. 3):  P(y | x^1..x^m) = softmax(h_l^m W_y), where h_l^m is the final
hidden state at the extract token <e>. For similarity tasks the two orderings' h_l^m are
summed element-wise before W_y (paper §3.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 40478      # 40,000 BPE merges + base symbols (HF openai-gpt vocab)
    n_positions: int = 512
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # fused Q, K, V projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)       # output projection W^O
        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)
        mask = torch.tril(torch.ones(cfg.n_positions, cfg.n_positions, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.n_positions, cfg.n_positions),
                             persistent=False)

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if return_attn:
            # explicit path: softmax(QK^T / sqrt(d_k) + M) V, keeps the attention matrix
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(q.dtype)
            probs = att
            att = self.attn_drop(att)
            y = att @ v
        else:
            # fused kernel, mathematically identical (causal mask, 1/sqrt(d_k) scaling)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_drop.p if self.training else 0.0)
            probs = None
        y = y.transpose(1, 2).contiguous().view(B, T, C)     # concat heads
        y = self.resid_drop(self.c_proj(y))
        return y, probs


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, cfg.n_inner)
        self.c_proj = nn.Linear(cfg.n_inner, cfg.n_embd)
        # tanh approximation of GELU: what the original OpenAI code used, and what HF maps
        # afn="gelu" to for openai-gpt (gelu_new). The exact erf form drifts by ~1e-2 over 12 layers.
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """Post-LN Transformer block (LayerNorm applied after each residual addition)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)

    def forward(self, x, return_attn: bool = False):
        a, probs = self.attn(x, return_attn)
        n = self.ln_1(x + a)
        h = self.ln_2(n + self.mlp(n))
        return h, probs


class GPT(nn.Module):
    """The GPT-1 Transformer decoder (the pre-trained body plus the tied LM head)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)      # W_e
        self.positions_embed = nn.Embedding(cfg.n_positions, cfg.n_embd)  # W_p
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.grad_checkpoint = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Paper: N(0, 0.02) initialisation; zero biases, unit LayerNorm gains."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def reinit_layers_from(self, k: int):
        """Keep embeddings + first k blocks; randomly re-initialise blocks k..n-1 (layer-transfer study)."""
        for blk in self.h[k:]:
            blk.apply(self._init_weights)

    def resize_token_embeddings(self, new_size: int):
        """Add rows for new special tokens, initialised N(0, 0.02) (paper §3.2 / §3.3)."""
        old = self.tokens_embed
        if new_size == old.num_embeddings:
            return
        new = nn.Embedding(new_size, self.cfg.n_embd).to(old.weight.device, old.weight.dtype)
        nn.init.normal_(new.weight, mean=0.0, std=self.cfg.init_std)
        n = min(new_size, old.num_embeddings)
        with torch.no_grad():
            new.weight[:n] = old.weight[:n]
        self.tokens_embed = new
        self.cfg.vocab_size = new_size

    def forward(self, input_ids, return_attn: bool = False):
        """input_ids: (B, T) -> final hidden states h_n: (B, T, 768).

        No padding mask is needed: sequences are right-padded and attention is causal, so real
        tokens never attend to pads that come after them.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.n_positions, f"sequence length {T} > {self.cfg.n_positions}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tokens_embed(input_ids) + self.positions_embed(pos))   # h_0 = U W_e + W_p
        attns = []
        for blk in self.h:
            if self.grad_checkpoint and self.training and not return_attn:
                x, _ = checkpoint(blk, x, use_reentrant=False)
            else:
                x, a = blk(x, return_attn)
                if return_attn:
                    attns.append(a)
        return (x, attns) if return_attn else x

    def lm_logits(self, hidden):
        """Tied output layer: logits = h W_e^T (Eq. 2)."""
        return F.linear(hidden, self.tokens_embed.weight)


class GPTClassifier(nn.Module):
    """GPT body + linear head W_y on the <e> hidden state (Eq. 3).

    Inputs:
      input_ids   (B, T)     for single-sequence tasks, or (B, 2, T) for similarity (both orderings)
      extract_idx (B,)  or (B, 2): position of the <e> token in each sequence
    Returns (class logits, hidden states (B*, T, C)).
    """

    def __init__(self, gpt: GPT, n_classes: int, clf_pdrop: float = 0.1):
        super().__init__()
        self.transformer = gpt
        self.clf_drop = nn.Dropout(clf_pdrop)
        self.clf_head = nn.Linear(gpt.cfg.n_embd, n_classes)       # W_y
        nn.init.normal_(self.clf_head.weight, std=gpt.cfg.init_std)
        nn.init.zeros_(self.clf_head.bias)

    def forward(self, input_ids, extract_idx):
        pair = input_ids.dim() == 3
        if pair:
            B, P, T = input_ids.shape
            input_ids = input_ids.reshape(B * P, T)
            extract_idx = extract_idx.reshape(B * P)
        h = self.transformer(input_ids)                               # (B*, T, C)
        rows = torch.arange(h.size(0), device=h.device)
        h_e = h[rows, extract_idx]                                    # h_l^m at <e>
        if pair:
            h_e = h_e.view(B, P, -1).sum(dim=1)                       # element-wise sum of orderings
        logits = self.clf_head(self.clf_drop(h_e))
        return logits, h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
