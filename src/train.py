"""Fine-tune GPT-1 on a GLUE task with L3 = L2 + lambda * L1 (paper Eq. 5).

Paper defaults (§4.1): lr 6.25e-5, batch 32, 3 epochs, linear decay with 0.2% warm-up,
lambda 0.5, classifier dropout 0.1, decoupled L2 (w = 0.01) on non-bias/gain weights.

Memory-saving measures for a 4 GB GPU (they do not change the maths):
  * effective batch 32 = micro-batch x gradient-accumulation steps
  * fp16 autocast + GradScaler
  * dynamic padding to the longest sequence in each micro-batch
  * aux LM loss: gather non-pad positions BEFORE the 40k-wide vocabulary projection
  * optional gradient checkpointing (--grad-checkpoint)
  * on CUDA OOM: halve micro-batch, double accumulation, restart the run (logged)

Examples:
  python src/train.py --task sst2 --run-name E1_sst2 --save-model
  python src/train.py --task mrpc --lambda 0 --run-name E2_mrpc --seed 43
  python src/train.py --task mrpc --transfer-layers 6 --run-name E6_mrpc_k6
  python src/train.py --task sst2 --quick                     # 100 examples, 1 epoch smoke test
  python src/train.py --task sst2 --time-steps 50             # timing test only
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TASKS, GlueDataset, SPECIAL_TOKENS, make_collate, special_ids  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from model import GPT, GPTClassifier, GPTConfig, count_parameters  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_VOCAB = 40478


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), default="sst2")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5, help="aux LM weight (paper 0.5)")
    ap.add_argument("--lr", type=float, default=6.25e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32, help="effective batch size (paper 32)")
    ap.add_argument("--micro-batch", type=int, default=8, help="per-step batch; accum = batch/micro")
    ap.add_argument("--max-len", type=int, default=None, help="default: per-task value from data.py")
    ap.add_argument("--warmup", type=float, default=0.002, help="fraction of steps (paper 0.2%%)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clf-dropout", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="gradient clipping (not specified in the paper; 0 disables)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=None, help="use N random training examples")
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--no-pretrain", action="store_true", help="random init (no generative pre-training)")
    ap.add_argument("--transfer-layers", type=int, default=None,
                    help="keep embeddings + first k pretrained blocks, re-init the rest")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--evals-per-epoch", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="smoke test: 100 train/100 val, 1 epoch")
    ap.add_argument("--time-steps", type=int, default=None,
                    help="only time N optimizer steps and write results/timing_<run>.json")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-model", action="store_true", help="save fp16 weights to checkpoints/")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    a = ap.parse_args(argv)
    if a.quick:
        a.subset, a.val_subset, a.epochs, a.evals_per_epoch = 100, a.val_subset or 100, 1, 2
    if a.run_name is None:
        tag = "nopre" if a.no_pretrain else (f"k{a.transfer_layers}" if a.transfer_layers is not None else "pre")
        a.run_name = f"{a.task}_{tag}_lam{a.lam:g}_seed{a.seed}" + ("_quick" if a.quick else "")
    assert a.batch_size % a.micro_batch == 0
    return a


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(a, n_classes) -> GPTClassifier:
    if a.no_pretrain:
        gpt = GPT(GPTConfig(vocab_size=BASE_VOCAB))
    else:
        from load_pretrained import load_pretrained_gpt
        gpt = load_pretrained_gpt()
        if a.transfer_layers is not None and a.transfer_layers < gpt.cfg.n_layer:
            gpt.reinit_layers_from(a.transfer_layers)
    gpt.resize_token_embeddings(BASE_VOCAB + len(SPECIAL_TOKENS))
    gpt.grad_checkpoint = a.grad_checkpoint
    return GPTClassifier(gpt, n_classes, a.clf_dropout)


def losses(model, batch, lam, device):
    ids = batch["input_ids"].to(device, non_blocking=True)
    ext = batch["extract_idx"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)
    logits, h = model(ids, ext)
    clf = F.cross_entropy(logits.float(), y)
    lm = None
    if lam > 0:
        T = ids.shape[-1]
        flat_ids = ids.reshape(-1, T)
        mask = batch["lm_mask"].to(device).reshape(-1, T - 1)
        h_sel = h[:, :-1][mask]                         # (N_real, C): gather BEFORE projecting
        tgt = flat_ids[:, 1:][mask]
        lm = F.cross_entropy(model.transformer.lm_logits(h_sel).float(), tgt)
    loss = clf + lam * lm if lm is not None else clf
    return loss, clf, lm, logits


@torch.no_grad()
def evaluate(model, loader, device, use_fp16, desc="eval"):
    model.eval()
    ys, preds, probs, tot, n = [], [], [], 0.0, 0
    for b in tqdm(loader, desc=f"    {desc}", leave=False, unit="batch", dynamic_ncols=True):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            logits, _ = model(b["input_ids"].to(device), b["extract_idx"].to(device))
        logits = logits.float()
        y = b["labels"].to(device)
        tot += F.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        p = logits.softmax(-1)
        ys += y.tolist()
        preds += p.argmax(-1).tolist()
        probs += p[:, 1].tolist()
    model.train()
    return tot / n, ys, preds, probs


def run(a, micro, accum, device):
    use_fp16 = device.type == "cuda" and not a.no_fp16
    spec = TASKS[a.task]
    set_seed(a.seed)
    sp = special_ids(BASE_VOCAB)
    train_ds = GlueDataset(a.task, "train", a.max_len, a.subset, a.seed)
    val_ds = GlueDataset(a.task, "validation", a.max_len, a.val_subset, a.seed)
    collate = make_collate(sp["<pad>"], spec.pair)
    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(train_ds, batch_size=micro, shuffle=True, collate_fn=collate,
                          num_workers=0, generator=g, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=max(micro, 16), shuffle=False, collate_fn=collate,
                        num_workers=0)

    model = build_model(a, spec.n_classes).to(device)
    n_params = count_parameters(model)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = math.ceil(len(train_dl) / accum)
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, round(a.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    eval_every = max(1, steps_per_epoch // a.evals_per_epoch)
    curves = {k: [] for k in ("step", "epoch", "lr", "train_loss", "train_lm_loss", "train_acc",
                              "val_loss", "val_acc")}
    win = {"clf": 0.0, "lm": 0.0, "lm_n": 0, "correct": 0, "n": 0, "b": 0}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    step, timing = 0, None
    model.train()
    n_steps_shown = min(total_steps, a.time_steps + 5) if a.time_steps else total_steps
    pbar = tqdm(total=n_steps_shown, desc=f"[{a.run_name}] train", unit="step", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} optimizer steps [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for epoch in range(a.epochs):
        pbar.set_description(f"[{a.run_name}] epoch {epoch + 1}/{a.epochs}")
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                loss, clf, lm, logits = losses(model, b, a.lam, device)
            scaler.scale(loss / accum).backward()
            win["clf"] += clf.item(); win["b"] += 1
            if lm is not None:
                win["lm"] += lm.item(); win["lm_n"] += 1
            win["correct"] += (logits.argmax(-1).cpu() == b["labels"]).sum().item()
            win["n"] += len(b["labels"])

            last = i + 1 == len(train_dl)
            if (i + 1) % accum == 0 or last:
                if a.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                if win["b"]:
                    pbar.set_postfix(clf=f"{win['clf'] / win['b']:.3f}",
                                     lm=f"{win['lm'] / win['lm_n']:.2f}" if win["lm_n"] else "-",
                                     acc=f"{win['correct'] / max(1, win['n']):.3f}",
                                     lr=f"{sched.get_last_lr()[0]:.2e}", refresh=False)

                if a.time_steps is not None:
                    if step == 5:                       # skip warm-up / cudnn autotune steps
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t_timing = time.time()
                    if step == a.time_steps + 5:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        sec = (time.time() - t_timing) / a.time_steps
                        timing = {"sec_per_step": sec, "steps_per_epoch": steps_per_epoch,
                                  "est_train_time_s": sec * total_steps}
                        break

                if step % eval_every == 0 or last:
                    vl, ys, pr, _ = evaluate(model, val_dl, device, use_fp16, "validation")
                    curves["step"].append(step)
                    curves["epoch"].append(step / steps_per_epoch)
                    curves["lr"].append(sched.get_last_lr()[0])
                    curves["train_loss"].append(win["clf"] / win["b"])
                    curves["train_lm_loss"].append(win["lm"] / win["lm_n"] if win["lm_n"] else None)
                    curves["train_acc"].append(win["correct"] / win["n"])
                    curves["val_loss"].append(vl)
                    curves["val_acc"].append(float(np.mean(np.array(ys) == np.array(pr))))
                    tqdm.write(f"  ep {step / steps_per_epoch:5.2f} step {step:5d}/{total_steps} "
                          f"train_clf {curves['train_loss'][-1]:.4f} "
                          f"lm {curves['train_lm_loss'][-1] or 0:.3f} "
                          f"train_acc {curves['train_acc'][-1]:.3f} | val_loss {vl:.4f} "
                          f"val_acc {curves['val_acc'][-1]:.4f} | {time.time() - t0:6.0f}s")
                    win = {k: 0 if isinstance(v, int) else 0.0 for k, v in win.items()}
        if timing:
            break
    pbar.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None
    if timing is not None:
        timing.update({"peak_vram_mib": peak, "micro_batch": micro, "grad_accum": accum,
                       "total_steps": total_steps, "n_train": len(train_ds)})
        return {"timing": timing}, model

    t1 = time.time()
    vl, ys, pr, probs = evaluate(model, val_dl, device, use_fp16, "final evaluation")
    eval_time = time.time() - t1
    metrics = compute_metrics(ys, pr, spec.positive_label)
    metrics["val_loss"] = vl
    preds = [{"idx": r["idx"], "text": r["text"], "label": y, "pred": p, "prob_pos": q}
             for r, y, p, q in zip(val_ds.rows, ys, pr, probs)]
    res = {
        "run_name": a.run_name, "task": a.task, "label_names": list(spec.label_names),
        "setting": {"pretrained": not a.no_pretrain, "transfer_layers": a.transfer_layers,
                    "lambda": a.lam},
        "hyperparams": {"lr": a.lr, "epochs": a.epochs, "effective_batch": micro * accum,
                        "micro_batch": micro, "grad_accum": accum, "max_len": train_ds.max_len,
                        "warmup_frac": a.warmup, "warmup_steps": warmup, "total_steps": total_steps,
                        "weight_decay": a.weight_decay, "clf_dropout": a.clf_dropout,
                        "max_grad_norm": a.max_grad_norm, "seed": a.seed, "fp16": use_fp16,
                        "grad_checkpoint": a.grad_checkpoint, "optimizer": "AdamW(0.9,0.999,1e-8)",
                        "schedule": "linear warmup + linear decay to 0"},
        "data": {"n_train": len(train_ds), "n_val": len(val_ds), "subset": a.subset,
                 "val_subset": a.val_subset, "n_train_truncated": train_ds.n_truncated,
                 "n_val_truncated": val_ds.n_truncated, "tokenizer_mode": train_ds.tokenizer_mode},
        "metrics": metrics, "curves": curves,
        "compute": {"train_time_s": train_time, "eval_time_s": eval_time, "peak_vram_mib": peak,
                    "peak_vram_reserved_mib": peak_reserved,
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "n_parameters": n_params, "torch": torch.__version__,
                    "platform": platform.platform()},
        "quick": a.quick,
    }
    return {"result": res, "preds": preds}, model


def main(argv=None):
    a = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import transformers
        tf_version = transformers.__version__
    except Exception:
        tf_version = None
    micro, accum = a.micro_batch, a.batch_size // a.micro_batch
    oom_events = []
    print(f"[{a.run_name}] device={device} micro={micro} accum={accum} lambda={a.lam} "
          f"pretrained={not a.no_pretrain} k={a.transfer_layers}")
    while True:
        oom = False
        try:
            out, model = run(a, micro, accum, device)
            break
        except torch.cuda.OutOfMemoryError as e:
            oom = True
            msg = str(e).splitlines()[0]
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            if micro == 1:
                raise RuntimeError("OOM even with micro-batch 1")
            oom_events.append({"micro_batch": micro, "grad_accum": accum, "error": msg[:200]})
            micro, accum = micro // 2, accum * 2
            print(f"[OOM] retrying with micro-batch {micro} x accum {accum}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    if "timing" in out:
        t = out["timing"]
        t["oom_events"] = oom_events
        t["run_name"] = a.run_name
        with open(os.path.join(a.out_dir, f"timing_{a.run_name}.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, indent=2)
        print(f"[timing] {t['sec_per_step']:.3f} s/step x {t['total_steps']} steps "
              f"= {t['est_train_time_s'] / 60:.1f} min est. | peak VRAM {t['peak_vram_mib']:.0f} MiB")
    else:
        res = out["result"]
        res["oom_events"] = oom_events
        res["compute"]["transformers"] = tf_version
        with open(os.path.join(a.out_dir, f"{a.run_name}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
        with open(os.path.join(a.out_dir, "preds", f"{a.run_name}.json"), "w", encoding="utf-8") as f:
            json.dump(out["preds"], f)
        m = res["metrics"]
        print(f"[{a.run_name}] acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
              f"f1_pos={m['f1_pos']:.4f} time={res['compute']['train_time_s'] / 60:.1f}min "
              f"peakVRAM={res['compute']['peak_vram_mib'] or 0:.0f}MiB")
        if a.save_model:
            ck = os.path.join(ROOT, "checkpoints")
            os.makedirs(ck, exist_ok=True)
            torch.save({"state_dict": {k: v.half() for k, v in model.state_dict().items()},
                        "config": model.transformer.cfg.to_dict(), "task": a.task,
                        "n_classes": TASKS[a.task].n_classes, "run_name": a.run_name},
                       os.path.join(ck, f"{a.run_name}.pt"))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
