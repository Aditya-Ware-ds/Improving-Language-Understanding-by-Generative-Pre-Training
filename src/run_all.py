"""Run the whole experiment suite with live progress, resumably.

    python src/run_all.py --smoke          # step 1: 100-example / 1-epoch smoke tests of every script
    python src/run_all.py --dry-run        # show the plan and the time estimate, run nothing
    python src/run_all.py                  # step 2: all experiments E1-E6 (+ tables, figures, report)
    python src/run_all.py --only E1 E5     # a subset
    python src/run_all.py --micro-batch 4  # if you want a smaller per-step batch from the start

Each experiment runs in its own subprocess (so GPU memory is fully released between runs) and
shows its own progress bar. An experiment whose results/<name>.json already exists is SKIPPED,
so after a crash or Ctrl-C just run the same command again and it continues where it stopped.
Status is also written to logs/run_all_status.json.

Time estimates come from 50-step timing tests on an RTX 3050 Ti Laptop GPU (4 GB):
SST-2 ~0.25-0.30 s/step x 6315 steps, MRPC ~1.7 s/step x 345 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
EST_MIN = {"sst2": 30.0, "mrpc": 10.5}           # full fine-tuning run, minutes (RTX 3050 Ti)


def plan(seeds, mb):
    mbf = ["--micro-batch", str(mb)]
    P = []   # (group, name, argv, est_minutes, output file that marks it done)

    def train(group, name, task, *extra):
        P.append((group, name, [PY, "src/train.py", "--task", task, "--run-name", name, *extra, *mbf],
                  EST_MIN[task], f"results/{name}.json"))

    P.append(("W", "weight_match_test", [PY, "src/load_pretrained.py", "--test"], 3, "results/weight_match.json"))
    P.append(("L", "length_analysis", [PY, "src/data.py"], 4, "results/length_stats_mrpc.json"))
    # SST-2 (primary application)
    train("E1", "E1_sst2_seed42", "sst2", "--seed", "42", "--save-model")
    train("E2", "E2_sst2_seed42", "sst2", "--seed", "42", "--lambda", "0")
    train("E3", "E3_sst2_seed42", "sst2", "--seed", "42", "--no-pretrain")
    # MRPC (secondary application) -- main runs over several seeds
    for s in seeds:
        train("E1", f"E1_mrpc_seed{s}", "mrpc", "--seed", str(s), *(["--save-model"] if s == 42 else []))
        train("E2", f"E2_mrpc_seed{s}", "mrpc", "--seed", str(s), "--lambda", "0")
    train("E3", "E3_mrpc_seed42", "mrpc", "--seed", "42", "--no-pretrain")
    for k in (0, 3, 6, 9):                        # k = 12 is E1_mrpc_seed42
        train("E6", f"E6_mrpc_k{k}_seed42", "mrpc", "--seed", "42", "--transfer-layers", str(k))
    P.append(("E4", "E4_tfidf_lr_sst2", [PY, "src/baselines.py", "--task", "sst2"], 4, "results/E4_tfidf_lr_sst2.json"))
    P.append(("E4", "E4_tfidf_lr_mrpc", [PY, "src/baselines.py", "--task", "mrpc"], 1, "results/E4_tfidf_lr_mrpc.json"))
    P.append(("E5", "E5_zeroshot_sst2", [PY, "src/zero_shot.py"], 2, "results/E5_zeroshot_sst2.json"))
    P.append(("R", "report_assets", [PY, "src/make_report_assets.py", "--attention"], 2, None))
    P.append(("R", "build_report", [PY, "src/build_report.py"], 1, None))
    return P


def smoke_plan(mb):
    q = ["--quick", "--micro-batch", str(mb)]
    t = lambda name, *a: ("S", name, [PY, "src/train.py", *a, *q], 1.5, None)
    return [
        ("S", "weight_match_test", [PY, "src/load_pretrained.py", "--test"], 3, None),
        ("S", "length_analysis", [PY, "src/data.py"], 4, None),
        t("smoke_sst2_lambda0.5 (+save)", "--task", "sst2", "--save-model"),
        t("smoke_sst2_lambda0", "--task", "sst2", "--lambda", "0"),
        t("smoke_sst2_no_pretrain", "--task", "sst2", "--no-pretrain"),
        t("smoke_mrpc_lambda0.5", "--task", "mrpc"),
        t("smoke_mrpc_transfer_k3", "--task", "mrpc", "--transfer-layers", "3"),
        t("smoke_mrpc_grad_checkpoint", "--task", "mrpc", "--grad-checkpoint"),
        ("S", "smoke_timing_50_steps_sst2", [PY, "src/train.py", "--task", "sst2", "--time-steps", "50",
                                             "--run-name", "timing_sst2", "--micro-batch", str(mb)], 1, None),
        ("S", "smoke_timing_50_steps_mrpc", [PY, "src/train.py", "--task", "mrpc", "--time-steps", "50",
                                             "--run-name", "timing_mrpc", "--micro-batch", str(mb)], 2, None),
        ("S", "smoke_baseline_sst2", [PY, "src/baselines.py", "--task", "sst2", "--quick"], 0.5, None),
        ("S", "smoke_baseline_mrpc", [PY, "src/baselines.py", "--task", "mrpc", "--quick"], 0.5, None),
        ("S", "smoke_zero_shot", [PY, "src/zero_shot.py", "--quick"], 1, None),
        ("S", "smoke_predict", [PY, "src/predict.py", "--ckpt", "checkpoints/sst2_pre_lam0.5_seed42_quick.pt",
                                "what a wonderful , moving film .", "a tedious mess ."], 0.5, None),
        ("S", "smoke_report_assets", [PY, "src/make_report_assets.py"], 1, None),
    ]


def fmt(m):
    m = max(0, m)
    return f"{int(m // 60)}h{int(m % 60):02d}m" if m >= 60 else f"{m:.0f}m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", help="groups to run, e.g. E1 E3 E6 (W=weight test, R=report)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="MRPC seeds for E1/E2")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs("logs", exist_ok=True)

    P = smoke_plan(a.micro_batch) if a.smoke else plan(a.seeds, a.micro_batch)
    if a.only:
        P = [p for p in P if p[0] in a.only]
    todo = [p for p in P if not (p[4] and os.path.exists(p[4]))]
    total_est = sum(p[3] for p in todo)
    print("=" * 78)
    print(f"{'SMOKE TESTS' if a.smoke else 'EXPERIMENT SUITE'}: {len(P)} steps, {len(P) - len(todo)} already done, "
          f"{len(todo)} to run, estimated {fmt(total_est)} (RTX 3050 Ti timings)")
    for i, (g, name, argv, est, out) in enumerate(P, 1):
        done = out and os.path.exists(out)
        print(f"  {i:2d}. [{g:2s}] {name:34s} ~{fmt(est):>6s}  {'(done - will skip)' if done else ''}")
    print("=" * 78, flush=True)
    if a.dry_run:
        return

    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}
    t_start, remaining, failed = time.time(), total_est, []
    for i, (g, name, argv, est, out) in enumerate(P, 1):
        if out and os.path.exists(out):
            print(f"\n[{i}/{len(P)}] {name}: results exist, skipping")
            continue
        elapsed = (time.time() - t_start) / 60
        print(f"\n{'#' * 78}\n[{i}/{len(P)}] {name}   (this step ~{fmt(est)}, "
              f"suite elapsed {fmt(elapsed)}, remaining ~{fmt(remaining)})\n$ {' '.join(argv[1:])}\n{'#' * 78}",
              flush=True)
        t0 = time.time()
        rc = subprocess.call(argv, env={**os.environ, "PYTHONWARNINGS": "ignore",
                                        "TOKENIZERS_PARALLELISM": "false",
                                        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        took = (time.time() - t0) / 60
        remaining -= est
        status["steps"].append({"step": name, "returncode": rc, "minutes": round(took, 2)})
        with open("logs/run_all_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        print(f"[{i}/{len(P)}] {name}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} in {fmt(took)}", flush=True)
        if rc != 0:
            failed.append(name)
            if not a.keep_going:
                print("Stopping. Fix the problem and re-run the same command - finished steps are skipped.")
                sys.exit(rc)
    print(f"\nAll done in {fmt((time.time() - t_start) / 60)}. Failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
