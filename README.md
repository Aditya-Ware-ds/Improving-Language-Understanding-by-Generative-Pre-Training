# GPT-1 assignment: study, from-scratch re-implementation and fine-tuning


Base paper: Radford, Narasimhan, Salimans & Sutskever (2018), *Improving Language Understanding by Generative
Pre-Training* (`paper\language_understanding_paper.pdf`; verified notes in `notes\paper_notes.md`).

GPT-1 is written from scratch in PyTorch (`src\model.py`), loaded with the released weights
(`openai-community/openai-gpt`), checked numerically against the reference implementation, and fine-tuned with
the paper's objective L3 = L2 + λ·L1 on:

- **SST-2**: sentiment of reviews (opinion mining on customer feedback). Primary application.
- **MRPC**: paraphrase / duplicate detection, using the paper's two-ordering similarity transformation.

The report (`report\report.md` → `report\report.docx`) is **generated from `results\`**. Nothing in it is typed
by hand, and a run you have not done yet shows up as "*not run*".

> Every command in this README is a **Windows PowerShell** command. Unless a step says otherwise, run it from the
> `gpt1-assignment` folder with the virtual environment activated.

---

## 1. Setup (Windows PowerShell, Python 3.12, NVIDIA GPU)

```powershell
Set-Location C:\Users\adity\Desktop\manit_assignmentes\soft_computing\assignments\gpt1-assignment

# one-time: allow the venv activation script to run for your user
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# one-time, only if needed: Python 3.12 (py -0p lists the Pythons you already have) and Pandoc (for the Word report)
winget install -e --id Python.Python.3.12
winget install -e --id JohnMacFarlane.Pandoc

# reload PATH so this window sees the new installs (do this BEFORE activating the venv)
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
pandoc --version

# virtual environment and packages
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# check that PyTorch sees the GPU (should print True and the GPU name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run `.\.venv\Scripts\Activate.ps1` again in every new PowerShell window.

- `transformers` is pinned to a 4.x release that still ships `OpenAIGPTModel` / `OpenAIGPTTokenizer`.
  It is used only to download the weights, to tokenise, and as the reference in the weight-match test.
- **ftfy + spaCy** make the tokenizer reproduce the original GPT pre-processing. Without them Hugging Face
  silently falls back to BERT's BasicTokenizer, which splits text slightly differently. Every script prints which
  mode is active (`[tokenizer] ftfy+spacy ...`) and it is recorded in each result file.
- **Slow or unreliable internet?** Download the ~479 MB of weights with the resumable script first. The code
  picks the local copy up automatically:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\download_weights.ps1   # re-run it if it stops; it resumes
  ```
- Data loading already uses `num_workers=0`, which Windows needs. If Hugging Face prints a warning about
  symlinks in its cache, it is harmless (it only uses a little more disk space).

## 2. What to run, in order

Everything shows live progress: a progress bar per run with optimizer steps, loss, accuracy, learning rate and
ETA, validation lines several times per epoch, and an overall `[i/N] … remaining ~Xh` banner from the driver.

### Step 1: smoke tests (about 15–20 min, mostly first-time downloads and tokenisation)

```powershell
python src\run_all.py --smoke
```

This runs every script on 100 examples for 1 epoch:

| # | What it runs | What to check |
|---|---|---|
| 1 | `load_pretrained.py --test` (**weight-match test**) | prints `PASSED` and each case has `max|diff|` < 1e-4; writes `results\weight_match.json` |
| 2 | `data.py` (length analysis) | prints length percentiles; `0.00%` over max_len for both tasks |
| 3–8 | `train.py --quick` for SST-2 λ=0.5 / λ=0 / no-pretrain, MRPC λ=0.5 / k=3 / grad-checkpoint | each finishes and prints `acc=… f1_macro=… peakVRAM=…`; no Python error |
| 9–10 | 50-step timing tests | prints `s/step x steps = … min est.` and peak VRAM (this is your runtime estimate) |
| 11–12 | `baselines.py --quick` | prints `C=… acc=…` |
| 13 | `zero_shot.py --quick` | prints the paper-rule accuracy |
| 14 | `predict.py` with the smoke checkpoint | prints a label and probability for two sentences |
| 15 | `make_report_assets.py` | writes files in `results\tables` and `results\figures` |

Smoke outputs are marked `"quick": true` and are ignored by the report.

If the weight-match test fails, stop there: every other result depends on it. The failure details are in
`results\weight_match_FAILED.json`:

```powershell
Get-Content results\weight_match_FAILED.json -Raw
```

### Step 2: the full experiment suite (about 3.5–4 h on an RTX 3050 Ti 4 GB)

Keep the laptop plugged in and stop Windows from sleeping during the run, otherwise the run pauses when the
machine sleeps:

```powershell
# while plugged in: never sleep, and do nothing when the lid is closed
powercfg /change standby-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
```

Then start the suite:

```powershell
python src\run_all.py --dry-run   # shows the plan and time estimate, runs nothing
python src\run_all.py             # runs everything, then builds tables, figures and the report
```

| Step | Experiment | Est. time (3050 Ti) | Output |
|---|---|---|---|
| E1 | GPT-1 pre-trained, λ = 0.5: SST-2 seed 42; MRPC seeds 42, 43, 44 | 30 min + 3 × 10 min | `results\E1_*.json`, `checkpoints\E1_sst2_seed42.pt` |
| E2 | pre-trained, λ = 0 (no auxiliary LM): same runs | 30 min + 3 × 10 min | `results\E2_*.json` |
| E3 | random init (no pre-training): SST-2 and MRPC, seed 42 | 30 min + 10 min | `results\E3_*.json` |
| E4 | TF-IDF + logistic regression (CPU) | ~5 min | `results\E4_tfidf_lr_*.json` |
| E5 | zero-shot SST-2 with the paper's "very" heuristic | ~2 min | `results\E5_zeroshot_sst2.json` |
| E6 | layer transfer on MRPC, k ∈ {0, 3, 6, 9} (k = 12 is E1) | 4 × 10 min | `results\E6_mrpc_k*_seed42.json` |
| R | tables, figures, attention map, report | ~3 min | `results\tables`, `results\figures`, `report\report.{md,docx}` |

The estimates come from 50-step timing tests on this laptop (`results\timing_*.json`): about 0.25–0.30 s per
optimizer step × 6,315 steps for SST-2 and about 1.7 s × 345 steps for MRPC, at micro-batch 8 × accumulation 4
with a peak of about 2.3 GB (SST-2) and 3.0 GB (MRPC) of VRAM. The laptop GPU idles at 80–86 °C, so thermal
throttling can make real runs slower.

When the suite has finished, put the power settings back (1 = sleep when the lid closes; use your usual
sleep timeout in minutes instead of 30):

```powershell
powercfg /change standby-timeout-ac 30
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setactive SCHEME_CURRENT
```

**It is resumable.** If a run crashes, the laptop sleeps or you press Ctrl+C, run the same command again.
Every experiment whose `results\<name>.json` exists is skipped. Status: `logs\run_all_status.json`.
To redo an experiment, delete its `results\<name>.json`.

Note: a few result files may already be present from the development session (for example
`results\E4_tfidf_lr_mrpc.json`, and possibly `results\E3_*` if those runs completed). `run_all.py` will skip
them. If you want every number to come from your own runs, delete those results first (the length statistics
can stay):

```powershell
Remove-Item results\E*.json, results\timing_*.json -ErrorAction SilentlyContinue
Remove-Item results\preds -Recurse -Force -ErrorAction SilentlyContinue
```

Useful variants:

```powershell
python src\run_all.py --only E1 E5          # a subset of groups (W, L, E1–E6, R)
python src\run_all.py --seeds 42            # MRPC with one seed only (saves ~40 min)
python src\run_all.py --micro-batch 4       # smaller per-step batch from the start (accumulation 8)
python src\run_all.py --keep-going          # do not stop at a failed step

# redo one experiment: delete its result file, then run again
Remove-Item results\E2_sst2_seed42.json
python src\run_all.py --only E2
```

Monitoring from a second PowerShell window (activate the venv there too if you run Python in it):

```powershell
# run status, refreshed every 30 s (Ctrl+C to stop watching; the run itself keeps going)
while ($true) { Clear-Host; Get-Content logs\run_all_status.json -Raw; Start-Sleep -Seconds 30 }

# GPU temperature, load, memory and clock every 10 s (Ctrl+C to stop)
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks.sm --format=csv -l 10
```

**Out of memory?** `train.py` catches CUDA OOM, halves the micro-batch, doubles accumulation (effective batch
stays 32), restarts the run and records it in `oom_events` in the result file. You can also add
`--grad-checkpoint` to a single run.

### Step 3: rebuild the report (any time, takes seconds)

```powershell
python src\make_report_assets.py --attention   # tables + figures (+ attention map, needs the E1 checkpoint)
python src\build_report.py                     # report\report.md + report\report.docx (+ .pdf if LaTeX exists)
Start-Process report\report.docx               # open it in Word
```

Then **open `report\report.docx`** and check:

1. Word asks to update fields for the table of contents: click *Yes* (or right-click the TOC → *Update field*).
2. Every figure renders. If a figure or table is missing, the report shows "*not generated yet*" in its place.
3. Every "not run" is an experiment you have not run yet. To list them with line numbers:
   ```powershell
   Select-String -Path report\report.md -Pattern "not run", "not generated yet" -SimpleMatch
   ```
4. Fill in the title-page placeholders (`[Course name and code]`, `[Faculty name…]`, `[Submission date]`) in
   `report\report_template.md` and rebuild, or edit them directly in Word:
   ```powershell
   Select-String -Path report\report_template.md -Pattern "Course name", "Faculty name", "Submission date" -SimpleMatch
   notepad report\report_template.md
   python src\build_report.py
   ```
5. For a PDF without LaTeX: Word → *File → Save as → PDF*.

Steps 1 and 5 can also be done from PowerShell (needs Microsoft Word installed). Close the document in Word
first. Word opens visibly so you can click *Yes* if it asks about updating fields:

```powershell
$docx = (Resolve-Path report\report.docx).Path
$pdf  = Join-Path (Resolve-Path report).Path "report.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $true
$doc  = $word.Documents.Open($docx)
foreach ($toc in $doc.TablesOfContents) { $toc.Update() }   # refresh the table of contents
$doc.Save()
$doc.ExportAsFixedFormat($pdf, 17)                           # 17 = PDF
$doc.Close()
$word.Quit()
```

## 3. Running single pieces by hand

```powershell
python src\load_pretrained.py --test                          # weight-match test
python src\data.py                                             # length statistics
python src\train.py --task sst2 --run-name E1_sst2_seed42 --save-model
python src\train.py --task mrpc --lambda 0 --seed 43 --run-name E2_mrpc_seed43
python src\train.py --task mrpc --no-pretrain --run-name E3_mrpc_seed42
python src\train.py --task mrpc --transfer-layers 6 --run-name E6_mrpc_k6_seed42
python src\train.py --task sst2 --time-steps 50                # timing only
python src\baselines.py --task sst2
python src\zero_shot.py
python src\predict.py "the delivery was late but the product is excellent"
python src\predict.py --ckpt checkpoints\E1_mrpc_seed42.pt --pair "The firm posted a loss." "The company lost money."
python src\evaluate.py results\E1_sst2_seed42.json             # metrics + plots for one run
```

Classifying many reviews at once (one review per line). The two settings make PowerShell and Python pass text
as UTF-8, so characters like é, ₹ or emoji arrive intact:

```powershell
$env:PYTHONUTF8 = "1"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Get-Content my_reviews.txt -Encoding UTF8 | python src\predict.py -    # from a text file
"battery lasts all day", "support never replied to my emails" | python src\predict.py -   # typed inline
```

`train.py` flags: `--lambda --lr --epochs --batch-size (effective) --micro-batch --max-len --seed --subset N
--val-subset N --no-pretrain --transfer-layers k --grad-checkpoint --no-fp16 --quick --time-steps N --save-model`.
The defaults are the paper's fine-tuning settings: lr 6.25e-5, batch 32, 3 epochs, linear decay with 0.2%
warm-up, λ = 0.5, classifier dropout 0.1, weight decay 0.01.

## 4. Colab alternative

`notebooks\GPT1_Colab.ipynb` runs the same pipeline on a free T4. Zip this folder without `cache`,
`checkpoints`, `weights`, `logs` and `.venv`, upload the zip, run all cells, then download `results.zip` and
unzip it over your local `results` folder.

Make the zip for upload (run from the `gpt1-assignment` folder; the zip is written next to it):

```powershell
Push-Location ..
$items = Get-ChildItem gpt1-assignment -Exclude cache, checkpoints, weights, .venv, logs | ForEach-Object FullName
Compress-Archive -Path $items -DestinationPath gpt1-assignment.zip -Force
Pop-Location
```

After downloading `results.zip` from Colab:

```powershell
Move-Item $HOME\Downloads\results.zip . -Force
Expand-Archive results.zip -DestinationPath . -Force
python src\build_report.py
```

`Compress-Archive` puts the folder's contents at the top level of the zip; the notebook's first cell handles
both that layout and a zip that contains the `gpt1-assignment` folder itself. On a T4 use `--micro-batch 32`;
the whole suite takes roughly 1.5–2 h there (not measured, an estimate from the relative speed of the GPUs).

## 5. Repository layout

```text
paper\          the PDF + extracted text
notes\          paper_notes.md: every paper number verified against the PDF
src\model.py              GPT-1 from scratch (no transformers model classes)
src\load_pretrained.py    HF weights → our model (Conv1D transposed) + weight-match test
src\paths.py              local weights\ folder if present, else the HF Hub
src\data.py               GLUE loading, BPE cache, input transformations, dynamic padding, length analysis
src\train.py              fine-tuning (L3), fp16, grad accumulation, OOM fallback, progress bars
src\evaluate.py           metrics + plots
src\baselines.py          E4 TF-IDF + LR
src\zero_shot.py          E5 zero-shot heuristic
src\predict.py            classify your own text
src\run_all.py            resumable driver for smoke tests / all experiments
src\make_report_assets.py results\*.json → results\tables\*.md, results\figures\*.png
src\build_report.py       report_template.md + results → report.md → report.docx (pandoc)
scripts\download_weights.ps1   resumable weight download (PowerShell)
notebooks\GPT1_Colab.ipynb     Colab version
results\        all metrics (JSON), predictions, tables, figures
report\         report_template.md (the text), report.md / report.docx (generated)
```

## 6. Caveats to keep in mind when reading the results

- Our numbers are on the **GLUE validation** splits (test labels are hidden); the paper reports **GLUE test**.
- SST-2 runs use one seed; the MRPC main runs use three. Differences of about one point are within seed noise.
- The paper's LSTM ablation is not reproduced (no pre-trained BooksCorpus LSTM is available).
- Pre-training itself is not repeated; we start from the released weights, as the assignment allows.
