| Exp | Model | Seeds | Accuracy (%) | Precision macro (%) | Recall macro (%) | F1 macro (%) | F1 positive class (%) |
|---|---|---|---|---|---|---|---|
| E1 | GPT-1 pre-trained, fine-tuned, λ = 0.5 | 42 | 92.78 | 92.85 | 92.74 | 92.77 | 93.02 |
| E2 | GPT-1 pre-trained, fine-tuned, λ = 0 (no aux LM) | 42 | 92.55 | 92.55 | 92.54 | 92.54 | 92.69 |
| E3 | Same Transformer, random init (no pre-training) | 42 | 82.34 | 82.61 | 82.25 | 82.27 | 83.37 |
| E4 | TF-IDF + Logistic Regression | – | 82.11 | 82.40 | 82.02 | 82.04 | 83.19 |
| E5 | Zero-shot LM heuristic (no fine-tuning) | – | 78.21 | 78.97 | 78.36 | 78.12 | 76.72 |

*SST2 GLUE validation split (n = 872). Mean ± sample std over seeds where more than one seed was run.*
