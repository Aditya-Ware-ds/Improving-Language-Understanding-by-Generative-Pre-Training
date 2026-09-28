| Exp | Model | Seeds | Accuracy (%) | Precision macro (%) | Recall macro (%) | F1 macro (%) | F1 positive class (%) |
|---|---|---|---|---|---|---|---|
| E1 | GPT-1 pre-trained, fine-tuned, λ = 0.5 | 42,43,44 | 79.08 ± 2.00 | 77.23 ± 2.55 | 71.86 ± 2.68 | 73.43 ± 2.79 | 85.69 ± 1.29 |
| E2 | GPT-1 pre-trained, fine-tuned, λ = 0 (no aux LM) | 42,43,44 | 81.21 ± 1.87 | 80.09 ± 3.16 | 74.80 ± 1.78 | 76.45 ± 2.00 | 87.02 ± 1.46 |
| E3 | Same Transformer, random init (no pre-training) | 42 | 71.32 | 66.10 | 61.74 | 62.37 | 80.72 |
| E4 | TF-IDF + Logistic Regression | – | 75.00 | 70.97 | 69.43 | 70.04 | 82.23 |

*MRPC GLUE validation split (n = 408). Mean ± sample std over seeds where more than one seed was run.*
