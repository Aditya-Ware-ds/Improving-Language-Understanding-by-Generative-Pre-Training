| Pre-trained layers transferred (k) | Accuracy (%) | F1 positive (%) | F1 macro (%) |
|---|---|---|---|
| 0 | 70.59 | 80.07 | 62.00 |
| 3 | 70.10 | 79.32 | 62.67 |
| 6 | 76.47 | 83.16 | 72.07 |
| 9 | 75.74 | 82.72 | 70.99 |
| 12 | 77.70 | 84.81 | 71.44 |
| none (random embeddings too; = E3) | 71.32 | 80.72 | 62.37 |

*MRPC validation, seed 42, λ = 0.5. k = 0 transfers only the token/position embeddings; k = 12 is the full model (E1, seed 42).*
