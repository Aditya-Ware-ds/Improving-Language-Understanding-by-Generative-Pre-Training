| Run | Task | Micro-batch × accum (= effective) | Max len | Train time (min) | Peak VRAM allocated (MiB) | OOM retries |
|---|---|---|---|---|---|---|
| E1_mrpc_seed42 | mrpc | 8 × 4 (= 32) | 104 | 8.8 | 3048 | 0 |
| E1_mrpc_seed43 | mrpc | 8 × 4 (= 32) | 104 | 8.9 | 3066 | 0 |
| E1_mrpc_seed44 | mrpc | 8 × 4 (= 32) | 104 | 9.3 | 3085 | 0 |
| E1_sst2_seed42 | sst2 | 8 × 4 (= 32) | 64 | 54.3 | 2330 | 0 |
| E2_mrpc_seed42 | mrpc | 8 × 4 (= 32) | 104 | 8.9 | 2546 | 0 |
| E2_mrpc_seed43 | mrpc | 4 × 8 (= 32) | 104 | 6.8 | 2280 | 0 |
| E2_mrpc_seed44 | mrpc | 4 × 8 (= 32) | 104 | 9.7 | 2281 | 0 |
| E2_sst2_seed42 | sst2 | 8 × 4 (= 32) | 64 | 46.7 | 2249 | 0 |
| E3_mrpc_seed42 | mrpc | 8 × 4 (= 32) | 104 | 10.0 | 3046 | 0 |
| E3_sst2_seed42 | sst2 | 8 × 4 (= 32) | 64 | 55.3 | 2328 | 0 |
| E6_mrpc_k0_seed42 | mrpc | 8 × 4 (= 32) | 104 | 10.5 | 3046 | 0 |
| E6_mrpc_k3_seed42 | mrpc | 8 × 4 (= 32) | 104 | 9.7 | 3048 | 0 |
| E6_mrpc_k6_seed42 | mrpc | 8 × 4 (= 32) | 104 | 8.7 | 3048 | 0 |
| E6_mrpc_k9_seed42 | mrpc | 8 × 4 (= 32) | 104 | 8.7 | 3048 | 0 |
| E4_tfidf_lr_mrpc | mrpc | – (CPU) | – | 0.3 | – | – |
| E4_tfidf_lr_sst2 | sst2 | – (CPU) | – | 0.8 | – | – |
| E5_zeroshot_sst2 | sst2 | – (inference only) | – | 0.1 | – | – |

*Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU; torch 2.6.0+cu124, transformers 4.57.6; Windows-11-10.0.26200-SP0. Classifier parameter count: 116,539,394.*
