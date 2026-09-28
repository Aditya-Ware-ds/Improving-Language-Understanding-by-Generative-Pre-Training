| Setting | Paper SST-2 acc (GLUE test) | Ours SST-2 acc (dev) | Paper MRPC F1 (GLUE test) | Ours MRPC F1 (dev) |
|---|---|---|---|---|
| full (w/ aux LM) | 91.3 | 92.78 | 82.3 | 85.69 ± 1.29 |
| w/o aux LM | 92.0 | 92.55 | 84.9 | 87.02 ± 1.46 |
| w/o pre-training | 84.0 | 82.34 | 79.4 | 80.72 |
| LSTM w/ aux LM | 90.5 | not run | 83.2 | not run |

*Paper values: Radford et al. (2018), Table 4 and Table 5 (GLUE benchmark). Ours: GLUE validation split. The two are not directly comparable (different split, and see caveats).*
