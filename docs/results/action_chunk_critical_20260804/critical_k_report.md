# Pi0.5 Action-Chunk Critical-Point Sweep

All reported groups contain 1,800/1,800 `status=ok` episodes with complete artifacts and semantic validation. Raw task success and physical diagnostics are reported separately.

## Critical Points

| Backend | Frozen RTX 5090 latency | Delay budget D | Theory K* | Empirical K* | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| JAX | 60.294 ms | 2 | 3 | 4 | K=3: 247.78; K=4: 0.00 steady underflow/1k |
| PYTORCH | 147.199 ms | 3 | 5 | 6 | K=5: 164.26; K=6: 0.00 steady underflow/1k |

## JAX

| K | Success | 0.03 m/s | 0.06 m/s | 0.09 m/s | Diagnostics | Total UF/1k | Steady UF/1k | Requests/1k | Action L2 | Delta L2 | Mean steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48/1800 (2.67%) | 2.67% | 2.67% | 2.67% | 98.17% | 750.00 | 746.65 | 500.01 | 1.191 | 0.044 | 596.8 |
| 3 | 670/1800 (37.22%) | 41.33% | 39.17% | 31.17% | 87.28% | 251.90 | 247.78 | 499.82 | 1.148 | 0.177 | 485.6 |
| 4 | 1075/1800 (59.72%) | 64.50% | 61.83% | 52.83% | 89.11% | 5.51 | 0.00 | 497.64 | 1.140 | 0.205 | 362.7 |
| 8 | 1009/1800 (56.06%) | 63.83% | 55.67% | 48.67% | 90.06% | 5.29 | 0.00 | 249.82 | 1.107 | 0.172 | 377.9 |
| 10 | 991/1800 (55.06%) | 64.83% | 52.83% | 47.50% | 90.44% | 5.20 | 0.00 | 200.01 | 1.094 | 0.169 | 384.8 |

## PYTORCH

| K | Success | 0.03 m/s | 0.06 m/s | 0.09 m/s | Diagnostics | Total UF/1k | Steady UF/1k | Requests/1k | Action L2 | Delta L2 | Mean steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/1800 (0.00%) | 0.00% | 0.00% | 0.00% | 98.94% | 833.33 | 828.33 | 333.33 | 1.196 | 0.021 | 600.0 |
| 3 | 203/1800 (11.28%) | 13.17% | 11.50% | 9.17% | 92.56% | 500.07 | 494.84 | 333.34 | 1.163 | 0.093 | 573.6 |
| 4 | 430/1800 (23.89%) | 27.83% | 24.83% | 19.00% | 89.94% | 335.20 | 329.53 | 333.18 | 1.147 | 0.124 | 528.8 |
| 5 | 695/1800 (38.61%) | 41.50% | 38.00% | 36.33% | 89.06% | 170.73 | 164.26 | 333.05 | 1.131 | 0.148 | 463.2 |
| 6 | 871/1800 (48.39%) | 55.67% | 45.17% | 44.33% | 90.33% | 7.31 | 0.00 | 331.30 | 1.126 | 0.166 | 410.3 |
| 8 | 833/1800 (46.28%) | 54.17% | 46.83% | 37.83% | 89.22% | 7.23 | 0.00 | 249.58 | 1.110 | 0.159 | 415.0 |
| 16 | 707/1800 (39.28%) | 48.83% | 35.33% | 33.67% | 91.00% | 6.72 | 0.00 | 125.07 | 1.069 | 0.147 | 446.6 |
| 32 | 492/1800 (27.33%) | 31.33% | 28.00% | 22.67% | 94.11% | 6.13 | 0.00 | 63.39 | 1.003 | 0.128 | 489.4 |
| 50 | 330/1800 (18.33%) | 20.00% | 19.00% | 16.00% | 95.11% | 5.77 | 0.00 | 40.14 | 0.952 | 0.115 | 520.2 |

## Interpretation

The theoretical boundary uses `D = ceil(latency * 20 Hz)` and `T = ceil(0.5 K)`. Equality `T = D` is not sufficient in the measured queue schedule: JAX K=3 and PyTorch K=5 both starve in steady state. The first strictly sustainable settings are therefore JAX K=4 and PyTorch K=6.

Success is not monotonic in K. Once K clears the queue-sustainability boundary, larger chunks reduce policy refresh frequency and can lower task success even though steady-state underflow remains zero.
