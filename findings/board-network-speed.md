# Board networks: only the small GNN is fast enough, and only on a GPU

*2026-10-01. `benchmarks/arch_speed.py` on this CPU (half the cores) and a free Colab T4; end-to-end PPO runs on
the T4. Raw tables: `benchmarks/results/arch_speed_t4.csv` (local only).*

**Network-only PPO estimate** (samples/s; inference at 256 envs, training at minibatch 4096, 4 epochs):

| Network | CPU | T4 | T4 + `torch.compile` |
|---|---|---|---|
| MLP 2×256 | ~18k | 153k | |
| GNN d64 L4 | ~250 | 2.4k | 3.3k |
| GNN d128 L4 | | 1.2k | 1.4k |
| transformer / hybrid d64 L4 | 60–75 | 0.7–0.9k | |
| transformer / hybrid d128 | 17–60 | 0.3–0.5k | |

- Board networks cost 20–30× the MLP's FLOPs per sample (145 nodes × d² per layer). A 6M-step diagnostic with the
  smallest GNN takes ~7 h on this CPU, ~35 min on the T4.
- The transformer and hybrid are 3–5× slower than the GNN of the same width. The transformer d128 runs out of
  memory (15 GB) at 4096-sample chunks; 1024 works.

**End-to-end d64 vs d128** (real PPO, fp16, CUDA-graph rollouts, league on, same recipe):

| | d64 | d128 | ratio |
|---|---|---|---|
| samples/s before / with league | 3,489 / 2,981 | 1,812 / 1,528 | 1.93× / 1.95× |
| PPO update per iteration (league) | 4.8 s | 10.6 s | 2.2× |
| rollout per env step (league) | 16.3 ms | 23.2 ms | 1.4× |

The update dominates and is GPU-bound (see [gpu-batch-size-and-precision.md](gpu-batch-size-and-precision.md)),
so d128 costs ~2×: 6M steps ≈ 35 min for d64, ≈ 70 min for d128.
