# GPU rollouts were bound by kernel launches; CUDA graphs fixed it

*2026-10-01. Run 5's `metrics.csv`, then microbenchmarks and short PPO runs on a free Colab T4 (GNN d64 L4, fp16).*

**Symptom.** Run 5 slowed from ~3.3k to ~2.1k samples/s once league snapshots started playing. Two causes:

1. Opponent moves aren't training data, so samples per iteration drop ~37% (32.8k → 20.7k). By design.
2. Rollout time per env step rose 3.5× (9.5 → 31–40 ms): four `act()` calls per step (learner + 3 snapshots)
   instead of one.

**Cause of 2.** An eager GNN `act()` costs ~8 ms at any batch size from 8 to 256 rows. The GPU work for a small
batch is ~2 ms; the rest is the host launching ~350 kernels per call (half of them autocast weight casts).

**What didn't work.** Stacking the four policies with `torch.vmap` (one call per step): 33 ms vs 42 ms for a
league step, because every policy is padded to the learner's batch and vmap adds its own overhead.

**Fix.** `GraphedPolicy`: capture the forward once per padded batch bucket as a CUDA graph and replay it. Identical
outputs; in-place optimizer updates are seen without recapture.

| | before | after |
|---|---|---|
| one `act()`, 8–64 rows | ~8 ms | ~3–3.5 ms |
| league step (190 + 3×22 rows) | 42 ms | 13 ms |
| rollout per env step in PPO, league on | 31–40 ms | 15–16 ms |
| samples/s, league on | 2.0–2.2k | 2.7–3.2k |

On the CPU none of this applies: compute dominates there, so actors stay eager.
