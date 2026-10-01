# Throughput on this CPU, and what happens when jobs share it

*2026-09-30. This machine: 14 cores, AVX2, no GPU.*

**PPO** (network-bound): ~6.5–7k samples/s for 2×256 with league inference, ~10k/s pure self-play, ~7.5k/s for
3×512 before league overhead. The engine does 13M raw steps/s per core (1.4–1.8M/s with observation encoding).

**AlphaZero** (64 sims, 256 games): ~500 searched moves/s with 2×256, ~300/s with 3×512; ~60% of the time is the
network, 40% the Rust search.

**GNN on CPU**: one forward costs 1.3 ms at batch 1 (dispatch overhead) and ~325 ms at batch 256 on one thread.
`strategy.py` on run 5 takes ~3.5 min per snapshot on 7 threads.

**Sharing.** Each heavy job spawns ~14 torch + ~14 rayon threads. Two at full width thrash: training fell from
7.3k to 1.2k samples/s next to a strategy analysis. Capping the second job to 4 threads
(`RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4`, `torch.set_num_threads(4)`) slowed training by only ~10%.
