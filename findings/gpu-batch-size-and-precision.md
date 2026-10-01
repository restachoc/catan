# T4: batch size helps the MLP, not the GNN; fp16 helps the GNN

*2026-10-01. Sweep on a free Colab T4 with the trainer's own networks. Raw table:
`benchmarks/results/batch_sweep_t4.csv` (local only).*

**MLP 2×256.** Inference saturates at ~4096 envs (650k/s), training at minibatch 8192 (1.1M/s). End to end with
the engine: ~140k samples/s at those sizes vs ~82k/s at the diagnostic recipe's 256 envs.

**GNN d64 L4.** Already saturated at 256 envs and minibatch 1024 (~27k/s inference, ~10k/s training, ~2.3k/s
end to end); larger batches change nothing. d128 being only ~2× slower than d64 (4× the FLOPs) points to memory
bandwidth rather than arithmetic as the limit.

- fp16 autocast: ~1.5× (3.5k/s end to end). It needs the policy mask applied in fp32, since `NEG_INF = -1e9`
  overflows fp16.
- `torch.compile`: ~1.4× (d64), ~1.2× (d128); compiling takes ~100 s and recompiles per batch shape. Not used.
- Training memory: ~0.75 GB per 1k minibatch; 16k fits in 15 GB, 32k runs out of memory.

**Colab CPU.** Free runtimes have 2 vCPUs. The engine does ~500k steps/s there up to 4096 envs, 230k/s at 16k.

**Implication.** For GNN comparisons keep the diagnostic recipe's batch sizes (they already saturate the T4) and
use `--device cuda --amp`. MLP runs on a GPU: `--num-envs 2048 --minibatch 8192`.
