# 10 · Limits & Measurements

Real numbers, not spec sheets. Measured on the target box — AMD RX 9070 XT (RDNA4,
16 GB, wgpu/Vulkan via radv), 124 GB RAM, 24 cores — with the two profiling binaries
(`cargo run --bin measure` and `--bin measure_gpu`). Re-run them; the numbers are the
ground truth, this page is the interpretation.

## Cognitive-loop scaling (CPU, stub agents)
Budgets: GPU 600 MB, CPU 1200 MB, `max_participants=6`, 160 MB/agent.

| agents | µs/tick | ticks/sec | resident | connectome | proc RSS |
|---:|---:|---:|---:|---:|---:|
| 16 | 14.7 | 67,800 | 9 | 0.00 MB | 0.1 MB |
| 64 | 33.7 | 29,700 | 10 | 0.02 MB | 0.2 MB |
| 256 | 124 | 8,000 | 10 | 0.25 MB | 0.4 MB |
| 1024 | 569 | 1,760 | 10 | 4.0 MB | 2.6 MB |
| 4096 | 2598 | 385 | 10 | 64 MB | 20 MB |

**Resident stays flat at ~10 no matter the population** — sparse activation works: only
the attended few are ever loaded. Growing the population grows *latent* capacity, not the
active footprint.

## Where a tick actually goes (n=1024)
| stage | time | scaling |
|---|---:|---|
| schedule (rank *all* agents) | 556 µs | **O(n)** |
| deliberate (the "inference") | 5.4 µs | O(participants), constant |
| consensus | 1.8 µs | O(participants) |

**The bottleneck is the O(n) re-ranking, not the inference.** The expensive-sounding part
— agents thinking — is already cheap and *constant* in population size; 99% of the tick is
the scheduler scoring every agent from scratch each step. That's the thing to optimize
(incremental / approximate ranking, a spatial index over competencies, only re-rank on
attention shift) — not the agent count.

## The connectome is O(n²)
`grow()` and storage scale with the square: 4 MB at n=1024, **64 MB at n=4096** (235 ms to
resize). Fine to a few thousand agents; a **sparse** graph (only real edges) is required
beyond that. This — not the agents — is the first memory wall.

## GPU: transfer & compute (measured)
| device→host transfer | | matmul (fp32) | |
|---|---:|---|---:|
| 4 MB | 6.2 GB/s (0.6 ms) | 512² | 769 GFLOP/s |
| 64 MB | 22 GB/s (2.8 ms) | 1024² | 2,360 GFLOP/s |
| 256 MB | 32 GB/s (7.8 ms) | 2048² | 4,900 GFLOP/s |

So **a few-MB LoRA adapter moves in ~0.6 ms; a 256 MB model in ~8 ms.** This is the whole
argument for shared-backbone + adapters: keep the big base resident, hot-swap tiny adapters
sub-millisecond. Compute tops out near **~5 TFLOP/s fp32** here — plenty for small-model
inference; memory movement, not FLOPs, is the limit.

**Honest caveat — measured < spec.** 32 GB/s (vs PCIe 5.0's ~50–63 GB/s theoretical) and
~5 TFLOP/s reflect the wgpu/Vulkan/radv path in fp32 with Burn overhead, not the card's
ceiling. fp16 + a native stack (ROCm) would lift both. These are *today's real* numbers.

## The envelope
| resource | limit | holds |
|---|---|---|
| VRAM (hot) | 16 GB | ~3,000 LoRA adapters over one resident base |
| RAM (warm) | 124 GB | effectively the whole warm population |
| cold (NVMe) | disk | the near-infinite tail |
| swap | ~32 GB/s measured | a few hundred MB of hot-set churn per tick |
| coordination | O(n) rank/tick | **the real scaling cost** — optimize this first |

The binding constraints are (1) the O(n) scheduler and (2) the O(n²) connectome — both
*coordination*, not the agents. "1000 agents" is ~3× under the VRAM ceiling and runs at
~1,800 ticks/s today; the work to scale further is smarter routing and a sparse graph.
