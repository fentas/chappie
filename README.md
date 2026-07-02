# Chappie

A brain-inspired, multi-agent cognitive architecture in Rust. Many small
specialist agents ("100 or 1000 small models"), a router that wakes only the
ones a moment needs, a global workspace that builds consensus among them, and a
sleep cycle that consolidates experience into weights.

> **Status: v1 walking skeleton.** Every region runs end-to-end, deterministically,
> in pure `std` Rust (zero dependencies). The agents are cheap deterministic stubs
> today; each is a trait seam you can back with a real model (Ollama / candle /
> LoRA) without touching the loop. This is the "simulation-first" stage — prove
> the architecture cheaply, then drop real cognition in behind the interfaces.

## The loop

```
perceive → attend/route → deliberate (active experts) → consensus → act
         → record episode → [tired?] sleep → consolidate
```

## Regions (crates)

| crate | region | role |
|-------|--------|------|
| `chappie-core` | — | shared vocabulary: `Stimulus`, `Percept`, `Action`, `Episode`, `Embedding`, seedable `Rng`, the `Mind` trait |
| `chappie-harness` | thalamus + cortex runtime | **manages & connects X agents**: load/unload lifecycle (Cold/Warm/Active), sparse top-k activation + LRU eviction under a memory budget, a weighted `Connectome`, and consensus |
| `chappie-brain` | the brain | wires the regions into the loop: `Senses` · `Thalamus`(router) · `Cortex`(L/R hemispheres) · `Hippocampus`(episodic memory) · `Sleep`(consolidation + staged-LoRA hook) · `Vitals`(drives) |
| `chappie-world` | environment | a simulation Chappie lives in, with human-like life stages (Infancy → Adulthood) that widen the curriculum |
| `chappie-examiner` | feedback loop | a fixed benchmark battery run in pure-inference mode; the score stream is the learning curve and gates life-stage progression |
| `chappie-cli` | — | assembles a population, runs one deterministic life, prints the report |

## Design notes that map to the five pillars

- **Only active sections loaded.** The harness wakes the top-k *relevant* agents
  per tick (relevance = cosine to the fused query) and LRU-evicts the rest to fit
  a MB budget. A life with 16 agents (2.6 GB if all resident) peaks under ~1.3 GB.
- **Human senses + hemispheres.** Modality encoders (sight/sound/touch/smell/…)
  feed a thalamic router. Left = sequential/linguistic/analytic/exploit; Right =
  holistic/spatial/novelty/explore. The "corpus callosum" hands the lead to the
  right hemisphere on novelty, the left on the familiar. Two processing styles =
  built-in error de-correlation.
- **Consensus.** Active agents bid in a shared workspace over two debate rounds;
  bids reduce to one action by hemisphere-weighted vote.
- **Connected & weighted.** A `Connectome` strengthens links between agents that
  fire together and are rewarded (Hebbian), and gently biases future routing.
  *Bounded* on purpose — unbounded, one cluster hijacks every stimulus (the MoE
  router-collapse failure mode).
- **Sleep creates weights.** When energy runs low, sleep replays episodes:
  decays unused links, consolidates rewarded co-activations, distills semantic
  prototypes, and nudges each agent's adapter (the staged stand-in for a LoRA
  step over replayed experience).

## Run

```sh
mise run life                       # one deterministic life (seed 42, 4000 ticks)
# or
cargo run --release --bin chappie -- --seed 42 --ticks 4000
```

Flags: `--seed N` · `--ticks N` · `--budget MB` (footprint budget) · `--active K`
(max agents awake per tick). Same seed → same life.

## Roadmap (staged)

1. **v1 — skeleton (done).** Pure-std, deterministic, all regions live behind traits.
2. **Model-backed agents.** Implement `Agent` over Ollama (Gemma/Qwen small
   models) — language/reasoning agents become real. Backend is GPU-agnostic
   (Ollama drives the RX 9070 XT via ROCm).
3. **Real learning during sleep.** Swap the adapter stub for LoRA fine-tuning
   over replayed episodes; keep consolidation + routing-weight updates from v1.
4. **Richer worlds & benchmarks.** Grounded tasks, real curricula, held-out suites.

## Why Rust + pure std for v1

Fast compiles, no supply-chain surface, and a hard architectural discipline:
if a region can be a deterministic stub behind a trait, the *shape* is right
before any model is loaded. Real dependencies enter at stage 2, pinned to latest.
