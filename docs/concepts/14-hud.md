# 14 · The HUD — Chappie's inner life, visible

A live dashboard for a running mind. The sim emits a rich telemetry frame; a
self-contained page ([`docs/hud.html`](../hud.html)) renders it — no build step, no
dependencies, works offline.

## What it shows
- **Status** — tick, day, life-stage.
- **Vitals** — mood (diverging −1…+1), energy, curiosity, boredom, sleep-pressure.
- **Character** — the learned routing gate's disposition (*distress → …, curious → …,
  bored → …*), the personality that formed over this life.
- **Connectome** — the agents as a graph: left/right hemisphere split across the field,
  node size = reliability, colour = compute tier (gpu/cpu/cold), edges = Hebbian weight.
  You watch coalitions wire together.
- **Agents** — the roster by tier, each with its concept, reliability, and activation count.
- **Deep memories** — the gatekeeper's fast-lane engrams (fear vs. safe, by strength).
- **Stats** — population, growth/pruning, reflexes, thoughts, sleeps, avg reward.
- **History** — mood and reward sparklines.

## How to run it
```
mise run hud          # live: run endlessly, streaming docs/hud.json
mise run hud-serve    # (other terminal) serve docs/ at localhost:8008
# → open http://localhost:8008/hud.html
```
Or, for a **one-shot snapshot** with no server: run any life with `--hud <path>` (e.g.
`cargo run --bin chappie -- --seed 7 --ticks 12000 --hud frame.json`), open `docs/hud.html`,
and **drag the JSON onto the page**. A committed sample (`docs/hud.json`) is shown by default.

## How it works
`Brain::telemetry_json()` gathers the whole live state — vitals, the harness `roster()` and
`top_edges()`, the deep memories, the gate's disposition, and rolling mood/reward history —
into one JSON frame. The CLI's `--hud <path>` writes it every ~100 ticks (and at the end);
the page polls it every 1.5 s. Pure-std on the Rust side — just JSON out — so the HUD adds no
dependencies to the core.
