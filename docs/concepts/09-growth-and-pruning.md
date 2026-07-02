# 9 · Growth & Pruning

The substrate is not fixed at birth. Agents and connections **grow** when the world
demands it and are **pruned** when they aren't used — the neural counterpart to growing
*senses* over development ([06](06-development.md)).

## One dial: limit ⇄ grow
- **Limit** — the footprint caps already do this: `budget.gpu_mb`, `budget.cpu_mb`,
  `budget.max_participants`, plus the memory caps. Set `growth.enabled = false` and the
  population is frozen; the budgets bound how much is ever resident.
- **Grow** — set `growth.enabled = true` and the *population* grows over life, up to
  `growth.max_agents`. The budgets don't change — they stay the **residency cap**. So
  growth adds *latent* capacity (more specialists exist), while attention + the budget
  still decide how few are hot at once. You don't start with 1000 agents; you **grow into**
  however many the problems you actually meet require.

## Growth is need-driven (neurogenesis)
Chappie doesn't add agents on a timer — it adds them where it **keeps failing on something
no one covers**. Each moment of low consensus agreement or negative valence deposits a
little "gap" against its dominant concept. In sleep, if a concept's gap is high *and* no
existing agent has real competency there *and* the population is below the cap, Chappie
**recruits a specialist** for that niche (a new agent whose competency is centred on it).
Growth follows unmet need.

Developmentally this is the point: an **infant starts with core senses**, and **recruits
the abstract faculties** — logic, number, social reasoning — *as it first meets them* in
later life-stages. The benchmark on those concepts starts low and climbs the moment the
specialist is grown.

## Pruning is the other half (synaptic exuberance → cull)
Real brains over-produce and then cut back. So agents that go **unused** (no recent
participation) *and* never earned trust (low reliability) are **culled** — their footprint
freed, their connectome edges left to decay. Growth without pruning only bloats; the pair
is what makes it *develop* rather than merely accumulate. (Weak connectome edges already
fade every sleep — that's connection-level pruning; culling agents is the neuron-level one.)

## What it touches
- `Connectome::grow` — the weighted graph resizes as agents are added (new row/col, zeros;
  existing weights preserved).
- `Harness::add_agent` / prune — the population is dynamic; pruned agents become inert
  (kept by id, never scheduled) so ids stay stable.
- The growth **policy** lives in the sleep cycle, where the day's gaps are already being
  reconciled — recruiting and pruning are just two more things sleep does.

## Config
```
growth.enabled           # false → frozen population (pure limit)
growth.max_agents        # the ceiling
growth.recruit_gap       # accumulated conflict in a niche before a specialist is grown
growth.recruit_coverage  # only recruit if the best existing competency is below this
growth.prune_idle        # idle span before an unused agent is culled
growth.prune_reliability # only cull agents below this reliability
```
`enabled=false` + flat budgets → a hard-limited brain. `enabled=true` + a cap → it grows
toward the ceiling, driven by need, pruned of what it doesn't use.
