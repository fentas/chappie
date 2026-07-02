//! chappie-harness — the connective tissue that manages and connects X agents.
//!
//! Two jobs:
//!   1. **Attention-driven placement.** Every tick a scheduler ranks all agents
//!      by *priority* (relevance + connectome coupling to whatever's currently
//!      hot + learned reliability + hysteresis) and places the top ones on the
//!      GPU (hot), the next on CPU (warm), and unloads the rest (cold), under
//!      separate GPU/CPU footprint budgets. This is how "100–1000 agents" fits:
//!      only the attended few are ever hot. Coupling to hot agents is the
//!      "shared priority" — a wired-together circuit gets promoted as a coalition.
//!   2. **Deliberation + consensus.** Resident, relevant agents bid in a shared
//!      workspace over two rounds; bids reduce to one action by weighted vote.
//!
//! An `Agent` is abstract: a `StubAgent` today, a Burn/Ollama model tomorrow.
//! When placement changes, the harness calls `Agent::on_placement`, so a real
//! model can move itself GPU↔CPU or unload.

use chappie_core::*;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Agent — the unit the harness manages.
// ============================================================================

/// Everything an active agent sees during deliberation.
pub struct Context<'a> {
    pub query: &'a Embedding,
    pub dominant: usize,
    pub lead: Hemisphere,
    pub curiosity: f32,
    pub prior: &'a [Proposal],
    // config knobs injected per tick
    pub lead_gain: f32,
    pub follow_gain: f32,
    pub curiosity_boost: f32,
    pub propose_threshold: f32,
}

pub trait Agent {
    fn id(&self) -> AgentId;
    fn name(&self) -> &str;
    fn hemisphere(&self) -> Hemisphere;
    /// Embedding describing what this agent is good at — used for routing.
    fn competency(&self) -> &Embedding;
    /// Memory cost when resident, in MB — drives the placement budgets.
    fn footprint_mb(&self) -> f32;
    fn reliability(&self) -> f32;

    /// Called when the scheduler moves this agent between tiers (incl. to/from
    /// cold). Stubs just record it; a real model moves weights GPU↔CPU or frees.
    fn on_placement(&mut self, tier: Placement);

    /// Produce a bid for the current moment.
    fn think(&mut self, ctx: &Context, rng: &mut Rng) -> Proposal;
    /// Online update after the world responds (fast, per-tick).
    fn reinforce(&mut self, reward: f32, was_winner: bool);
    /// Offline update during sleep (slow, staged-LoRA hook).
    fn consolidate(&mut self, episodes: &[Episode], rng: &mut Rng);

    /// Optional introspection: the concept this agent would name for a query.
    /// Model-backed agents (the parametric long-term memory) override this so
    /// their learning can be probed; others return `None`.
    fn predict_concept(&self, _query: &Embedding) -> Option<usize> {
        None
    }
}

// ============================================================================
// StubAgent — a data-driven policy standing in for a real small model.
// ============================================================================

pub struct StubAgent {
    id: AgentId,
    name: String,
    hemisphere: Hemisphere,
    competency: Embedding,
    kind: ActionKind,
    utterance: String,
    footprint_mb: f32,
    reliability: f32,
    /// Scalar gain adjusted during sleep — the staged stand-in for a LoRA adapter.
    adapter: f32,
    placement: Placement,
}

impl StubAgent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: AgentId,
        name: impl Into<String>,
        hemisphere: Hemisphere,
        competency: Embedding,
        kind: ActionKind,
        utterance: impl Into<String>,
        footprint_mb: f32,
    ) -> Self {
        StubAgent {
            id,
            name: name.into(),
            hemisphere,
            competency,
            kind,
            utterance: utterance.into(),
            footprint_mb,
            reliability: 0.5,
            adapter: 0.0,
            placement: Placement::Cold,
        }
    }

    pub fn boxed(self) -> Box<dyn Agent> {
        Box::new(self)
    }
}

impl Agent for StubAgent {
    fn id(&self) -> AgentId {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn hemisphere(&self) -> Hemisphere {
        self.hemisphere
    }
    fn competency(&self) -> &Embedding {
        &self.competency
    }
    fn footprint_mb(&self) -> f32 {
        self.footprint_mb
    }
    fn reliability(&self) -> f32 {
        self.reliability
    }

    fn on_placement(&mut self, tier: Placement) {
        self.placement = tier;
    }

    fn think(&mut self, ctx: &Context, rng: &mut Rng) -> Proposal {
        let align = cosine(ctx.query, &self.competency).max(0.0);
        let mut strength = align * (0.5 + self.reliability) * (1.0 + self.adapter);

        if self.hemisphere == ctx.lead {
            strength *= ctx.lead_gain;
        } else {
            strength *= ctx.follow_gain;
        }
        if self.hemisphere == Hemisphere::Right {
            strength *= 1.0 + ctx.curiosity_boost * ctx.curiosity;
        }
        for p in ctx.prior {
            if p.action.kind == self.kind && p.agent != self.id {
                strength *= 1.05;
                break;
            }
        }
        strength *= 0.95 + 0.1 * rng.next_f32();

        let (kind, utter) = if strength > ctx.propose_threshold {
            (self.kind, self.utterance.clone())
        } else {
            (ActionKind::Attend, String::new())
        };

        Proposal {
            agent: self.id,
            agent_name: self.name.clone(),
            hemisphere: self.hemisphere,
            action: Action {
                kind,
                utterance: utter,
                target: self.competency.clone(),
                confidence: strength.min(1.0),
            },
            weight: strength.max(0.0),
            rationale: format!("align={:.2} rel={:.2}", align, self.reliability),
        }
    }

    fn reinforce(&mut self, reward: f32, was_winner: bool) {
        let lr = if was_winner { 0.05 } else { 0.01 };
        self.reliability = (self.reliability + lr * reward).clamp(0.0, 1.5);
    }

    fn consolidate(&mut self, episodes: &[Episode], _rng: &mut Rng) {
        let mut sum = 0.0f32;
        let mut n = 0u32;
        for e in episodes {
            if e.active_agents.contains(&self.id) {
                sum += e.reward;
                n += 1;
            }
        }
        if n > 0 {
            let avg = sum / n as f32;
            self.adapter = (self.adapter + 0.02 * avg).clamp(-0.5, 0.5);
        }
    }
}

// ============================================================================
// Connectome — weighted graph linking agents that fire together.
// ============================================================================

pub struct Connectome {
    n: usize,
    w: Vec<f32>,
    max: f32,
}

impl Connectome {
    pub fn new(n: usize, max: f32) -> Self {
        Connectome { n, w: vec![0.0; n * n], max }
    }

    pub fn weight(&self, a: usize, b: usize) -> f32 {
        self.w[a * self.n + b]
    }

    pub fn strengthen(&mut self, a: usize, b: usize, d: f32) {
        if a == b || a >= self.n || b >= self.n {
            return;
        }
        let na = (self.w[a * self.n + b] + d).clamp(0.0, self.max);
        self.w[a * self.n + b] = na;
        self.w[b * self.n + a] = na;
    }

    pub fn decay(&mut self, factor: f32) {
        for x in self.w.iter_mut() {
            *x *= factor;
        }
    }

    /// How strongly `a` is wired to a set of agents (used for shared priority).
    pub fn bias_for(&self, a: usize, group: &[AgentId]) -> f32 {
        group.iter().map(|&b| self.weight(a, b as usize)).sum()
    }
}

// ============================================================================
// Harness — registry + placement scheduler + deliberation + consensus.
// ============================================================================

struct Slot {
    agent: Box<dyn Agent>,
    placement: Placement,
    priority: f32,
    last_used: u64,
    activations: u64,
}

pub struct Decision {
    pub action: Action,
    pub agreement: f32,
    pub winners: Vec<AgentId>,
}

pub struct Harness {
    slots: Vec<Slot>,
    connectome: Connectome,
    cfg: Config,
    gpu_mb: f32,
    cpu_mb: f32,
    peak_gpu_mb: f32,
    peak_cpu_mb: f32,
    total_mb: f32,
    tick: u64,
    active: Vec<AgentId>,
    last_active: Vec<AgentId>,
    hot: Vec<AgentId>,
}

impl Harness {
    /// Build a harness over a population. Agent ids must be `0..agents.len()`.
    pub fn new(agents: Vec<Box<dyn Agent>>, cfg: &Config) -> Self {
        let n = agents.len();
        let total_mb = agents.iter().map(|a| a.footprint_mb()).sum();
        let slots = agents
            .into_iter()
            .map(|agent| Slot {
                agent,
                placement: Placement::Cold,
                priority: 0.0,
                last_used: 0,
                activations: 0,
            })
            .collect();
        Harness {
            slots,
            connectome: Connectome::new(n, cfg.hebbian.max_weight),
            cfg: cfg.clone(),
            gpu_mb: 0.0,
            cpu_mb: 0.0,
            peak_gpu_mb: 0.0,
            peak_cpu_mb: 0.0,
            total_mb,
            tick: 0,
            active: Vec::new(),
            last_active: Vec::new(),
            hot: Vec::new(),
        }
    }

    /// Rank all agents by priority and assign compute tiers under the budgets.
    /// Returns the participants (resident + relevant) that will deliberate.
    pub fn schedule(&mut self, query: &Embedding, rng: &mut Rng) -> Vec<AgentId> {
        let pc = self.cfg.priority.clone();
        let hot = self.hot.clone();
        let hot_denom = hot.len().max(1) as f32;

        // Pass 1 — priority for every agent (relevance + shared + reliability + hysteresis).
        let mut scored: Vec<(AgentId, f32)> = self
            .slots
            .iter()
            .map(|s| {
                let a = s.agent.id();
                let mut pri = pc.w_relevance * cosine(query, s.agent.competency());
                pri += pc.w_shared * self.connectome.bias_for(a as usize, &hot) / hot_denom;
                pri += pc.w_reliability * s.agent.reliability();
                pri += match s.placement {
                    Placement::Gpu => pc.hysteresis,
                    Placement::Cpu => pc.hysteresis * 0.5,
                    Placement::Cold => 0.0,
                };
                pri += 0.02 * rng.next_gauss();
                (a, pri)
            })
            .collect();
        scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

        // Pass 2 — greedy tier assignment: fill GPU first, then CPU, by priority.
        let n = self.slots.len();
        let mut new_place = vec![Placement::Cold; n];
        let mut gpu_mb = 0.0f32;
        let mut cpu_mb = 0.0f32;
        for &(id, pri) in &scored {
            if pri < pc.floor {
                continue;
            }
            let fp = self.slots[id as usize].agent.footprint_mb();
            if gpu_mb + fp <= self.cfg.budget.gpu_mb {
                new_place[id as usize] = Placement::Gpu;
                gpu_mb += fp;
            } else if cpu_mb + fp <= self.cfg.budget.cpu_mb {
                new_place[id as usize] = Placement::Cpu;
                cpu_mb += fp;
            }
        }

        // Apply — record priority, fire on_placement for changes, update accounting.
        for &(id, pri) in &scored {
            self.slots[id as usize].priority = pri;
        }
        for i in 0..n {
            let neu = new_place[i];
            if self.slots[i].placement != neu {
                self.slots[i].agent.on_placement(neu);
                self.slots[i].placement = neu;
            }
            if neu != Placement::Cold {
                self.slots[i].last_used = self.tick;
            }
        }
        self.gpu_mb = gpu_mb;
        self.cpu_mb = cpu_mb;
        self.peak_gpu_mb = self.peak_gpu_mb.max(gpu_mb);
        self.peak_cpu_mb = self.peak_cpu_mb.max(cpu_mb);

        // Participants — resident + relevant, capped, highest priority first.
        let participants: Vec<AgentId> = scored
            .iter()
            .filter(|&&(id, pri)| {
                new_place[id as usize] != Placement::Cold && pri > pc.participate_floor
            })
            .take(self.cfg.budget.max_participants)
            .map(|&(id, _)| id)
            .collect();

        for &id in &participants {
            self.slots[id as usize].activations += 1;
        }
        self.active = participants.clone();
        self.hot = (0..n)
            .filter(|&i| self.slots[i].placement == Placement::Gpu)
            .map(|i| i as AgentId)
            .collect();
        participants
    }

    /// Two-round deliberation among participants; warm (CPU) bids are down-weighted.
    pub fn deliberate(
        &mut self,
        query: &Embedding,
        dominant: usize,
        lead: Hemisphere,
        curiosity: f32,
        rng: &mut Rng,
    ) -> Vec<Proposal> {
        let ids = self.active.clone();
        let h = self.cfg.hemisphere.clone();
        let thr = self.cfg.propose_threshold;
        let cpu_pen = self.cfg.priority.cpu_penalty;

        let think_round = |slots: &mut [Slot], prior: &[Proposal], rng: &mut Rng| -> Vec<Proposal> {
            let mut out = Vec::with_capacity(ids.len());
            for &id in &ids {
                let ctx = Context {
                    query,
                    dominant,
                    lead,
                    curiosity,
                    prior,
                    lead_gain: h.lead_gain,
                    follow_gain: h.follow_gain,
                    curiosity_boost: h.curiosity_boost,
                    propose_threshold: thr,
                };
                let place = slots[id as usize].placement;
                let mut p = slots[id as usize].agent.think(&ctx, rng);
                if place == Placement::Cpu {
                    p.weight *= 1.0 - cpu_pen;
                }
                out.push(p);
            }
            out
        };

        let round1 = think_round(&mut self.slots, &[], rng);
        think_round(&mut self.slots, &round1, rng)
    }

    /// Reduce proposals to a single action by weighted vote.
    pub fn consensus(&self, proposals: &[Proposal]) -> Decision {
        let mut tally: HashMap<ActionKind, f32> = HashMap::new();
        for p in proposals {
            *tally.entry(p.action.kind).or_insert(0.0) += p.weight;
        }
        let total: f32 = tally.values().sum();
        let best_kind = tally
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k)
            .unwrap_or(ActionKind::Noop);

        let mut best_p: Option<&Proposal> = None;
        let mut winners = Vec::new();
        let mut win_weight = 0.0f32;
        for p in proposals {
            if p.action.kind == best_kind {
                winners.push(p.agent);
                win_weight += p.weight;
                if best_p.map_or(true, |b| p.weight > b.weight) {
                    best_p = Some(p);
                }
            }
        }

        let agreement = if total > 1e-6 { win_weight / total } else { 0.0 };
        let action = best_p
            .map(|p| Action {
                confidence: agreement,
                ..p.action.clone()
            })
            .unwrap_or_else(Action::noop);

        Decision {
            action,
            agreement,
            winners,
        }
    }

    /// Online learning: winners strengthen mutual wiring; actives update reliability.
    pub fn reinforce(&mut self, winners: &[AgentId], active: &[AgentId], reward: f32) {
        let rate = self.cfg.hebbian.online_rate;
        if reward > 0.0 {
            for i in 0..winners.len() {
                for j in (i + 1)..winners.len() {
                    self.connectome
                        .strengthen(winners[i] as usize, winners[j] as usize, rate * reward);
                }
            }
        }
        let winset: HashSet<AgentId> = winners.iter().copied().collect();
        for &id in active {
            let was = winset.contains(&id);
            self.slots[id as usize].agent.reinforce(reward, was);
        }
    }

    /// End the tick: remember participants for continuity; placement persists
    /// (the scheduler re-decides next tick — attention naturally drifts).
    pub fn end_tick(&mut self) {
        self.last_active = self.active.clone();
        self.active.clear();
        self.tick += 1;
    }

    /// Sleep-time consolidation: decay, then reinforce rewarded co-activations,
    /// then let each agent update its own weights (adapter / LoRA step).
    pub fn consolidate(&mut self, episodes: &[Episode], rng: &mut Rng) {
        self.connectome.decay(self.cfg.hebbian.decay);
        let rate = self.cfg.hebbian.sleep_rate;
        for e in episodes {
            if e.reward > 0.0 {
                let a = &e.active_agents;
                for i in 0..a.len() {
                    for j in (i + 1)..a.len() {
                        self.connectome
                            .strengthen(a[i] as usize, a[j] as usize, rate * e.reward);
                    }
                }
            }
        }
        for s in self.slots.iter_mut() {
            s.agent.consolidate(episodes, rng);
        }
    }

    // ---- introspection ----------------------------------------------------

    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    pub fn name(&self, id: AgentId) -> &str {
        self.slots[id as usize].agent.name()
    }
    pub fn placement_of(&self, id: AgentId) -> Placement {
        self.slots[id as usize].placement
    }
    pub fn id_of(&self, name: &str) -> Option<AgentId> {
        self.slots.iter().find(|s| s.agent.name() == name).map(|s| s.agent.id())
    }
    /// Probe an agent's concept prediction (None unless it's model-backed).
    pub fn probe(&self, id: AgentId, query: &Embedding) -> Option<usize> {
        self.slots.get(id as usize).and_then(|s| s.agent.predict_concept(query))
    }
    pub fn gpu_mb(&self) -> f32 {
        self.gpu_mb
    }
    pub fn cpu_mb(&self) -> f32 {
        self.cpu_mb
    }
    pub fn peak_gpu_mb(&self) -> f32 {
        self.peak_gpu_mb
    }
    pub fn peak_cpu_mb(&self) -> f32 {
        self.peak_cpu_mb
    }
    pub fn total_mb(&self) -> f32 {
        self.total_mb
    }
    pub fn gpu_budget(&self) -> f32 {
        self.cfg.budget.gpu_mb
    }
    pub fn cpu_budget(&self) -> f32 {
        self.cfg.budget.cpu_mb
    }
    pub fn count_tier(&self, t: Placement) -> usize {
        self.slots.iter().filter(|s| s.placement == t).count()
    }
    pub fn tier_names(&self, t: Placement) -> Vec<String> {
        self.slots
            .iter()
            .filter(|s| s.placement == t)
            .map(|s| s.agent.name().to_string())
            .collect()
    }

    pub fn top_reliability(&self, k: usize) -> Vec<(String, f32, u64)> {
        let mut v: Vec<(String, f32, u64)> = self
            .slots
            .iter()
            .map(|s| (s.agent.name().to_string(), s.agent.reliability(), s.activations))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(k);
        v
    }

    pub fn top_edges(&self, k: usize) -> Vec<(String, String, f32)> {
        let n = self.slots.len();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let w = self.connectome.weight(i, j);
                if w > 0.01 {
                    edges.push((
                        self.slots[i].agent.name().to_string(),
                        self.slots[j].agent.name().to_string(),
                        w,
                    ));
                }
            }
        }
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        edges.truncate(k);
        edges
    }
}
