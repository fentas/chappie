//! chappie-core — the shared vocabulary of the Chappie cognitive architecture.
//!
//! Pure `std`, fully deterministic, zero dependencies. Everything that flows
//! between brain regions is defined here: stimuli, percepts, actions, episodes,
//! embeddings, a seedable RNG, and the [`Mind`] trait the outside world drives.

// ============================================================================
// Deterministic RNG (SplitMix64) — reproducible "lives" from a single seed.
// ============================================================================

/// A small, fast, seedable PRNG. Determinism is a feature: same seed → same life.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform integer in `[0, n)` (returns 0 when `n == 0`).
    pub fn next_range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Cheap approx-Gaussian noise, mean 0, unit-ish variance.
    pub fn next_gauss(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += self.next_f32();
        }
        (s - 3.0) / 1.732
    }
}

// ============================================================================
// Concepts & embeddings — a small, legible latent space.
// ============================================================================

/// The fixed concept axes of the skeleton's latent space. Real encoders will
/// produce higher-dim opaque embeddings later; here the axes are human-readable
/// so routing, specialization, and consensus are all inspectable in the logs.
pub const CONCEPTS: [&str; 12] = [
    "visual", "auditory", "tactile", "olfactory", "gustatory", "language", "spatial",
    "logical", "numeric", "social", "danger", "reward",
];

pub const EMB_DIM: usize = 12;

/// Marker label for a "recall" cue: a stimulus asking the mind to report what it
/// just perceived, from short-term working memory. Fails until that memory exists.
pub const RECALL_CUE: &str = "__recall__";

/// A point in concept space.
pub type Embedding = Vec<f32>;

pub fn concept_index(name: &str) -> Option<usize> {
    CONCEPTS.iter().position(|&c| c == name)
}

/// Build a normalized embedding from `(concept, weight)` pairs.
pub fn embed(pairs: &[(&str, f32)]) -> Embedding {
    let mut v = vec![0.0f32; EMB_DIM];
    for &(name, w) in pairs {
        if let Some(i) = concept_index(name) {
            v[i] += w;
        }
    }
    normalize(&mut v);
    v
}

pub fn normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Index of the largest component (0 if empty).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

// ============================================================================
// Senses — "orient from the human example."
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Modality {
    Sight,
    Sound,
    Touch,
    Smell,
    Taste,
    Language,
    Interoception,
}

impl Modality {
    pub fn name(&self) -> &'static str {
        match self {
            Modality::Sight => "sight",
            Modality::Sound => "sound",
            Modality::Touch => "touch",
            Modality::Smell => "smell",
            Modality::Taste => "taste",
            Modality::Language => "language",
            Modality::Interoception => "interoception",
        }
    }
}

/// Raw input from the world, in concept space (a stand-in for pixels/audio/text).
#[derive(Clone, Debug)]
pub struct Stimulus {
    pub modality: Modality,
    pub label: String,
    pub features: Embedding,
    pub intensity: f32,
}

/// A stimulus after a sense encoder: an embedding plus how much it grabs attention.
#[derive(Clone, Debug)]
pub struct Percept {
    pub modality: Modality,
    pub label: String,
    pub embedding: Embedding,
    pub salience: f32,
}

// ============================================================================
// Hemispheres — two institutionalized processing styles (built-in diversity).
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Hemisphere {
    /// Sequential · linguistic · analytic · exploit-known.
    Left,
    /// Holistic · spatial · novelty-seeking · explore.
    Right,
}

impl Hemisphere {
    pub fn tag(&self) -> &'static str {
        match self {
            Hemisphere::Left => "L",
            Hemisphere::Right => "R",
        }
    }
}

// ============================================================================
// Actions & proposals — the output side and the deliberation currency.
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ActionKind {
    Speak,
    Move,
    Manipulate,
    Attend,
    Rest,
    Noop,
}

impl ActionKind {
    pub fn name(&self) -> &'static str {
        match self {
            ActionKind::Speak => "speak",
            ActionKind::Move => "move",
            ActionKind::Manipulate => "manipulate",
            ActionKind::Attend => "attend",
            ActionKind::Rest => "rest",
            ActionKind::Noop => "noop",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Action {
    pub kind: ActionKind,
    pub utterance: String,
    pub target: Embedding,
    pub confidence: f32,
}

impl Action {
    pub fn noop() -> Self {
        Action {
            kind: ActionKind::Noop,
            utterance: String::new(),
            target: vec![0.0; EMB_DIM],
            confidence: 0.0,
        }
    }
}

pub type AgentId = u32;

/// One agent's bid during deliberation in the global workspace.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub agent: AgentId,
    pub agent_name: String,
    pub hemisphere: Hemisphere,
    pub action: Action,
    pub weight: f32,
    pub rationale: String,
}

// ============================================================================
// Episodes — the substrate the hippocampus stores and sleep consolidates.
// ============================================================================

#[derive(Clone, Debug)]
pub struct Episode {
    pub tick: u64,
    pub stage: String,
    pub query: Embedding,
    pub dominant: String,
    pub decision: Action,
    pub active_agents: Vec<AgentId>,
    pub reward: f32,
    pub surprise: f32,
}

// ============================================================================
// Mind — the interface the World and Examiner drive. Brains are pluggable.
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct MindStats {
    pub tick: u64,
    pub day: u64,
    pub stage: String,
    pub energy: f32,
    pub curiosity: f32,
    pub agents_total: usize,
    pub gpu_count: usize,
    pub cpu_count: usize,
    pub cold_count: usize,
    pub gpu_mb: f32,
    pub cpu_mb: f32,
    pub peak_gpu_mb: f32,
    pub peak_cpu_mb: f32,
    pub gpu_budget: f32,
    pub cpu_budget: f32,
    pub total_mb: f32,
    pub sleeps: u64,
    pub avg_reward: f32,
    /// How many decisions this life required escalating from reflex to "thinking".
    pub thinks: u64,
}

// ============================================================================
// Placement — where an agent runs (the attention / compute hierarchy).
// ============================================================================

/// The compute tier an agent occupies, re-decided every tick from priority:
/// attended agents go **hot** (GPU), the working set stays **warm** (CPU/RAM),
/// the long tail is **cold** (unloaded). This is how "100–1000 agents" scales.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Placement {
    #[default]
    Cold,
    Cpu,
    Gpu,
}

impl Placement {
    pub fn tag(&self) -> &'static str {
        match self {
            Placement::Cold => "cold",
            Placement::Cpu => "cpu",
            Placement::Gpu => "gpu",
        }
    }
}

// ============================================================================
// Config — the refined, tunable model. Every knob lives here, grouped by
// subsystem, so runs are reproducible and benchmarks reference an exact config.
// ============================================================================

/// Thalamic attention gate.
#[derive(Clone, Debug)]
pub struct AttentionCfg {
    /// Percepts below this salience are ignored.
    pub floor: f32,
}
impl Default for AttentionCfg {
    fn default() -> Self {
        Self { floor: 0.15 }
    }
}

/// Left/Right arbitration during deliberation.
#[derive(Clone, Debug)]
pub struct HemisphereCfg {
    pub lead_gain: f32,
    pub follow_gain: f32,
    pub curiosity_boost: f32,
    /// surprise+curiosity above this hands the lead to the Right hemisphere.
    pub novelty_threshold: f32,
}
impl Default for HemisphereCfg {
    fn default() -> Self {
        Self { lead_gain: 1.15, follow_gain: 0.9, curiosity_boost: 0.5, novelty_threshold: 0.6 }
    }
}

/// The placement scheduler (GPU/CPU/cold tiering).
#[derive(Clone, Debug)]
pub struct PriorityCfg {
    pub w_relevance: f32,
    /// weight on connectome coupling to currently-hot agents (shared priority).
    pub w_shared: f32,
    pub w_reliability: f32,
    /// bonus for staying where you are (anti-thrash).
    pub hysteresis: f32,
    /// below this priority an agent goes cold regardless of budget.
    pub floor: f32,
    /// minimum priority to join deliberation.
    pub participate_floor: f32,
    /// down-weight applied to proposals from warm (CPU) agents.
    pub cpu_penalty: f32,
}
impl Default for PriorityCfg {
    fn default() -> Self {
        Self {
            w_relevance: 1.0,
            w_shared: 0.10,
            w_reliability: 0.10,
            hysteresis: 0.15,
            floor: 0.20,
            participate_floor: 0.20,
            cpu_penalty: 0.10,
        }
    }
}

/// "Fire together, wire together" dynamics.
#[derive(Clone, Debug)]
pub struct HebbianCfg {
    pub online_rate: f32,
    pub sleep_rate: f32,
    pub decay: f32,
    pub max_weight: f32,
}
impl Default for HebbianCfg {
    fn default() -> Self {
        Self { online_rate: 0.06, sleep_rate: 0.03, decay: 0.97, max_weight: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct SleepCfg {
    pub replay_cap: usize,
}
impl Default for SleepCfg {
    fn default() -> Self {
        Self { replay_cap: 256 }
    }
}

/// Homeostatic drives.
#[derive(Clone, Debug)]
pub struct VitalsCfg {
    pub energy_cost_base: f32,
    pub energy_cost_per_agent: f32,
    pub tired_threshold: f32,
    pub curiosity_gain: f32,
    pub curiosity_reward_decay: f32,
}
impl Default for VitalsCfg {
    fn default() -> Self {
        Self {
            energy_cost_base: 0.02,
            energy_cost_per_agent: 0.008,
            tired_threshold: 0.2,
            curiosity_gain: 0.1,
            curiosity_reward_decay: 0.15,
        }
    }
}

/// Hardware footprint budgets for the placement tiers.
#[derive(Clone, Debug)]
pub struct BudgetCfg {
    pub gpu_mb: f32,
    pub cpu_mb: f32,
    pub max_participants: usize,
}
impl Default for BudgetCfg {
    fn default() -> Self {
        Self { gpu_mb: 600.0, cpu_mb: 1200.0, max_participants: 6 }
    }
}

/// Dual-process control: when consensus is conflicted (low agreement), stop
/// reflexing and "think" — widen the coalition and re-deliberate.
#[derive(Clone, Debug)]
pub struct ThinkingCfg {
    /// Below this consensus agreement, escalate to thinking.
    pub agreement_threshold: f32,
    /// Max escalation rounds before committing anyway.
    pub max_escalations: usize,
    /// Extra resident agents to pull into deliberation per escalation.
    pub widen_participants: usize,
    /// Multiplier that lowers the participation floor while thinking.
    pub widen_floor_mult: f32,
}
impl Default for ThinkingCfg {
    fn default() -> Self {
        Self { agreement_threshold: 0.65, max_escalations: 2, widen_participants: 4, widen_floor_mult: 0.5 }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub ticks: usize,
    pub propose_threshold: f32,
    pub attention: AttentionCfg,
    pub hemisphere: HemisphereCfg,
    pub priority: PriorityCfg,
    pub hebbian: HebbianCfg,
    pub sleep: SleepCfg,
    pub vitals: VitalsCfg,
    pub budget: BudgetCfg,
    pub thinking: ThinkingCfg,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 42,
            ticks: 4000,
            propose_threshold: 0.25,
            attention: AttentionCfg::default(),
            hemisphere: HemisphereCfg::default(),
            priority: PriorityCfg::default(),
            hebbian: HebbianCfg::default(),
            sleep: SleepCfg::default(),
            vitals: VitalsCfg::default(),
            budget: BudgetCfg::default(),
            thinking: ThinkingCfg::default(),
        }
    }
}

impl Config {
    /// Override one setting by dotted key (e.g. `budget.gpu_mb`, `hebbian.online_rate`).
    /// Returns false for an unknown key or unparseable value. Powers `--set k=v`
    /// flags and simple `key = value` config files — the fine-adjustment surface.
    pub fn set(&mut self, key: &str, val: &str) -> bool {
        macro_rules! pf {
            () => {
                match val.parse() {
                    Ok(v) => v,
                    Err(_) => return false,
                }
            };
        }
        match key {
            "seed" => self.seed = pf!(),
            "ticks" => self.ticks = pf!(),
            "propose_threshold" => self.propose_threshold = pf!(),
            "attention.floor" => self.attention.floor = pf!(),
            "hemisphere.lead_gain" => self.hemisphere.lead_gain = pf!(),
            "hemisphere.follow_gain" => self.hemisphere.follow_gain = pf!(),
            "hemisphere.curiosity_boost" => self.hemisphere.curiosity_boost = pf!(),
            "hemisphere.novelty_threshold" => self.hemisphere.novelty_threshold = pf!(),
            "priority.w_relevance" => self.priority.w_relevance = pf!(),
            "priority.w_shared" => self.priority.w_shared = pf!(),
            "priority.w_reliability" => self.priority.w_reliability = pf!(),
            "priority.hysteresis" => self.priority.hysteresis = pf!(),
            "priority.floor" => self.priority.floor = pf!(),
            "priority.participate_floor" => self.priority.participate_floor = pf!(),
            "priority.cpu_penalty" => self.priority.cpu_penalty = pf!(),
            "hebbian.online_rate" => self.hebbian.online_rate = pf!(),
            "hebbian.sleep_rate" => self.hebbian.sleep_rate = pf!(),
            "hebbian.decay" => self.hebbian.decay = pf!(),
            "hebbian.max_weight" => self.hebbian.max_weight = pf!(),
            "sleep.replay_cap" => self.sleep.replay_cap = pf!(),
            "vitals.energy_cost_base" => self.vitals.energy_cost_base = pf!(),
            "vitals.energy_cost_per_agent" => self.vitals.energy_cost_per_agent = pf!(),
            "vitals.tired_threshold" => self.vitals.tired_threshold = pf!(),
            "vitals.curiosity_gain" => self.vitals.curiosity_gain = pf!(),
            "vitals.curiosity_reward_decay" => self.vitals.curiosity_reward_decay = pf!(),
            "budget.gpu_mb" => self.budget.gpu_mb = pf!(),
            "budget.cpu_mb" => self.budget.cpu_mb = pf!(),
            "budget.max_participants" => self.budget.max_participants = pf!(),
            "thinking.agreement_threshold" => self.thinking.agreement_threshold = pf!(),
            "thinking.max_escalations" => self.thinking.max_escalations = pf!(),
            "thinking.widen_participants" => self.thinking.widen_participants = pf!(),
            "thinking.widen_floor_mult" => self.thinking.widen_floor_mult = pf!(),
            _ => return false,
        }
        true
    }
}

/// What a sleep cycle produced — surfaced so a life is legible.
#[derive(Clone, Debug, Default)]
pub struct DreamLog {
    pub day: u64,
    pub replayed: usize,
    pub strengthened: Vec<(String, String, f32)>,
    pub new_prototypes: usize,
    pub note: String,
    /// Mean reward over the day just ended (for the diary).
    pub day_reward: f32,
    /// What was attended to today: (concept, count), sorted desc.
    pub concept_counts: Vec<(String, u32)>,
    /// The goal/task in effect this day, if any.
    pub goal: Option<String>,
}

/// A cognitive agent the environment can drive. Implemented by `chappie-brain`.
pub trait Mind {
    /// Perceive a scene and commit to an action. `learn=false` = pure inference
    /// (used by the Examiner) — no memory writes, no weight updates.
    fn perceive_act(&mut self, stimuli: &[Stimulus], learn: bool) -> Action;
    /// Deliver the world's reward for the most recent action.
    fn reward(&mut self, r: f32);
    /// Is the agent out of energy and due for a sleep cycle?
    fn tired(&self) -> bool;
    /// Run a sleep/consolidation cycle; returns what it changed.
    fn sleep(&mut self) -> DreamLog;
    fn stats(&self) -> MindStats;
}
