//! chappie-burn — Burn-backed neural agents (the long-term *parametric* memory).
//!
//! A [`BurnAgent`] is a real, trainable network behind `chappie_harness::Agent`:
//!   * `think` runs a forward pass (query embedding → concept logits) on the GPU
//!     (wgpu/Vulkan);
//!   * `consolidate` (called during sleep) **trains its own weights** on the
//!     replayed episodic buffer via Burn autodiff — reward-filtered supervised
//!     learning. This is the neocortical half of Complementary Learning Systems:
//!     the fast episodic *heap* is replayed to slowly improve the *weights*.
//!
//! Because it starts ignorant and learns, it is the piece that makes the
//! benchmark curve actually climb. The cognitive loop is unchanged.
//!
//! Kept in a separate, opt-in crate so the pure-std core stays fast to build.

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};

use chappie_core::*;
use chappie_harness::{Agent, Context};

/// Compute backend. Flip `Back`/`Dev` between `NdArray` (CPU) and `Wgpu`
/// (GPU/Vulkan) here; `Autodiff<_>` adds the training graph on top.
type Back = NdArray;
type AD = Autodiff<Back>;
type Dev = burn::backend::ndarray::NdArrayDevice;

// ============================================================================
// The network — a tiny MLP: query embedding → concept logits.
// ============================================================================

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(device: &B::Device, hidden: usize) -> Self {
        Self {
            l1: LinearConfig::new(EMB_DIM, hidden).init(device),
            l2: LinearConfig::new(hidden, EMB_DIM).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.l1.forward(x));
        self.l2.forward(h)
    }
}

// ============================================================================
// BurnAgent
// ============================================================================

pub struct BurnAgent {
    id: AgentId,
    name: String,
    hemisphere: Hemisphere,
    competency: Embedding,
    footprint_mb: f32,
    reliability: f32,
    placement: Placement,
    epochs: usize,
    lr: f64,
    device: Dev,
    model: Option<Mlp<AD>>,
    /// Interleaved-replay reservoir: a diverse sample per concept ever seen, so
    /// training doesn't catastrophically forget concepts the curriculum moved past.
    reservoir: std::collections::HashMap<usize, Vec<Embedding>>,
    per_concept_cap: usize,
}

impl BurnAgent {
    pub fn new(
        id: AgentId,
        name: impl Into<String>,
        hemisphere: Hemisphere,
        competency: Embedding,
        footprint_mb: f32,
    ) -> Self {
        let device = Dev::default();
        let model = Mlp::<AD>::init(&device, 32);
        BurnAgent {
            id,
            name: name.into(),
            hemisphere,
            competency,
            footprint_mb,
            reliability: 0.5,
            placement: Placement::Cold,
            epochs: 8,
            lr: 0.02,
            device,
            model: Some(model),
            reservoir: std::collections::HashMap::new(),
            per_concept_cap: 12,
        }
    }

    pub fn boxed(self) -> Box<dyn Agent> {
        Box::new(self)
    }

    /// Forward pass → (best concept index, its probability).
    pub fn predict(&self, query: &Embedding) -> (usize, f32) {
        let model = self.model.as_ref().expect("model resident");
        let x = Tensor::<AD, 2>::from_data(TensorData::new(query.clone(), [1, EMB_DIM]), &self.device);
        let probs = activation::softmax(model.forward(x), 1);
        let v: Vec<f32> = probs.into_data().to_vec().expect("f32 probs");
        let mut bi = 0usize;
        let mut bp = f32::MIN;
        for (i, &p) in v.iter().enumerate() {
            if p > bp {
                bp = p;
                bi = i;
            }
        }
        (bi, bp)
    }

    /// Convenience for demos/tests: the concept name this agent would say.
    pub fn top_concept(&self, query: &Embedding) -> String {
        CONCEPTS[self.predict(query).0].to_string()
    }
}

impl Agent for BurnAgent {
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
    fn set_reliability(&mut self, r: f32) {
        self.reliability = r;
    }

    fn on_placement(&mut self, tier: Placement) {
        // v1: weights stay resident (they're tiny) so learning is never lost;
        // the tier is tracked for reporting. Moving GPU↔CPU backends and record-
        // based unload is a follow-on optimization.
        self.placement = tier;
    }

    fn predict_concept(&self, query: &Embedding) -> Option<usize> {
        Some(self.predict(query).0)
    }

    fn think(&mut self, ctx: &Context, rng: &mut Rng) -> Proposal {
        let (idx, p) = self.predict(ctx.query);
        let mut strength = p * (0.5 + self.reliability);
        if self.hemisphere == ctx.lead {
            strength *= ctx.lead_gain;
        } else {
            strength *= ctx.follow_gain;
        }
        strength *= 0.95 + 0.1 * rng.next_f32();

        let (kind, utter) = if strength > ctx.propose_threshold {
            (ActionKind::Speak, CONCEPTS[idx].to_string())
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
                confidence: p,
            },
            weight: strength.max(0.0),
            rationale: format!("burn p={p:.2}"),
        }
    }

    fn reinforce(&mut self, reward: f32, was_winner: bool) {
        let lr = if was_winner { 0.05 } else { 0.01 };
        self.reliability = (self.reliability + lr * reward).clamp(0.0, 1.5);
    }

    fn consolidate(&mut self, episodes: &[Episode], _rng: &mut Rng) {
        // Reward-filtered, interleaved-replay supervised learning: fold this
        // sleep's rewarded episodes into a per-concept reservoir, then train the
        // weights on the whole reservoir (recent + old) so it keeps what it knew.
        for e in episodes {
            if e.reward > 0.0 {
                if let Some(c) = concept_index(&e.dominant) {
                    let v = self.reservoir.entry(c).or_default();
                    v.push(e.query.clone());
                    if v.len() > self.per_concept_cap {
                        v.remove(0);
                    }
                }
            }
        }
        let mut xs: Vec<f32> = Vec::new();
        let mut ys: Vec<i64> = Vec::new();
        for (c, samples) in &self.reservoir {
            for q in samples {
                xs.extend_from_slice(q);
                ys.push(*c as i64);
            }
        }
        let n = ys.len();
        if n == 0 {
            return;
        }

        let x = Tensor::<AD, 2>::from_data(TensorData::new(xs, [n, EMB_DIM]), &self.device);
        let y = Tensor::<AD, 1, Int>::from_data(TensorData::new(ys, [n]), &self.device);
        let loss_fn = CrossEntropyLossConfig::new().init(&self.device);

        let mut model = self.model.take().expect("model resident");
        let mut opt = AdamConfig::new().init::<AD, Mlp<AD>>();
        for _ in 0..self.epochs {
            let logits = model.forward(x.clone());
            let loss = loss_fn.forward(logits, y.clone());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads::<AD, Mlp<AD>>(grads, &model);
            model = opt.step(self.lr, model, grads);
        }
        self.model = Some(model);
    }
}
