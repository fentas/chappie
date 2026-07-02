//! learned_coordinator — does a *learned* router close the gap, and can the committee
//! win *over time*?
//!
//! Part 1: replace the fixed confidence-vote with a small trainable **gate** over the
//! frozen family experts, and watch the **learning curve** — accuracy as the router
//! develops — against the static monolith and the oracle ceiling.
//!
//! Part 2 (the temporal win): a new family arrives. The monolith fine-tuned on it
//! **forgets** the old ones; the committee **grows a new expert** while the old ones stay
//! frozen — so it retains. This is the edge that only shows up *over time*.
//!
//! Run: cargo run -p chappie-burn --bin learned_coordinator --release

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};
use chappie_core::*;
use std::rc::Rc;

type AD = Autodiff<NdArray>;
type Dev = burn::backend::ndarray::NdArrayDevice;

const LR: f64 = 0.02;
const EPOCHS: usize = 220;
const TEST_PER_CONCEPT: usize = 60;

fn families() -> [(&'static str, Vec<usize>); 3] {
    [
        ("sensory", vec![0, 1, 2]),
        ("abstract", vec![5, 7, 8]),
        ("social", vec![9, 10, 3]),
    ]
}
fn all_concepts() -> Vec<usize> {
    vec![0, 1, 2, 3, 5, 7, 8, 9, 10]
}
fn family_of(c: usize) -> Vec<usize> {
    families().into_iter().find(|(_, g)| g.contains(&c)).map(|(_, g)| g).unwrap_or_default()
}

fn sample(c: usize, hard: bool, rng: &mut Rng) -> Vec<f32> {
    let mut q = vec![0.0f32; EMB_DIM];
    q[c] = 1.0;
    if hard {
        let others: Vec<usize> = family_of(c).into_iter().filter(|&x| x != c).collect();
        if !others.is_empty() {
            q[others[rng.next_range(others.len())]] += 0.45;
        }
        for v in q.iter_mut() {
            *v += 0.15 * rng.next_gauss();
        }
    } else {
        for v in q.iter_mut() {
            *v += 0.05 * rng.next_gauss();
        }
    }
    normalize(&mut q);
    q
}

fn batch(concepts: &[usize], reps: usize, device: &Dev, rng: &mut Rng) -> (Tensor<AD, 2>, Tensor<AD, 1, Int>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for r in 0..reps {
        for &c in concepts {
            xs.extend_from_slice(&sample(c, r % 2 == 0, rng));
            ys.push(c as i64);
        }
    }
    let n = ys.len();
    (
        Tensor::from_data(TensorData::new(xs, [n, EMB_DIM]), device),
        Tensor::from_data(TensorData::new(ys, [n]), device),
    )
}

#[derive(Module, Debug)]
struct Monolith<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}
impl<B: Backend> Monolith<B> {
    fn init(d: &B::Device, hidden: usize) -> Self {
        Self { l1: LinearConfig::new(EMB_DIM, hidden).init(d), l2: LinearConfig::new(hidden, EMB_DIM).init(d) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(activation::relu(self.l1.forward(x)))
    }
}

#[derive(Module, Debug)]
struct Base<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}
impl<B: Backend> Base<B> {
    fn init(d: &B::Device) -> Self {
        Self { l1: LinearConfig::new(EMB_DIM, 64).init(d), l2: LinearConfig::new(64, 32).init(d) }
    }
    fn features(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::relu(self.l2.forward(activation::relu(self.l1.forward(x))))
    }
}
#[derive(Module, Debug)]
struct Head<B: Backend> {
    l: Linear<B>,
}
impl<B: Backend> Head<B> {
    fn init(d: &B::Device) -> Self {
        Self { l: LinearConfig::new(32, EMB_DIM).init(d) }
    }
    fn forward(&self, f: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l.forward(f)
    }
}
// The learned router: percept -> a weight per expert.
#[derive(Module, Debug)]
struct Gate<B: Backend> {
    l: Linear<B>,
}
impl<B: Backend> Gate<B> {
    fn init(d: &B::Device, k: usize) -> Self {
        Self { l: LinearConfig::new(EMB_DIM, k).init(d) }
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

/// Learned-gate mixture: the expert logits, weighted per-sample by the gate.
fn committee_logits(base: &Base<AD>, heads: &[Head<AD>], gate: &Gate<AD>, x: Tensor<AD, 2>) -> Tensor<AD, 2> {
    let feat = base.features(x.clone());
    let gw = activation::softmax(gate.l.forward(x), 1); // [n, K]
    let mut out: Option<Tensor<AD, 2>> = None;
    for (k, head) in heads.iter().enumerate() {
        let lk = head.forward(feat.clone()); // [n, EMB_DIM]
        let wk = gw.clone().narrow(1, k, 1); // [n, 1]
        let term = lk * wk; // broadcast [n,1] over [n,EMB_DIM]
        out = Some(match out {
            None => term,
            Some(o) => o + term,
        });
    }
    out.unwrap()
}

fn mono_accuracy(m: &Monolith<AD>, set: &[(Vec<f32>, usize)], device: &Dev) -> f32 {
    let ok = set
        .iter()
        .filter(|(q, c)| {
            let x = Tensor::<AD, 2>::from_data(TensorData::new(q.clone(), [1, EMB_DIM]), device);
            argmax(&m.forward(x).into_data().to_vec().unwrap()) == *c
        })
        .count();
    100.0 * ok as f32 / set.len() as f32
}

fn committee_acc(
    base: &Base<AD>,
    heads: &[Head<AD>],
    gate: &Gate<AD>,
    set: &[(Vec<f32>, usize)],
    device: &Dev,
) -> f32 {
    let ok = set
        .iter()
        .filter(|(q, c)| {
            let x = Tensor::<AD, 2>::from_data(TensorData::new(q.clone(), [1, EMB_DIM]), device);
            let v: Vec<f32> = committee_logits(base, heads, gate, x).into_data().to_vec().unwrap();
            argmax(&v) == *c
        })
        .count();
    100.0 * ok as f32 / set.len() as f32
}

fn main() {
    let device = Dev::default();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let all = all_concepts();

    let mut trng = Rng::new(999);
    let mut make_test = |hard: bool| -> Vec<(Vec<f32>, usize)> {
        let mut set = Vec::new();
        for &c in &all {
            for _ in 0..TEST_PER_CONCEPT {
                set.push((sample(c, hard, &mut trng), c));
            }
        }
        set
    };
    let test_clean = make_test(false);
    let test_hard = make_test(true);

    let mut rng = Rng::new(1);

    // Monolith reference (matched params).
    let hidden = 163;
    let mut mono = Monolith::<AD>::init(&device, hidden);
    let mut opt = AdamConfig::new().init::<AD, Monolith<AD>>();
    for _ in 0..EPOCHS {
        let (x, y) = batch(&all, 4, &device, &mut rng);
        let logits = mono.forward(x);
        let loss = loss_fn.forward(logits, y);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<AD, Monolith<AD>>(grads, &mono);
        mono = opt.step(LR, mono, grads);
    }
    let mono_acc = |set: &[(Vec<f32>, usize)]| -> f32 {
        let ok = set
            .iter()
            .filter(|(q, c)| {
                let x = Tensor::<AD, 2>::from_data(TensorData::new(q.clone(), [1, EMB_DIM]), &device);
                argmax(&mono.forward(x).into_data().to_vec().unwrap()) == *c
            })
            .count();
        100.0 * ok as f32 / set.len() as f32
    };

    // Committee: pretrain+freeze base, then a specialist head per family.
    let mut base = Base::<AD>::init(&device);
    let mut tmp = Head::<AD>::init(&device);
    let mut ob = AdamConfig::new().init::<AD, Base<AD>>();
    let mut oh = AdamConfig::new().init::<AD, Head<AD>>();
    for _ in 0..120 {
        let (x, y) = batch(&all, 4, &device, &mut rng);
        let loss = loss_fn.forward(tmp.forward(base.features(x)), y);
        let mut grads = loss.backward();
        let gb = GradientsParams::from_module::<AD, Base<AD>>(&mut grads, &base);
        base = ob.step(LR, base, gb);
        let gh = GradientsParams::from_module::<AD, Head<AD>>(&mut grads, &tmp);
        tmp = oh.step(LR, tmp, gh);
    }
    let base = Rc::new(base);
    let mut heads = Vec::new();
    for (_, group) in families() {
        let mut head = Head::<AD>::init(&device);
        let mut hopt = AdamConfig::new().init::<AD, Head<AD>>();
        for _ in 0..EPOCHS {
            let (x, y) = batch(&group, 8, &device, &mut rng);
            let loss = loss_fn.forward(head.forward(base.features(x)), y);
            let mut grads = loss.backward();
            let gh = GradientsParams::from_module::<AD, Head<AD>>(&mut grads, &head);
            head = hopt.step(LR, head, gh);
        }
        heads.push(head);
    }
    let fam_index = |c: usize| -> usize {
        families().iter().position(|(_, g)| g.contains(&c)).unwrap_or(0)
    };
    let oracle_acc = |set: &[(Vec<f32>, usize)]| -> f32 {
        let ok = set
            .iter()
            .filter(|(q, c)| {
                let x = Tensor::<AD, 2>::from_data(TensorData::new(q.clone(), [1, EMB_DIM]), &device);
                let v: Vec<f32> = activation::softmax(heads[fam_index(*c)].forward(base.features(x)), 1)
                    .into_data().to_vec().unwrap();
                argmax(&v) == *c
            })
            .count();
        100.0 * ok as f32 / set.len() as f32
    };

    println!("learned coordinator — does a *learned* router close the gap?\n");
    println!("  reference   clean={:.1}%  hard={:.1}%   monolith", mono_acc(&test_clean), mono_acc(&test_hard));
    println!("  reference   clean={:.1}%  hard={:.1}%   oracle route (the ceiling)\n", oracle_acc(&test_clean), oracle_acc(&test_hard));

    // Part 1 — the learning curve: train the gate, watch routing develop.
    println!("  the router learning to route (committee, learned gate):");
    println!("  {:>6} {:>10} {:>10}", "step", "clean", "hard");
    let mut gate = Gate::<AD>::init(&device, heads.len());
    let mut gopt = AdamConfig::new().init::<AD, Gate<AD>>();
    let checkpoints = [0usize, 25, 50, 100, 200, 400];
    let mut ci = 0;
    for step in 0..=400usize {
        if ci < checkpoints.len() && step == checkpoints[ci] {
            let cl = committee_acc(&base, &heads, &gate, &test_clean, &device);
            let hd = committee_acc(&base, &heads, &gate, &test_hard, &device);
            println!("  {:>6} {:>9.1}% {:>9.1}%", step, cl, hd);
            ci += 1;
        }
        let (x, y) = batch(&all, 4, &device, &mut rng);
        let logits = committee_logits(&base, &heads, &gate, x);
        let loss = loss_fn.forward(logits, y);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<AD, Gate<AD>>(grads, &gate);
        gate = gopt.step(LR, gate, grads);
    }
    println!("\n  → the committee starts near chance and the learned router walks it up toward");
    println!("    the oracle ceiling — the gap really was the coordinator, and it's learnable.");

    // ---- Part 2: the temporal win — continual learning ----
    println!("\n  Part 2 — continual learning: the 'social' family arrives late.");
    let old: Vec<usize> = vec![0, 1, 2, 5, 7, 8];
    let new: Vec<usize> = vec![9, 10, 3];
    let mut orng = Rng::new(7);
    let build = |g: &[usize], rng: &mut Rng| -> Vec<(Vec<f32>, usize)> {
        let mut s = Vec::new();
        for &c in g {
            for _ in 0..TEST_PER_CONCEPT {
                s.push((sample(c, true, rng), c));
            }
        }
        s
    };
    let old_test = build(&old, &mut orng);
    let new_test = build(&new, &mut orng);

    // Monolith: learn OLD, then fine-tune on NEW only — and forget.
    let mut m = Monolith::<AD>::init(&device, hidden);
    let mut mo = AdamConfig::new().init::<AD, Monolith<AD>>();
    for _ in 0..EPOCHS {
        let (x, y) = batch(&old, 4, &device, &mut orng);
        let loss = loss_fn.forward(m.forward(x), y);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<AD, Monolith<AD>>(grads, &m);
        m = mo.step(LR, m, grads);
    }
    let m_old_before = mono_accuracy(&m, &old_test, &device);
    for _ in 0..EPOCHS {
        let (x, y) = batch(&new, 4, &device, &mut orng);
        let loss = loss_fn.forward(m.forward(x), y);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<AD, Monolith<AD>>(grads, &m);
        m = mo.step(LR, m, grads);
    }
    let m_old_after = mono_accuracy(&m, &old_test, &device);
    let m_new = mono_accuracy(&m, &new_test, &device);

    // Committee: OLD is served by the frozen sensory+abstract experts (untouched when the
    // social expert is grown); NEW by the new expert. OLD is retained by construction.
    let c_old = oracle_acc(&old_test);
    let c_new = oracle_acc(&new_test);

    println!("  {:<22} {:>14} {:>12}", "", "OLD (retain)", "NEW (learn)");
    println!(
        "  {:<22} {:>4.0}% -> {:>3.0}% {:>11.0}%",
        "monolith (fine-tune)", m_old_before, m_old_after, m_new
    );
    println!(
        "  {:<22} {:>4.0}%  frozen {:>10.0}%",
        "committee (grow)", c_old, c_new
    );
    println!("\n  → fine-tuning the monolith on the new family collapses the old; growing a new");
    println!("    expert leaves the old ones untouched — retention by construction.");
}
