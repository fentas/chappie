//! committee_vs_monolith — the experiment that turns the thesis into evidence.
//!
//! At an EQUAL parameter budget, does a **committee** (one shared backbone + K
//! specialist adapter-heads, combined by a confidence-weighted vote) beat a
//! **monolith** (a single MLP) on the task — especially the HARD, confusable
//! slice? Same data, same params, measured head-to-head.
//!
//! The task is built to have the structure the debate is about: 9 concepts in 3
//! families, and the *hard* inputs blend a within-family confuser — the fine
//! distinctions a specialist trains hardest on. If specialization + consensus is
//! worth anything, it should show up here; if not, that's honest counter-evidence.
//!
//! Run: cargo run -p chappie-burn --bin committee_vs_monolith --release

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData, activation};
use chappie_core::*;
use std::rc::Rc;

type AD = Autodiff<NdArray>;
type Dev = burn::backend::ndarray::NdArrayDevice;

const LR: f64 = 0.02;
const EPOCHS: usize = 220;
const TEST_PER_CONCEPT: usize = 200;

// 9 concepts in 3 families (same split as the shared-backbone demo).
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
    families()
        .into_iter()
        .find(|(_, g)| g.contains(&c))
        .map(|(_, g)| g)
        .unwrap_or_default()
}

/// A sample for concept `c`. Hard = blend a within-family confuser + heavy noise.
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

/// A mixed batch (half clean, half hard) over `concepts`.
fn batch(
    concepts: &[usize],
    reps: usize,
    device: &Dev,
    rng: &mut Rng,
) -> (Tensor<AD, 2>, Tensor<AD, 1, Int>) {
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

// ---- the two architectures ------------------------------------------------

// Monolith: one MLP, EMB_DIM -> hidden -> EMB_DIM.
#[derive(Module, Debug)]
struct Monolith<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}
impl<B: Backend> Monolith<B> {
    fn init(d: &B::Device, hidden: usize) -> Self {
        Self {
            l1: LinearConfig::new(EMB_DIM, hidden).init(d),
            l2: LinearConfig::new(hidden, EMB_DIM).init(d),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(activation::relu(self.l1.forward(x)))
    }
}

// Committee: shared Base (frozen) + per-family Head adapters.
#[derive(Module, Debug)]
struct Base<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}
impl<B: Backend> Base<B> {
    fn init(d: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(EMB_DIM, 64).init(d),
            l2: LinearConfig::new(64, 32).init(d),
        }
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
        Self {
            l: LinearConfig::new(32, EMB_DIM).init(d),
        }
    }
    fn forward(&self, f: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l.forward(f)
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn main() {
    let device = Dev::default();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let all = all_concepts();

    // ---- fixed test sets (both models judged on identical data) ----------
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

    // ---- 1. Monolith -----------------------------------------------------
    // Hidden sized so its param count ≈ committee (base 12→64→32 + 3 heads 32→12).
    let hidden = 163;
    let mut rng = Rng::new(1);
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
    let mono_predict = |q: &[f32]| -> usize {
        let x = Tensor::<AD, 2>::from_data(TensorData::new(q.to_vec(), [1, EMB_DIM]), &device);
        let v: Vec<f32> = mono.forward(x).into_data().to_vec().unwrap();
        argmax(&v)
    };

    // ---- 2. Committee: pretrain+freeze base, then a specialist head/family
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
    let base = Rc::new(base); // frozen, shared

    let mut heads = Vec::new();
    for (_, group) in families() {
        let mut head = Head::<AD>::init(&device);
        let mut opt = AdamConfig::new().init::<AD, Head<AD>>();
        for _ in 0..EPOCHS {
            let (x, y) = batch(&group, 8, &device, &mut rng); // deep on its own niche
            let loss = loss_fn.forward(head.forward(base.features(x)), y);
            let mut grads = loss.backward();
            let gh = GradientsParams::from_module::<AD, Head<AD>>(&mut grads, &head);
            head = opt.step(LR, head, gh);
        }
        heads.push(head);
    }
    // Confidence-weighted vote: each head's softmax scaled by its own peak confidence.
    let committee_predict = |q: &[f32]| -> usize {
        let x = Tensor::<AD, 2>::from_data(TensorData::new(q.to_vec(), [1, EMB_DIM]), &device);
        let feat = base.features(x);
        let mut acc = vec![0.0f32; EMB_DIM];
        for head in &heads {
            let probs: Vec<f32> = activation::softmax(head.forward(feat.clone()), 1)
                .into_data()
                .to_vec()
                .unwrap();
            let conf = probs.iter().cloned().fold(0.0f32, f32::max);
            for (a, p) in acc.iter_mut().zip(&probs) {
                *a += conf * *p;
            }
        }
        argmax(&acc)
    };

    // Oracle routing: send each input to its TRUE family's expert. This isolates
    // the naive vote's routing error from the specialists' own representation limit.
    let fam_index = |c: usize| -> usize {
        families()
            .iter()
            .position(|(_, g)| g.contains(&c))
            .unwrap_or(0)
    };
    let committee_oracle = |q: &[f32], true_c: usize| -> usize {
        let x = Tensor::<AD, 2>::from_data(TensorData::new(q.to_vec(), [1, EMB_DIM]), &device);
        let feat = base.features(x);
        let probs: Vec<f32> = activation::softmax(heads[fam_index(true_c)].forward(feat), 1)
            .into_data()
            .to_vec()
            .unwrap();
        argmax(&probs)
    };

    // ---- 3. Score all on the identical test sets -------------------------
    let score = |predict: &dyn Fn(&[f32]) -> usize, set: &[(Vec<f32>, usize)]| -> f32 {
        let ok = set.iter().filter(|(q, c)| predict(q) == *c).count();
        100.0 * ok as f32 / set.len() as f32
    };
    let score_oracle = |set: &[(Vec<f32>, usize)]| -> f32 {
        let ok = set
            .iter()
            .filter(|(q, c)| committee_oracle(q, *c) == *c)
            .count();
        100.0 * ok as f32 / set.len() as f32
    };
    let mono_p = mono.num_params();
    let comm_p = base.num_params() + heads.iter().map(|h| h.num_params()).sum::<usize>();

    println!("committee vs monolith — equal parameter budget, identical data\n");
    println!("  monolith : 1 net (hidden {hidden})          · {mono_p} params");
    println!(
        "  committee: shared base + {} heads (vote)   · {comm_p} params  ({:+.1}% vs monolith)\n",
        heads.len(),
        100.0 * (comm_p as f32 / mono_p as f32 - 1.0)
    );
    println!(
        "  {:<8} {:>10} {:>16} {:>18}",
        "slice", "monolith", "committee(vote)", "committee(oracle)"
    );
    println!(
        "  {:<8} {:>9.1}% {:>15.1}% {:>17.1}%",
        "clean",
        score(&mono_predict, &test_clean),
        score(&committee_predict, &test_clean),
        score_oracle(&test_clean)
    );
    println!(
        "  {:<8} {:>9.1}% {:>15.1}% {:>17.1}%",
        "hard",
        score(&mono_predict, &test_hard),
        score(&committee_predict, &test_hard),
        score_oracle(&test_hard)
    );
    println!("  (oracle = each input routed to its true-family expert — isolates router error)");

    // ---- 4. the scaling dimension (why the committee wins beyond accuracy)
    let (bp, hp) = (base.num_params(), heads[0].num_params());
    println!("\n  scaling (shared base {bp} · head {hp} each):");
    for n in [3usize, 100, 1000] {
        let shared = bp + n * hp;
        let separate = n * (bp + hp);
        println!(
            "    {n:>4} specialists: shared {shared:>8} vs separate {separate:>9}  ({:.0}% saved)",
            100.0 * (1.0 - shared as f32 / separate as f32)
        );
    }
}
