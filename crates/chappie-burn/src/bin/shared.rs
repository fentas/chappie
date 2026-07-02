//! Shared backbone + per-agent adapters — the "100–1000 agents" scaling trick.
//!
//! One deep base network is pretrained and then **frozen and shared** (Rc) across
//! many agents; each agent owns only a tiny trainable **head** (the adapter) for
//! its niche. Backprop flows through the frozen base but only the head is stepped.
//! Result: N specialists cost `base + N·head`, not `N·(base+head)`.
//!
//! Run:  cargo run -p chappie-burn --bin shared

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

// Deep shared trunk: EMB_DIM -> 64 -> 32. Head (adapter): 32 -> EMB_DIM.
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
    fn forward(&self, feat: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l.forward(feat)
    }
}

fn one_hot_noisy(c: usize, rng: &mut Rng) -> Vec<f32> {
    let mut q = vec![0.0f32; EMB_DIM];
    q[c] = 1.0;
    for v in q.iter_mut() {
        *v += 0.05 * rng.next_gauss();
    }
    normalize(&mut q);
    q
}

fn batch(concepts: &[usize], reps: usize, device: &Dev, rng: &mut Rng) -> (Tensor<AD, 2>, Tensor<AD, 1, Int>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..reps {
        for &c in concepts {
            xs.extend_from_slice(&one_hot_noisy(c, rng));
            ys.push(c as i64);
        }
    }
    let n = ys.len();
    (
        Tensor::from_data(TensorData::new(xs, [n, EMB_DIM]), device),
        Tensor::from_data(TensorData::new(ys, [n]), device),
    )
}

fn accuracy(base: &Base<AD>, head: &Head<AD>, concepts: &[usize], device: &Dev) -> f32 {
    let mut correct = 0;
    for &c in concepts {
        let mut q = vec![0.0f32; EMB_DIM];
        q[c] = 1.0;
        let x = Tensor::<AD, 2>::from_data(TensorData::new(q, [1, EMB_DIM]), device);
        let logits = head.forward(base.features(x));
        let v: Vec<f32> = logits.into_data().to_vec().unwrap();
        let pred = v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if pred == c {
            correct += 1;
        }
    }
    correct as f32 / concepts.len() as f32
}

fn main() {
    let device = Dev::default();
    let mut rng = Rng::new(1);
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let lr = 0.02;

    // --- 1. pretrain the base (with a throwaway head) on ALL concepts, then freeze it ---
    let all = [0usize, 1, 2, 3, 5, 7, 8, 9, 10]; // the 9 nameable concepts
    let mut base = Base::<AD>::init(&device);
    let mut tmp = Head::<AD>::init(&device);
    let mut ob = AdamConfig::new().init::<AD, Base<AD>>();
    let mut oh = AdamConfig::new().init::<AD, Head<AD>>();
    for _ in 0..80 {
        let (x, y) = batch(&all, 4, &device, &mut rng);
        let loss = loss_fn.forward(tmp.forward(base.features(x)), y);
        let mut grads = loss.backward();
        let gb = GradientsParams::from_module::<AD, Base<AD>>(&mut grads, &base);
        base = ob.step(lr, base, gb);
        let gh = GradientsParams::from_module::<AD, Head<AD>>(&mut grads, &tmp);
        tmp = oh.step(lr, tmp, gh);
    }
    let base = Rc::new(base); // frozen, shared from here on

    println!("Shared backbone + per-agent adapters");
    println!("  base pretrained on all concepts, then FROZEN and shared (Rc).\n");

    // --- 2. each specialist trains ONLY its own head over the frozen shared base ---
    let groups: [(&str, Vec<usize>); 3] = [
        ("sight/sound/touch", vec![0, 1, 2]),
        ("language/logic/number", vec![5, 7, 8]),
        ("social/danger/smell", vec![9, 10, 3]),
    ];
    let mut heads = Vec::new();
    for (name, group) in &groups {
        let mut head = Head::<AD>::init(&device);
        let acc0 = accuracy(&base, &head, group, &device);
        let mut opt = AdamConfig::new().init::<AD, Head<AD>>();
        for _ in 0..50 {
            let (x, y) = batch(group, 6, &device, &mut rng);
            let loss = loss_fn.forward(head.forward(base.features(x)), y);
            let mut grads = loss.backward();
            let gh = GradientsParams::from_module::<AD, Head<AD>>(&mut grads, &head);
            head = opt.step(lr, head, gh); // only the adapter moves; base stays frozen
        }
        let acc1 = accuracy(&base, &head, group, &device);
        println!("  agent [{name:<22}] adapter accuracy on its niche: {:.0}% → {:.0}%", acc0 * 100.0, acc1 * 100.0);
        heads.push(head);
    }

    // --- 3. the scaling story ---
    let base_p = base.num_params();
    let head_p = heads[0].num_params();
    println!("\nparameter budget (base {base_p} · head {head_p}):");
    for n in [3usize, 100, 1000] {
        let shared = base_p + n * head_p;
        let separate = n * (base_p + head_p);
        println!(
            "  {n:>4} agents:  shared {shared:>8}  vs  separate {separate:>9}   →  {:.0}% saved",
            100.0 * (1.0 - shared as f32 / separate as f32)
        );
    }
    println!("\n  (the deeper the shared trunk and the smaller the adapters, the bigger the win —");
    println!("   this is how a population scales to hundreds of agents on one backbone.)");
}
