//! measure_gpu — real GPU numbers on this card (RX 9070 XT, RDNA4, wgpu/Vulkan).
//!
//! Grounds the swap/scaling envelope with measurements instead of spec sheets:
//! device<->host transfer bandwidth (how fast a model moves out of VRAM) and
//! matmul throughput (whether compute or memory is the bottleneck).
//!
//! Run: WGPU_BACKEND=vulkan cargo run -p chappie-burn --bin measure_gpu --release

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

type B = Wgpu;

fn main() {
    let device = WgpuDevice::default();
    println!("device: {device:?}\n");

    // Warm up — the first op compiles kernels.
    let w = Tensor::<B, 2>::ones([64, 64], &device);
    let _ = w.clone().matmul(w).into_data();

    // ---- device -> host transfer (reading weights back out of VRAM) ------
    println!("== device->host transfer (moving weights out of VRAM) ==");
    for &mb in &[4usize, 64, 256] {
        let n = mb * 1024 * 1024 / 4;
        let data: Vec<f32> = (0..n).map(|i| (i & 0xff) as f32).collect();
        let x = Tensor::<B, 1>::from_data(TensorData::new(data, [n]), &device);
        let _ = x.clone().into_data(); // warm this size
        let iters = 30;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = x.clone().into_data(); // device->host readback
        }
        let s = t.elapsed().as_secs_f64() / iters as f64;
        let gbps = (mb as f64 / 1024.0) / s;
        println!(
            "  {:>4} MB: {:>7.3} ms  ->  {:>5.1} GB/s   (~a {}MB model unloads in {:.1} ms)",
            mb,
            s * 1e3,
            gbps,
            mb,
            s * 1e3
        );
    }

    // ---- matmul throughput (compute) -------------------------------------
    println!("\n== matmul throughput (compute) ==");
    for &m in &[512usize, 1024, 2048] {
        let a = Tensor::<B, 2>::ones([m, m], &device);
        let b = Tensor::<B, 2>::ones([m, m], &device);
        let _ = a.clone().matmul(b.clone()).into_data(); // warm
        let iters = 30;
        let t = Instant::now();
        let mut acc = a.clone();
        for _ in 0..iters {
            acc = acc.matmul(b.clone());
        }
        let _ = acc.into_data(); // single sync at the end
        let s = t.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * (m as f64).powi(3) / s / 1e9;
        println!(
            "  {:>4}x{:<4}: {:>7.3} ms/matmul  ->  {:>6.0} GFLOP/s",
            m,
            m,
            s * 1e3,
            gflops
        );
    }

    println!(
        "\n→ a few-MB LoRA adapter moves in well under a ms; a multi-hundred-MB model is 10s of ms."
    );
    println!(
        "→ so: keep the big base resident, hot-swap tiny adapters — exactly the shared-backbone design."
    );
}
