//! GPU spike: does Burn actually run on this AMD RX 9070 XT (RDNA4) via
//! wgpu/Vulkan — for both inference and autodiff (the sleep-training path)?
//!
//! Run:  WGPU_BACKEND=vulkan cargo run -p chappie-burn --bin spike

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Tensor, activation};

type B = Wgpu;
type Ad = Autodiff<Wgpu>;

fn main() {
    let device = WgpuDevice::default();
    println!("wgpu device: {device:?}");

    // --- inference on the GPU ---
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
    let c = activation::relu(a.matmul(b));
    println!("matmul+relu =\n{}", c.into_data());

    // --- autodiff on the GPU (this is what sleep will use to train adapters) ---
    let x = Tensor::<Ad, 1>::from_floats([2.0, 3.0], &device).require_grad();
    let loss = x.clone().powf_scalar(2.0).sum(); // L = sum(x^2)  ->  dL/dx = 2x
    let grads = loss.backward();
    let gx = x.grad(&grads).expect("gradient exists");
    println!("autodiff dL/dx (expect [4, 6]) = {}", gx.into_data());

    println!("\n✅ Burn runs on this GPU (wgpu/Vulkan): inference + autodiff both work.");
}
