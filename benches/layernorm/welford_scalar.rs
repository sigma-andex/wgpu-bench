#![allow(non_snake_case)]
use encase::ShaderType;
use inline_python::{python, Context};
use numpy::PyArrayDyn;
use pyo3::Python;
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use wgpu_bencher::{
    dispatch_validate, shape, wgc, wgs, CPUTensor, GPUHandle, KernelBench, KernelContextExt,
    OpMetadata, WgpuTimer, Workload,
};

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

#[derive(ShaderType, derive_new::new, Debug)]
pub struct LayerNormMeta {
    M: u32,
    N: u32,
    ND4: u32,
    eps: f32,
}

impl OpMetadata for LayerNormMeta {}

#[derive(derive_new::new, Debug)]
pub struct LayerNorm {
    eps: f32,
}

const PROB_M: usize = 2048;
const PROB_N: usize = 512;
const WARP_SIZE: usize = 32;

impl KernelBench for LayerNorm {
    type Metadata = LayerNormMeta;

    fn name() -> &'static str {
        "WelfordScalar"
    }

    fn source(&self, workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template(
            Self::name(),
            include_str!("../../kernels/layernorm/welford_scalar.wgsl"),
        )
        .unwrap();
        context.insert_workload(workload);
        tera.render(Self::name(), &context).unwrap()
    }

    fn tensors(&self) -> Vec<CPUTensor> {
        let input = CPUTensor::randn::<f32>(shape![1, PROB_M, PROB_N]);
        let scale = CPUTensor::randn::<f32>(shape![PROB_N]);
        let bias = CPUTensor::randn::<f32>(shape![PROB_N]);
        let output = CPUTensor::zeros::<f32>(shape![1, PROB_M, PROB_N]);
        vec![input, scale, bias, output]
    }

    fn workload(&self, tensors: &[CPUTensor]) -> Workload {
        let input = &tensors[0];
        let [_B, M, _N] = input.shape().try_into().unwrap();
        Workload::new(wgs![WARP_SIZE as _, 1, 1], wgc![M as _, 1, 1])
    }

    fn metadata(&self, tensors: &[CPUTensor]) -> Self::Metadata {
        let input = &tensors[0];
        let [_B, M, N] = input.shape().try_into().unwrap();
        LayerNormMeta::new(M as _, N as _, (N / 4) as _, self.eps)
    }

    fn validate(&self, tensors: &[CPUTensor]) {
        let (input, scale, bias) = (&tensors[0], &tensors[1], &tensors[2]);
        let ground = Python::with_gil(|py| {
            let (py_input, py_scale, py_bias) = (
                input.to_py::<f32>(&py),
                scale.to_py::<f32>(&py),
                bias.to_py::<f32>(&py),
            );
            let result: Context = python! {
                import torch
                import torch.nn.functional as F

                (input, scale, bias) = (torch.from_numpy('py_input), torch.from_numpy('py_scale), torch.from_numpy('py_bias))
                result = F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
            };
            CPUTensor::from(result.get_with_gil::<&PyArrayDyn<f32>>(py, "result"))
        });
        let mut gpu_tensors = dispatch_validate(TIMER.handle(), self, tensors);
        let cpu_result = gpu_tensors.remove(3).into_cpu(TIMER.handle()).unwrap();
        ground.all_close(&cpu_result, 1e-5, 1e-5).unwrap();
    }
}

fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    let throughput = Throughput::Elements((PROB_M * PROB_N) as u64);
    wgpu_bencher::benchmark(c, &TIMER, LayerNorm::new(1e-5), throughput)
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
