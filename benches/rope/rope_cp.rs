#![allow(non_snake_case)]
use encase::ShaderType;
use inline_python::{python, Context};
use numpy::PyArrayDyn;
use pyo3::Python;
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use wgpu_bencher::{
    dispatch_validate, shape, wgc, wgs, CPUTensor, GPUHandle, KernelBench, KernelContextExt,
    OpMetadata, Strides, WgpuTimer, Workload,
};

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

#[derive(ShaderType, derive_new::new, Debug)]
pub struct RopeMeta {
    in_strides: glam::UVec4,
    out_strides: glam::UVec4,
    offset: u32,
    base: f32,
    rotary_dim: u32,
}

impl OpMetadata for RopeMeta {}

#[derive(Debug)]
pub struct Rope {}

impl KernelBench for Rope {
    type Metadata = RopeMeta;

    fn name() -> &'static str {
        "RoPE"
    }

    fn source(&self, workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template(
            Self::name(),
            include_str!("../../kernels/rope/rope_cp.wgsl"),
        )
        .unwrap();
        context.insert_workload(workload);
        tera.render(Self::name(), &context).unwrap()
    }

    // [ batch_size, num_heads, seq_len, head_dim ]
    fn tensors(&self) -> Vec<CPUTensor> {
        let input = CPUTensor::randn::<f32>(shape![1, 16, 64, 128]);
        let output = CPUTensor::zeros::<f32>(shape![1, 16, 64, 128]);
        vec![input, output]
    }

    //  rotary_ndims = int(args.rotary_pct * head_dim)
    //    threads_per_block = (rotary_ndims, )
    //    blocks_per_grid = (batch_size, n_heads, seq_len)
    fn workload(&self, tensors: &[CPUTensor]) -> Workload {
        let input = &tensors[0];
        let [BS, NH, SL, HD] = input.shape().try_into().unwrap();
        let wl = Workload::new(wgs![32, 1, 1], wgc![BS as _, NH as _, SL as _]);
        println!("{:?}", wl);
        wl
    }

    fn metadata(&self, tensors: &[CPUTensor]) -> Self::Metadata {
        let input = &tensors[0];
        let out = &tensors[1];
        let mut input_shape = input.shape().clone();
        let mut out_shape = out.shape().clone();
        //input_shape.remove(0);
        //out_shape.remove(0);
        let in_strides = Strides::from(&input_shape);
        let out_strides = Strides::from(&out_shape);
        let meta = RopeMeta::new((&in_strides).into(), (&out_strides).into(), 0, 10000.0, 32);
        println!("{:?}", meta);
        meta
    }

    fn validate(&self, tensors: &[CPUTensor]) {
        let input = &tensors[0];
        let ground = Python::with_gil(|py| {
            let py_input = input.to_py::<f32>(&py);
            let result: Context = python! {
                import mlx.core as mx
                import mlx.nn as nn
                import numpy as np

                rope = nn.RoPE(128)
                mx_input = mx.array('py_input).astype(mx.float16)
                y = rope(mx_input)
                mx.eval(y)
                result = np.array(y.astype(mx.float32))
                print("MLX RESULT: ", result)
                print("")

                import torch
                from rotary_embedding_torch import RotaryEmbedding
                rotary_emb = RotaryEmbedding(dim = 32)
                result = rotary_emb.rotate_queries_or_keys(torch.from_numpy('py_input)).numpy()
            };
            CPUTensor::from(result.get_with_gil::<&PyArrayDyn<f32>>(py, "result"))
        });
        let mut gpu_tensors = dispatch_validate(TIMER.handle(), self, tensors);
        let cpu_result = gpu_tensors.remove(1).into_cpu(TIMER.handle()).unwrap();
        println!("TORCH: {}\n", ground);
        println!("US: {}", cpu_result);
        //ground.all_close(&cpu_result, 1e-5, 1e-5).unwrap();
    }
}

fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    let throughput = Throughput::Elements(1 as u64);
    wgpu_bencher::benchmark(c, &TIMER, Rope {}, throughput)
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
