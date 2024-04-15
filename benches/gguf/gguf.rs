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
pub struct GgufMeta {}

impl OpMetadata for GgufMeta {}

#[derive(Debug)]
pub struct Gguf {}

impl KernelBench for Gguf {
    type Metadata = GgufMeta;

    fn name() -> &'static str {
        "GGUF"
    }

    fn source(&self, workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        tera.add_raw_template(Self::name(), include_str!("../../kernels/gguf/gguf.wgsl"))
            .unwrap();
        context.insert_workload(workload);
        tera.render(Self::name(), &context).unwrap()
    }

    // [batch_size, num_heads, seq_len, head_dim]
    fn tensors(&self) -> Vec<CPUTensor> {
        let input = CPUTensor::randn::<f32>(shape![2, 16, 64, 128]);
        let output = CPUTensor::zeros::<f32>(shape![2, 16, 64, 128]);
        vec![input, output]
    }

    fn workload(&self, tensors: &[CPUTensor]) -> Workload {
        let input = &tensors[0];
        let [BS, NH, SL, HD] = input.shape().try_into().unwrap();

        let total_x = 128 / 2;
        let total_y = SL;
        println!("INPUT: {:?}", input.shape());
        println!("SL * HD: {}", SL * HD);
        let total_z = input.shape().numel() / (SL * HD);

        let wgsx = 16;
        let wgsy = 8;
        let wgsz = 8;

        let wgcx = total_x / wgsx;
        let wgcy = total_y / wgsy;
        let wgcz = total_z / wgsz;

        let wl = Workload::new(
            wgs![wgsx as _, wgsy as _, wgsz as _],
            wgc![wgcx as _, wgcy as _, wgcz as _],
        );
        println!("{:?}", wl);
        wl
    }

    fn metadata(&self, tensors: &[CPUTensor]) -> Self::Metadata {
        let input = &tensors[0];
        let out = &tensors[1];
        let mut input_shape = input.shape().clone();
        let mut out_shape = out.shape().clone();
        input_shape.remove(0);
        out_shape.remove(0);
        let in_strides = Strides::from(&input_shape);
        let out_strides = Strides::from(&out_shape);
        let meta = GgufMeta::new();
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
                mx_input = mx.array('py_input)
                y = rope(mx_input)
                mx.eval(y)
                result = np.array(y)
            };
            CPUTensor::from(result.get_with_gil::<&PyArrayDyn<f32>>(py, "result"))
        });
        let mut gpu_tensors = dispatch_validate(TIMER.handle(), self, tensors);
        let cpu_result = gpu_tensors.remove(1).into_cpu(TIMER.handle()).unwrap();
        println!("MLX: {}\n", ground);
        println!("US: {}", cpu_result);
        ground.all_close(&cpu_result, 1e-5, 1e-5).unwrap();
    }
}

fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    let throughput = Throughput::Elements(16 * 64 * 128);
    wgpu_bencher::benchmark(c, &TIMER, Gguf {}, throughput)
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
