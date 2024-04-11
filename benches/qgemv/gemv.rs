#![allow(non_snake_case)]
use encase::ShaderType;
use inline_python::{python, Context};
use numpy::PyArrayDyn;
use pyo3::{IntoPy, Python};
use smallvec::smallvec;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use wgpu_bencher::{
    dispatch_validate, shape, wgc, wgs, CPUTensor, GPUHandle, KernelBench, KernelContextExt,
    OpMetadata, Quantization, Quantizer, WgpuTimer, Workload,
};

lazy_static::lazy_static! {
    pub static ref TIMER: WgpuTimer = WgpuTimer::new(pollster::block_on(async {
        GPUHandle::new().await.unwrap()
    }));
}

#[derive(ShaderType, Debug)]
pub struct QGEMVMeta {
    aShape: glam::IVec3,
    aStrides: glam::IVec3,
    bShape: glam::IVec3,
    bStrides: glam::IVec3,
    outShape: glam::IVec3,
    outStrides: glam::IVec3,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

impl OpMetadata for QGEMVMeta {}

#[derive(derive_new::new, Debug)]
pub struct QGEMVBenchmark {
    B: usize,
    M: usize,
    N: usize,
    K: usize,
    TILE_DIM: usize,
    ROW_PER_THREAD: usize,
    trans_a: bool,
    trans_b: bool,
}

impl KernelBench for QGEMVBenchmark {
    type Metadata = QGEMVMeta;

    fn name() -> &'static str {
        "QGEMVBenchmark"
    }

    fn source(&self, workload: &Workload) -> String {
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();

        let template = include_str!("../../kernels/qgemv/sgemv_2v.wgsl");
        tera.add_raw_template(Self::name(), template).unwrap();

        context.insert("TILE_DIM", &self.TILE_DIM);
        context.insert("ROW_PER_THREAD", &self.ROW_PER_THREAD);
        context.insert_workload(workload);
        let kernel = tera.render(Self::name(), &context).unwrap();
        println!("{}", kernel);
        kernel
    }

    fn tensors(&self) -> Vec<CPUTensor> {
        let (B, M, N, K) = (self.B, self.M, self.N, self.K);
        println!("B: {}, M: {}, N: {}, K: {}", B, M, N, K);
        let a_unquant = CPUTensor::randn::<f32>(shape![B, M, K]);
        let b = CPUTensor::randn::<f32>(shape![B, K, N]);
        let quantizer = Quantizer::new(Quantization::SInt8);
        let quantized_a = quantizer.quantize(a_unquant.clone());
        let output = CPUTensor::zeros::<f32>(shape![B, M, N]);
        vec![quantized_a, b, output]
    }

    fn workload(&self, _: &[CPUTensor]) -> Workload {
        let wgsx: usize = 4;
        let workgroup_size = wgs![wgsx as _, 256, 1];
        let workgroup_count = wgc![(self.M / wgsx) as _, 1, self.B as _];
        let dispatch = Workload::new(workgroup_size, workgroup_count);
        println!("DISPATCH: {:?}", dispatch);
        dispatch
    }

    fn metadata(&self, _: &[CPUTensor]) -> Self::Metadata {
        let (B, M, N, K) = (self.B as i32, self.M as i32, self.N as i32, self.K as i32);

        let aShape = glam::IVec3::new(B, M, K);
        let aStrides = glam::IVec3::new(M * K, K, 1);
        let bShape = glam::IVec3::new(B, K, N);
        let bStrides = glam::IVec3::new(K * N, N, 1);
        let outShape = glam::IVec3::new(B, M, N);
        let outStrides = glam::IVec3::new(M * N, N, 1);

        let dimAOuter = if self.trans_a { K } else { M };
        let dimBOuter = if self.trans_b { K } else { N };
        let dimInner = if self.trans_a { M } else { K };

        let meta = QGEMVMeta {
            aShape,
            aStrides,
            bShape,
            bStrides,
            outShape,
            outStrides,
            dimAOuter,
            dimBOuter,
            dimInner,
        };
        println!("META: {:?}", meta);
        meta
    }

    fn validate(&self, tensors: &[CPUTensor]) {
        let (aquant, b) = (&tensors[0], &tensors[1]);
        let dequantized = Quantizer::new(Quantization::SInt8).dequantize(aquant.clone());
        let ground = Python::with_gil(|py| {
            let (py_a, py_b) = (dequantized.to_py::<f32>(&py), b.to_py::<f32>(&py));
            let result: Context = python! {
                import torch
                (a, b) = (torch.from_numpy('py_a), torch.from_numpy('py_b))
                print("A: ", a)
                print("B: ", b)
                result = (a @ b).numpy()
            };
            CPUTensor::from(result.get_with_gil::<&PyArrayDyn<f32>>(py, "result"))
        });
        let mut gpu_tensors = dispatch_validate(TIMER.handle(), self, tensors);
        let cpu_result = gpu_tensors.remove(2).into_cpu(TIMER.handle()).unwrap();
        println!("OURS: {}", cpu_result);
        println!("GROUND: {}", ground);
        ground.all_close(&cpu_result, 1e-2, 1e-2).unwrap();
    }
}

pub fn benchmark(c: &mut Criterion<&WgpuTimer>) {
    let B = 1;
    let M = 2560;
    let N = 1;
    let K = 10240;
    let TILE_DIM = 32;
    let ROW_PER_THREAD = 4;

    let trans_a = false;
    let trans_b = false;

    let bench = QGEMVBenchmark::new(B, M, N, K, TILE_DIM, ROW_PER_THREAD, trans_a, trans_b);
    let throughput = Throughput::Elements(2 * (B * M * N * K) as u64);
    wgpu_bencher::benchmark(c, &TIMER, bench, throughput)
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement(&*TIMER);
    targets = benchmark
);
criterion_main!(bench);
