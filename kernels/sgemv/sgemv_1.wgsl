//https://www.bealto.com/gpu-gemv_v1.html
var<private> localId: vec3<u32>;
var<private> globalId: vec3<u32>;
var<private> workgroupId: vec3<u32>;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(1) @binding(0) var<uniform> metadata: Meta;


struct Meta {
    aShape: vec3<i32>,
    aStrides: vec3<i32>,
    bShape: vec3<i32>,
    bStrides: vec3<i32>,
    outShape: vec3<i32>,
    outShapeStrides: vec3<i32>,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

@compute @workgroup_size({{workgroup_size_x}},{{workgroup_size_y}},{{workgroup_size_z}})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = i32(globalId.z);
    let batchA = batch % metadata.aShape[0];
    let batchB = batch % metadata.bShape[0];

    let aOffset = metadata.aStrides.x * batchA;
    let bOffset = metadata.bStrides.x * batchB;
    let outOffset = metadata.outShapeStrides.x * batch;

    var sum = 0.0;
    let row = i32(globalId.x);
    let aIndex = aOffset + metadata.aStrides.y * row;
    for (var k = 0; k < metadata.dimInner; k+=1) {
        sum = fma(A[aIndex + k], X[bOffset + k], sum);
    }
    let outIndex = outOffset + metadata.outShapeStrides.y * row;
    result[outIndex] = sum;
}
