//https://www.bealto.com/gpu-gemv_v1.html
var<private> localId: vec3<u32>;
var<private> globalId: vec3<u32>;
var<private> workgroupId: vec3<u32>;

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
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
fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);

    let row = i32(globalId.x) * 2;

    let aIndex0 = metadata.aStrides.y * row / 4;
    let aIndex1 = metadata.aStrides.y * (row + 1) / 4;

    for (var k = 0; k < metadata.dimInner / 4; k+=1) {
        sum0 = fma(A[aIndex0 + k], X[k], sum0);
        sum1 = fma(A[aIndex1 + k], X[k], sum1);
    }
    let outIndex0 = metadata.outShapeStrides.y * row;
    let outIndex1 = metadata.outShapeStrides.y * (row + 1);
    result[outIndex0] = dot(sum0, vec4<f32>(1.0)); 
    result[outIndex1] = dot(sum1, vec4<f32>(1.0));
}
