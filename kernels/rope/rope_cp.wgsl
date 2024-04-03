// Kernel by Carson Poole 

@group(0) @binding(0)
var<storage, read> X: array<f32>;

@group(0) @binding(1)
var<storage, read_write> Y: array<f32>;

struct Meta {
    in_strides: vec4<u32>,
    out_strides: vec4<u32>,
    offset: u32,
    base: f32,
    rotary_dim: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_workgroups) groups: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let tid = local_id.x;
    let batch_idx = group_id.x;
    let head_idx = group_id.y;
    let tok_idx = group_id.z;
    let half_rotary_dim = metadata.rotary_dim / 2u;

    let hx = tid / half_rotary_dim;
    let hy = tid % half_rotary_dim;
    let rot_sign = select(-1.0, 1.0, hx == 0u); 

    let global_offset = dot(vec4<u32>(batch_idx, head_idx, tok_idx, 1), metadata.in_strides) + tid;
    let global_offset_rot = dot(vec4<u32>(batch_idx, head_idx, tok_idx, 1), metadata.in_strides) + (1u-hx) * half_rotary_dim + hy;

    let x = X[global_offset];

    let x_rot = rot_sign * X[global_offset_rot];

    let ar = f32(hy) * 2.0;
    let inv_freq = f32(tok_idx + metadata.offset) * (1.0 / pow(metadata.base, ar / f32(metadata.rotary_dim)));

    let sin = sin(inv_freq);
    let cos = cos(inv_freq);

    workgroupBarrier();
    Y[global_offset] = x * cos + x_rot * sin;
}


