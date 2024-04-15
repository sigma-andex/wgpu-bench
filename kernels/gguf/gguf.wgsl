//Translated from: https://github.com/ggerganov/ggml/blob/master/src/ggml-metal.metal#L5056-L5082
        
@group(0) @binding(0)
var<storage, read> in: array<u32>;

@group(0) @binding(1)
var<storage, read_write> out: array<u32>;


@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) pos: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_workgroups) groups: vec3<u32>,
) {
    // [TODO] Do something here
}
