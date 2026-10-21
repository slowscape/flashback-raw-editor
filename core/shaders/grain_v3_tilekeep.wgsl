// Film Grain v3, pass A of 3: per-tile reduction.
//
// One thread per 32x32 output tile. Produces two numbers per tile:
//   .x  max `keep` in the tile — the density the scatter pass throws at
//   .y  max chroma deviation   — lets the host decide whether the frame is
//                                really grey, so a colour stock's per-layer
//                                tint is not applied to a mono picture
//
// This exists so the host can draw the per-tile Poisson clump counts without
// reading the whole (GPU-resident) image back: only 2 floats per tile come
// back across the bus.

struct Uniforms {
    dims:   vec4u,     // w, h, ntx, nty
    consts: vec4f,     // span, floor, log(1-CEIL), tile
}

@group(0) @binding(0) var          src:   texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_tile: array<vec2f>;
@group(0) @binding(2) var<uniform>             u:        Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let ntx = u.dims.z;
    let nty = u.dims.w;
    let t = id.x;
    if t >= ntx * nty { return; }

    let ty = t / ntx;
    let tx = t % ntx;
    let tile = u32(u.consts.w);

    let w = u.dims.x;
    let h = u.dims.y;
    let x0 = tx * tile;
    let y0 = ty * tile;

    var best_keep = 0.0;
    var best_chroma = 0.0;
    for (var dy = 0u; dy < tile; dy = dy + 1u) {
        let y = y0 + dy;
        if y >= h { break; }
        for (var dx = 0u; dx < tile; dx = dx + 1u) {
            let x = x0 + dx;
            if x >= w { break; }
            let c = textureLoad(src, vec2i(i32(x), i32(y)), 0);
            let tone = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
            // keep = log1p(-need) / log1p(-CEIL), need = clamp(FLOOR+span*tone)
            let need = clamp(u.consts.y + u.consts.x * tone,
                             u.consts.y, u.consts.y + u.consts.x);
            let k = log(1.0 - need) / u.consts.z;
            best_keep = max(best_keep, k);
            best_chroma = max(best_chroma,
                              max(abs(c.r - c.g), abs(c.g - c.b)));
        }
    }
    out_tile[t] = vec2f(best_keep, best_chroma);
}
