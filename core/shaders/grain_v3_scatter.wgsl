// Film Grain v3, pass B of 3: scatter the clumps.
//
// One thread per clump. Thread (x, y) is clump x of tile y; threads past that
// tile's count exit immediately, which is why empty shadows cost almost
// nothing (their Poisson count is ~0).
//
// ACCUMULATION. The CPU model multiplies transmission per clump
// (canvas *= 1 - v). Threads cannot race-free multiply, so this adds
// -log(1 - v) into a fixed-point atomic instead; the resolve pass takes
// exp(-x). Same product, and because integer addition is exact and
// order-independent the result does not depend on thread scheduling — the
// preview and the export of one frame are identical every time.
//
// RNG. Each clump hashes (tile, clump index) into four independent randoms
// rather than walking a sequential per-tile stream, so clumps are independent
// and can run in parallel. mix32/tile_seed here must stay identical to
// _mix32/_tile_seed in core/grain_v3.py — the numpy fallback shares them.

struct Uniforms {
    dims:    vec4u,    // w, h, canvas_w, canvas_h
    tiles:   vec4u,    // ntx, nty, seed, half
    consts:  vec4f,    // span, floor, log(1-CEIL), accum_scale
    shape:   vec4u,    // n_shapes, tile, samples, max_count
}

@group(0) @binding(0) var          src:      texture_2d<f32>;
@group(0) @binding(1) var<storage, read>       counts:   array<u32>;
@group(0) @binding(2) var<storage, read>       ceilings: array<f32>;    // tile_keep
@group(0) @binding(3) var<storage, read>       start:    array<u32>;
@group(0) @binding(4) var<storage, read>       patch_off: array<u32>;   // dy<<16 | dx
@group(0) @binding(5) var<storage, read>       patch_val: array<f32>;
@group(0) @binding(6) var<storage, read_write> accum:    array<atomic<u32>>;
@group(0) @binding(7) var<uniform>             u:        Uniforms;

const INV32: f32 = 1.0 / 4294967296.0;

fn mix32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn tile_seed(ty: u32, tx: u32, seed: u32) -> u32 {
    let v = (ty * 73856093u) ^ (tx * 19349663u) ^ (seed * 83492791u);
    return v | 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let ntx = u.tiles.x;
    let nty = u.tiles.y;
    let t = id.y;
    if t >= ntx * nty { return; }

    let n = counts[t];
    let k = id.x;
    if k >= n { return; }

    let ty = t / ntx;
    let tx = t % ntx;

    let tile     = u.shape.y;
    let samples  = u.shape.z;
    let w        = u.dims.x;
    let h        = u.dims.y;
    let cw       = u.dims.z;
    let chh      = u.dims.w;
    let half     = i32(u.tiles.w);

    // Four independent randoms for this clump.
    let s  = tile_seed(ty, tx, u.tiles.z) ^ (k * 0x9E3779B1u);
    let r0 = mix32(s);
    let r1 = mix32(r0);
    let r2 = mix32(r1);
    let r3 = mix32(r2);
    let fx = f32(r0) * INV32;
    let fy = f32(r1) * INV32;
    let fa = f32(r2) * INV32;
    let fp = f32(r3) * INV32;

    let iy = u32((f32(ty) + fy) * f32(tile));
    let ix = u32((f32(tx) + fx) * f32(tile));
    if iy >= h || ix >= w { return; }

    // How readily a clump survives where it landed. Recomputed from the image
    // rather than read from a `keep` texture — a handful of ALU is cheaper
    // than another full-frame texture.
    let c = textureLoad(src, vec2i(i32(ix), i32(iy)), 0);
    let tone = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    let need = clamp(u.consts.y + u.consts.x * tone,
                     u.consts.y, u.consts.y + u.consts.x);
    let keep = log(1.0 - need) / u.consts.z;

    // `ceiling` is the tile's densest keep, which the host already used to set
    // this tile's clump count; the test below thins that back down to what
    // each clump's own spot actually wants.
    let ceiling = ceilings[t];
    if fa * ceiling > keep { return; }

    let y0 = i32((f32(ty) + fy) * f32(tile) * f32(samples)) - half;
    let x0 = i32((f32(tx) + fx) * f32(tile) * f32(samples)) - half;
    let p  = u32(fp * f32(u.shape.x));

    let lo = start[p];
    let hi = start[p + 1u];
    for (var i = lo; i < hi; i = i + 1u) {
        let off = patch_off[i];
        let y = y0 + i32(off >> 16u);
        if y < 0 || y >= i32(chh) { continue; }
        let x = x0 + i32(off & 0xFFFFu);
        if x < 0 || x >= i32(cw) { continue; }
        let v = min(patch_val[i], 0.999);
        let add = u32(-log(1.0 - v) * u.consts.w);
        if add > 0u {
            atomicAdd(&accum[u32(y) * cw + u32(x)], add);
        }
    }
}
