// Film Grain v4: scatter, with no clump library.
//
// v3 stamps one of 128 pre-rasterised patches per clump. v4 grows the clump
// where it lands: it draws its own crystal count, then gives every crystal its
// own angle, its own radius within the disc and its own size. Nothing is
// precomputed and nothing but the per-tile clump counts is uploaded, so the
// three patch-geometry buffers v3 needs are gone and every clump is unique.
//
// The frame decomposition, the acceptance test against `keep`, and the
// fixed-point log accumulator are all v3's, unchanged — this shader is a drop-in
// replacement for grain_v3_scatter.wgsl and runs between the same tile-reduction
// and resolve passes.
//
// mix32/tile_seed/gauss must stay identical to their twins in core/grain_v4.py;
// the numpy fallback shares this RNG and is the oracle this is checked against.

struct Uniforms {
    dims:   vec4u,    // w, h, canvas_w, canvas_h
    tiles:  vec4u,    // ntx, nty, seed, max_crystals
    consts: vec4f,    // span, floor, log(1-CEIL), accum_scale
    grain:  vec4f,    // spread, crystal radius, opacity, crystals per clump
    shape:  vec4u,    // tile, samples, _, _
}

@group(0) @binding(0) var          src:      texture_2d<f32>;
@group(0) @binding(1) var<storage, read>       counts:   array<u32>;
@group(0) @binding(2) var<storage, read>       ceilings: array<f32>;
@group(0) @binding(3) var<storage, read_write> accum:    array<atomic<u32>>;
@group(0) @binding(4) var<uniform>             u:        Uniforms;

const INV32: f32 = 1.0 / 4294967296.0;
const TAU: f32 = 6.283185307179586;
const CRYSTAL_SIGMA: f32 = 0.25;
const SUBPIXEL_R: f32 = 0.5;

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

// Box-Muller. Used for the crystal count and for the lognormal size spread.
fn gauss(u1: f32, u2: f32) -> f32 {
    return sqrt(-2.0 * log(max(u1, 1e-7))) * cos(TAU * u2);
}

fn deposit(idx: u32, v: f32, scale: f32) {
    let add = u32(-log(1.0 - min(max(v, 0.0), 0.999)) * scale);
    if add > 0u {
        atomicAdd(&accum[idx], add);
    }
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

    let tile    = u.shape.x;
    let samples = u.shape.y;
    let w       = u.dims.x;
    let h       = u.dims.y;
    let cw      = u.dims.z;
    let chh     = u.dims.w;

    // Five draws for the clump: where it falls, whether it survives, and how
    // many crystals it grew.
    let s  = tile_seed(ty, tx, u.tiles.z) ^ (k * 0x9E3779B1u);
    let r0 = mix32(s);
    let r1 = mix32(r0);
    let r2 = mix32(r1);
    let r3 = mix32(r2);
    let r4 = mix32(r3);
    let fx = f32(r0) * INV32;
    let fy = f32(r1) * INV32;
    let fa = f32(r2) * INV32;

    let iy = u32((f32(ty) + fy) * f32(tile));
    let ix = u32((f32(tx) + fx) * f32(tile));
    if iy >= h || ix >= w { return; }

    // How readily a clump survives where it landed (v3's tone inversion).
    let c = textureLoad(src, vec2i(i32(ix), i32(iy)), 0);
    let tone = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    let need = clamp(u.consts.y + u.consts.x * tone,
                     u.consts.y, u.consts.y + u.consts.x);
    let keep = log(1.0 - need) / u.consts.z;
    if fa * ceilings[t] > keep { return; }

    // Clump centre in canvas pixels, left fractional so clumps land between
    // pixels rather than snapping to the grid.
    let ccy = (f32(ty) + fy) * f32(tile) * f32(samples);
    let ccx = (f32(tx) + fx) * f32(tile) * f32(samples);

    let spread   = u.grain.x;
    let radius   = u.grain.y;
    let opacity  = u.grain.z;
    let per_clump = u.grain.w;
    let scale    = u.consts.w;

    // Crystals per clump varies the way it does across v3's library, but
    // continuously rather than over 128 fixed shapes.
    let nz = gauss(f32(r3) * INV32, f32(r4) * INV32);
    let ncry = u32(clamp(round(per_clump + sqrt(per_clump) * nz),
                         1.0, f32(u.tiles.w)));

    for (var j = 0u; j < ncry; j = j + 1u) {
        let cs = mix32(r4 ^ (j * 0x85EBCA6Bu));
        let c0 = mix32(cs);
        let c1 = mix32(c0);
        let c2 = mix32(c1);
        let c3 = mix32(c2);

        // sqrt of a uniform fills the disc evenly instead of bunching at the
        // centre — the crystal distribution v3 draws its patches with.
        let ang = TAU * (f32(c0) * INV32);
        let rad = sqrt(f32(c1) * INV32) * spread;
        let rr = max(1e-4, radius * exp(CRYSTAL_SIGMA
                                        * gauss(f32(c2) * INV32, f32(c3) * INV32)));

        let py = ccy + rad * sin(ang);
        let px = ccx + rad * cos(ang);
        let area = min(3.14159265 * rr * rr * opacity, 0.999);

        if rr <= SUBPIXEL_R {
            // Finer than the grid it is drawn on: the whole crystal drops into
            // the pixel it landed in.
            let yi = i32(floor(py));
            let xi = i32(floor(px));
            if yi >= 0 && yi < i32(chh) && xi >= 0 && xi < i32(cw) {
                deposit(u32(yi) * cw + u32(xi), area, scale);
            }
        } else {
            // Wide enough to read as a disc: spread the area over the footprint
            // by coverage, normalised so no silver is gained or lost.
            let kk = i32(min(3.0, ceil(rr + 0.5)));
            let cy = i32(floor(py));
            let cx = i32(floor(px));
            var wsum = 0.0;
            for (var dy = -kk; dy <= kk; dy = dy + 1) {
                for (var dx = -kk; dx <= kk; dx = dx + 1) {
                    let d = length(vec2f(f32(cx + dx) + 0.5 - px,
                                         f32(cy + dy) + 0.5 - py));
                    wsum = wsum + clamp(rr + 0.5 - d, 0.0, 1.0);
                }
            }
            wsum = max(wsum, 1e-9);
            for (var dy = -kk; dy <= kk; dy = dy + 1) {
                let yi = cy + dy;
                if yi < 0 || yi >= i32(chh) { continue; }
                for (var dx = -kk; dx <= kk; dx = dx + 1) {
                    let xi = cx + dx;
                    if xi < 0 || xi >= i32(cw) { continue; }
                    let d = length(vec2f(f32(xi) + 0.5 - px, f32(yi) + 0.5 - py));
                    let wgt = clamp(rr + 0.5 - d, 0.0, 1.0);
                    if wgt > 0.0 {
                        deposit(u32(yi) * cw + u32(xi), area * wgt / wsum, scale);
                    }
                }
            }
        }
    }
}
