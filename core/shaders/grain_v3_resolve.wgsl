// Film Grain v3, pass C of 3: silver -> picture.
//
// One thread per output pixel. Undoes the scatter pass's fixed-point log
// accumulator (exp(-x) back to transmission), area-averages the SAMPLES x
// SAMPLES canvas cell behind this pixel to get the silver, then composites.
//
// Black and white is simply the silver. Colour is carried THROUGH the silver
// by multiplying rather than having a shift added on top, so hue and
// saturation come out untouched; near black it hands over to plain addition,
// because a share of nothing is nothing and multiplying alone would leave a
// pure black black however much silver landed on it.
//
// Must stay in step with _composite() in core/grain_v3.py, which is the oracle.

struct Uniforms {
    dims:    vec4u,    // w, h, canvas_w, samples
    consts:  vec4f,    // span, floor, accum_scale, lift
    look:    vec4f,    // strength, ch_r, ch_g, ch_b
    flags:   vec4u,    // mono, _, _, _
}

@group(0) @binding(0) var          src:    texture_2d<f32>;
@group(0) @binding(1) var<storage, read>  accum: array<u32>;
@group(0) @binding(2) var          dst:    texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform>        u:     Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let w = u.dims.x;
    let h = u.dims.y;
    if id.x >= w || id.y >= h { return; }

    let cw = u.dims.z;
    let s  = u.dims.w;

    // Average the canvas cell behind this output pixel.
    var total = 0.0;
    let by = id.y * s;
    let bx = id.x * s;
    for (var dy = 0u; dy < s; dy = dy + 1u) {
        for (var dx = 0u; dx < s; dx = dx + 1u) {
            let a = accum[(by + dy) * cw + (bx + dx)];
            // transmission = exp(-sum(-log(1-v))) ; silver = 1 - transmission
            total = total + (1.0 - exp(-f32(a) / u.consts.z));
        }
    }
    let silver = total / f32(s * s);

    let span  = u.consts.x;
    let floor_v = u.consts.y;
    let developed = clamp((silver - floor_v) / span, 0.0, 1.0);

    let c = textureLoad(src, vec2i(i32(id.x), i32(id.y)), 0);
    let base = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    let strength = u.look.x;

    var out: vec3f;
    if u.flags.x == 1u {
        // Black and white: there is no colour to carry, the picture IS the silver.
        let grey = base + (developed - base) * strength;
        out = vec3f(grey, grey, grey);
    } else {
        let safe = max(base, 1e-4);
        let lean = base / (base + u.consts.w);
        let gains = vec3f(u.look.y, u.look.z, u.look.w);
        let here = vec3f(c.r, c.g, c.b);
        let lit = vec3f(base) + (vec3f(developed) - vec3f(base)) * (strength * gains);
        out = lean * (here * (lit / safe))
            + (1.0 - lean) * (here + (lit - vec3f(base)));
    }

    textureStore(dst, vec2i(i32(id.x), i32(id.y)),
                 vec4f(clamp(out, vec3f(0.0), vec3f(1.0)), 1.0));
}
