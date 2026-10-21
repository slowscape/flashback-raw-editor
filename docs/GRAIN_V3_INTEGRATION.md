# Film Grain v3 / v4 — integration notes

How the standalone **Film Grain v3** tool was folded into LoFi Logic, what was
changed and why, and where to touch things when updating. Written so a future
session can pick this up without re-deriving any of it.

Source of the port: `Apps/lofiLogic/Film Grain v3/` (left in place, untouched —
it is still runnable on its own and is the reference for the model).

---

## The short version

| | |
|---|---|
| **New dependencies** | **none** |
| **Runs on** | GPU (wgpu compute, the device LoFi Logic already uses) |
| **CPU fallback** | yes — numpy, slow, also serves as the test oracle |
| **Cost** | ≈ +110 ms on a full 2072 px render; **0 ms** on scrub previews |
| **Selected by** | `VibeConfig.grain_model` — `'standard'`, `'v3'` or `'v4'` |

The standalone tool JIT-compiles its scatter kernel with **numba**. Numba is not
installed here and was deliberately **not** added: the kernel is a compute shader
instead. `build_library` (the expensive part of the model) was already pure
numpy and carried over almost verbatim.

---

## Files

| File | What it is |
|---|---|
| `core/grain_v3.py` | The model: stock/format tables, clump library, tone→silver planning, numpy fallback + composite. **The oracle.** |
| `core/shaders/grain_v3_tilekeep.wgsl` | Pass A — per-tile reduction |
| `core/shaders/grain_v3_scatter.wgsl` | Pass B — clump scatter (atomics) |
| `core/shaders/grain_v3_resolve.wgsl` | Pass C — silver → picture |
| `core/gpu.py` | `grain_v3_frame()` + 3 pipeline-table rows + `_upload_raw()` |
| `core/processor.py` | `_grain_is_v3()`, `grain_v3_look()`, dispatch in both render paths |
| `core/config.py` | 6 new `VibeConfig` fields |
| `ui/debug_panel.py` | F12 controls + `_combo_widgets` schema table |
| `tests/test_grain_v3.py` | Model invariants + GPU-vs-oracle parity |
| `core/grain_v4.py` | **v4**: the same emulsion with no library — clumps grow their own crystals |
| `core/shaders/grain_v4_scatter.wgsl` | **v4** scatter (replaces pass B only) |
| `tests/test_grain_v4.py` | v4 invariants, v3-vs-v4 agreement, shared-plumbing regression |

---

## The three deliberate deviations from the standalone tool

These are the things that would look like bugs if you diffed the two.

**1. Counter-based RNG instead of a sequential per-tile stream.**
The original walks one xorshift stream per tile, so clump *k* can only be drawn
after clump *k−1* — fine for a serial kernel, fatal for parallelism. Here each
clump hashes `(tile, clump index)` into its four randoms independently, so every
clump is its own GPU thread and its own vectorised numpy row.
`mix32`/`tile_seed` are duplicated in `grain_v3_scatter.wgsl` and
`core/grain_v3.py` (`_mix32`/`_tile_seed`) and **must be edited together** — that
is the one real coupling in this port.

*Consequence:* not bit-identical to the standalone tool. Statistically identical
— verified: mean, std and grain magnitude match to 4 decimals on every stock.

**2. Additive log accumulation instead of multiplicative transmission.**
The original does `canvas *= (1 - v)` per clump. GPU threads cannot race-free
multiply, so the shader accumulates `-log(1 - v)` into `array<atomic<u32>>` as
fixed-point (`ACCUM_SCALE = 2²⁰`) and the resolve pass takes `exp(-x)`. Same
product. Because integer addition is exact and order-independent, the output does
not depend on thread scheduling — **the same frame renders bit-identically every
time**, which is what preview-matches-export needs.

**3. No region/preview windowing.** The standalone studio renders a window onto a
big frame. LoFi Logic always renders whole frames (previews are already
downscaled upstream), so that plumbing is gone.

---

## How a frame is rendered (GPU path)

`gpu.grain_v3_frame(frame, look)` — three passes, one small readback:

```
A  grain_v3_tilekeep   1 thread per 32×32 tile
   → per tile: max `keep`, max chroma deviation      [2 floats/tile]
        ↓ readback (only ~25 KB at full res — the image never leaves the GPU)
   host: Poisson clump counts per tile, clump library (cached), composite scalars
        ↓ upload counts + library
B  grain_v3_scatter    1 thread per clump, dispatch (clumps ÷ 64, n_tiles)
   → atomicAdd -log(1-v) into the fixed-point canvas   [w·h·9 u32 ≈ 115 MB @ full res]
        ↓
C  grain_v3_resolve    1 thread per output pixel
   → exp(-accum) → average the SAMPLES² cell → develop → composite → dst texture
```

The readback in the middle is why the host can draw a *Poisson* count per tile
(numpy) without pulling the image back. It also means the GPU and CPU paths draw
counts identically.

**Why `keep` is recomputed in the scatter shader** rather than passed as a
texture: it is ~10 ALU ops from the source pixel, cheaper than another full-frame
texture and its bandwidth.

---

## Where it plugs into the pipeline

Same slot as the standard grain — post-LUT, display-referred sRGB, in
`processor._resident_post_lut_stages`:

```
LUT → CA → edge softness → softness → [GRAIN] → sharpen → sRGB
```

`enable_grain` is the master switch for **both** models; `grain_model` picks
which one runs. They are mutually exclusive (`if v3 … elif standard …`) — v3
replaces the noise-blend stage rather than stacking on it, and carries its own
strength (`grain_v3_strength_pct`), so `grain_strength_pct` is ignored under v3.

Two dispatch sites, both must stay in step:
* `_resident_post_lut_stages` → `gpu.grain_v3_frame` (the real path)
* `_render`'s per-op block → `render_v3_cpu` (no-GPU fallback)

`run_resident` turns a `None` from any stage into a whole-chain `None`, so if
`grain_v3_frame` bails (no GPU, or a frame too big to dispatch) the render falls
through to the per-op CPU path automatically. **Scrub previews (`downscale=True`)
skip the post-LUT tail entirely**, which is why v3 costs nothing while dragging a
slider — it only runs on the committed full render.

---

## Config fields (`VibeConfig`)

```python
grain_model:            str   = 'standard'    # 'standard' | 'v3'
grain_v3_stock:         str   = 'Tri-X 400'
grain_v3_format:        str   = '35mm'
grain_v3_strength_pct:  float = 100.0         # 0–150; the model's 0–1.5 × 100
grain_v3_mono:          bool  = False
grain_v3_variation:     int   = 1             # 1–40
```

They ride the existing dataclass serialization, so saved vibes, projects, and
copy/paste picked them up for free. Every read in `processor.py` uses `getattr`
with a default, so vibes saved before v3 existed load unchanged.

---

## The F12 controls

All five of the standalone tool's controls, in LoFi Logic styling, in the Grain
section: **Grain Model** (dropdown) → then either *Grain Strength* (standard) or
*Film / Shot on / v3 Strength / Variation / black-and-white* (v3). The
film's one-line note shows under the Film picker. Rows swap visibility with the
model via `_update_grain_rows()`.

Dropdowns needed a third schema table — `_combo_widgets` — because the existing
`_bool_widgets`/`_numeric_widgets` tables assume `setChecked`/`setValue`. Values
live in Qt item *data*, not the label, so labels can be reworded freely.

> **Gotcha already hit:** a fresh `QComboBox` sits at index 0, so without seeding
> it from `VibeConfig()` the first `update_config()` wrote *Acros 100* over the
> `Tri-X 400` default. The seeding loop at the end of `__init__` is load-bearing.

---

## Updating it

* **New film stock / format** → `STOCKS` or `FORMATS` in `core/grain_v3.py`.
  The dropdowns build themselves from `stock_names()`/`format_names()`; nothing
  else to touch.
* **Model constants** (`SAMPLES`, `TILE`, `FLOOR`, `CEIL`, `LIFT`) live in
  `core/grain_v3.py` and are pushed to the shaders through uniforms — but
  `SHAPES`, and the RNG, are duplicated in WGSL. Grep both.
* **Composite maths** is in two places by necessity: `_composite()` (numpy) and
  `grain_v3_resolve.wgsl`. Change together; `tests/test_grain_v3.py` catches drift.
* **Too slow?** `opacity` in `STOCKS` is one number per film — cost runs as
  `1/opacity`, so doubling it roughly halves render time and costs ~40 % more
  grain noise (the standalone README's own guidance, still true here).

---

## Measured

RTX-class GPU, synthetic ramp, warm:

| | preview (690 px) | full render (2072 px) |
|---|---|---|
| Delta 3200 | 37 ms | 221 ms |
| Tri-X 400 | 37 ms | 243 ms |
| Acros 100 | 39 ms | 351 ms |

(Finer films are slower — smaller crystals, so more of them.) In the full
pipeline v3 adds ≈ 110 ms to a 2072 px render. The standalone tool's numba CPU
kernel needed 1.7–6.8 s for a *smaller* frame.

Parity vs the numpy oracle: grain magnitude **0.0 %** apart, mean/std identical
to 5 decimals, and bit-identical across repeat runs.


---

## v4 — the same emulsion without the lookup table

v3's 128-shape clump library was a speed hack from when the crystal-level maths
was too slow to run per clump. The compute shader removed that constraint, so
**v4 grows every clump where it lands**: it draws its own crystal count, then
gives each crystal its own angle, its own radius within the disc (sqrt-uniform,
so the disc fills evenly) and its own size from the same lognormal spread. Each
clump is unique instead of one of 128 reused shapes.

**Shared with v3, not copied** (`core/grain_v4.py` imports them, so v3 is
untouched): the `STOCKS` table and formats, `grain_sizes` / `MIN_DRAWN_CLUMP` /
the size fade, the whole tone model, `_composite`, and the GPU tile-reduction
and resolve passes. Only pass B differs — `grain_v4_scatter.wgsl` drops in where
`grain_v3_scatter.wgsl` goes, between the same two shaders.

**The one number that had to change.** v3 *measures* the silver its patches
carry (`mean_area`), and that number is what ties clump count back to tone. With
no patches to measure, v4 computes the expectation:

```
mean_area = per_clump x (pi x r^2 x e^0.125) x opacity
```

The `e^0.125` is `E[exp(2 * 0.25 * N)]` — the lognormal spread on crystal radius
means the average crystal *area* is 13% larger than the nominal radius implies.
Get that wrong and exposure drifts, which is what
`test_exposure_does_not_move` guards.

**Config is shared.** `grain_v3_*` are the five emulsion controls for *both*
engines — they describe the film, not the renderer — so switching v3 <-> v4
keeps your settings. `grain_model` picks the engine.

**Measured** (RTX-class, synthetic ramp, warm):

| | preview (690 px) | full render (2072 px) |
|---|---|---|
| v3 Tri-X 400 | 31 ms | 234 ms |
| v4 Tri-X 400 | 75 ms | 260 ms |
| v4 Acros 100 | 59 ms | 497 ms |

v4 runs 1.1-2.4x v3 — far less than the raw crystal count suggests, because a v4
clump carries more silver (no union loss inside the clump), so fewer are thrown.
v4 lands a consistent **+9-10% grainier** than v3 for the same film; that is the
model, not drift, and `test_lands_near_v3_for_the_same_film` pins it.

Parity: v4 GPU vs its numpy oracle is **0.0%** apart on grain magnitude, and
bit-identical across repeat runs.
