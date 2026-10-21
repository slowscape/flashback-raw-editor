"""Film Grain v3 — the emulsion model, ported into LoFi Logic.

Adapted from the standalone "Film Grain v3" tool. The picture is *made of*
grain the way a negative is: tones say how much silver belongs where, clumps
of silver are laid down until that is satisfied, and what you see is the
silver — not a noise layer blended on top (that is the standard grain path,
``effects.apply_grain``).

WHAT CHANGED FROM THE STANDALONE TOOL

* **No numba.** The standalone tool JIT-compiles its scatter kernel with
  ``@njit``. Here the scatter runs as a WGSL compute shader on the wgpu device
  LoFi Logic already uses (see ``core/shaders/grain_v3_*.wgsl`` and
  ``gpu.grain_v3_frame``), so the dependency is dropped entirely — nothing new
  is added to requirements. ``build_library`` below was already pure numpy and
  is carried over almost verbatim.
* **Counter-based RNG.** The original walks one sequential xorshift stream per
  tile, which forces the clumps in a tile to be drawn in order. Each clump here
  derives its four random numbers independently from (tile, clump index) via an
  integer hash, so every clump is an independent thread on the GPU and a single
  vectorised row in the numpy fallback. Statistically equivalent, not
  bit-identical to the standalone tool.
* **Additive accumulation.** The original multiplies transmission per clump
  (``canvas *= 1 - value``). GPU threads cannot race-free multiply, so the
  shader accumulates ``-log(1 - value)`` as fixed-point integers with
  ``atomicAdd`` and takes ``exp(-x)`` at the end. Mathematically the same
  product, and because integer addition is exact and order-independent the
  result is deterministic regardless of thread scheduling.
* **No region/preview windowing.** The standalone studio renders a window onto
  a big frame; LoFi Logic always renders whole frames (already downscaled for
  previews), so the region plumbing is dropped.

Everything else — the stock table, the clump library, the tone→silver
inversion and the colour-through-silver composite — is the original model.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# =============================================================================
# STOCKS AND FORMATS  (verbatim from the standalone tool's stocks.py)
# =============================================================================

# Frame size in millimetres, width x height.
FORMATS = {
    "35mm": (36.0, 24.0),
    "645": (56.0, 41.5),
    "6x6": (56.0, 56.0),
    "6x7": (70.0, 56.0),
    "4x5": (127.0, 102.0),
    "8x10": (254.0, 203.0),
}
DEFAULT_FORMAT = "35mm"

# crystal_um  one lump of developed silver, in microns — sets pixel-level noise.
# clump_um    how wide a clump reads. THIS is the grain you see.
# per_clump   average crystals per clump; how hard a film bunches.
# opacity     how solid a crystal is. Decouples grain SIZE from grain NOISE.
#             Cost runs as 1/opacity, noise as its root.
# channels    per-layer gain R/G/B (colour negative's blue layer is coarser).
STOCKS = {
    "Acros 100": dict(
        crystal_um=1.6, clump_um=12.0, per_clump=14, opacity=0.17, strength=1.0,
        channels=(1.0, 1.0, 1.0),
        note="Modern fine-grain black and white. Hardly clumps at all.",
    ),
    "T-Max 400": dict(
        crystal_um=1.9, clump_um=17.0, per_clump=18, opacity=0.20, strength=1.0,
        channels=(1.0, 1.0, 1.0),
        note="Fast but tidy. Tabular crystals, so the grain stays even.",
    ),
    "Portra 400": dict(
        crystal_um=2.4, clump_um=20.0, per_clump=20, opacity=0.19, strength=1.0,
        channels=(1.0, 0.92, 1.2),
        note="Colour negative. Soft dye clouds in loose clusters, a touch "
             "more in the blue.",
    ),
    "Tri-X 400": dict(
        crystal_um=2.4, clump_um=26.0, per_clump=30, opacity=0.22, strength=1.0,
        channels=(1.0, 1.0, 1.0),
        note="The classic. Bunches hard, which is why it reads so chunky.",
    ),
    "HP5 pushed": dict(
        crystal_um=2.8, clump_um=34.0, per_clump=38, opacity=0.26, strength=1.0,
        channels=(1.0, 1.0, 1.0),
        note="Pushed two stops. Coarse, gritty, thin in the shadows.",
    ),
    "Delta 3200": dict(
        crystal_um=3.2, clump_um=44.0, per_clump=46, opacity=0.30, strength=1.0,
        channels=(1.0, 1.0, 1.0),
        note="As coarse as it gets before it stops looking like a photograph.",
    ),
}
DEFAULT_STOCK = "Tri-X 400"


def stock_names():
    return list(STOCKS)


def format_names():
    return list(FORMATS)


def frame_diagonal(fmt):
    """The frame's diagonal in mm — how much it has to be enlarged."""
    w, h = FORMATS.get(fmt, FORMATS[DEFAULT_FORMAT])
    return math.hypot(w, h)


def grain_sizes(stock, image_diagonal_px, fmt=DEFAULT_FORMAT):
    """(clump width, crystal width) in output pixels.

    Grain comes out the same size relative to the frame whatever the scan
    resolution, so the same negative at 2000 px and 6000 px looks identical
    side by side at the same display size.
    """
    microns_per_pixel = frame_diagonal(fmt) * 1000.0 / float(image_diagonal_px)
    spec = STOCKS.get(stock, STOCKS[DEFAULT_STOCK])
    return (spec["clump_um"] / microns_per_pixel,
            spec["crystal_um"] / microns_per_pixel)


# =============================================================================
# MODEL CONSTANTS  (shared with the shaders — keep in sync with grain_v3_*.wgsl)
# =============================================================================

SAMPLES = 3          # canvas pixels per output pixel (sub-pixel clump placement)
TILE = 32            # frame worked in squares this many output pixels across
FLOOR, CEIL = 0.05, 0.95   # where pure black / white sit on the silver's scale
LIFT = 4.0 / 255.0   # how far above black before colour, not grey, shares silver
MIN_DRAWN_CLUMP = 0.7      # finer than this is drawn at this size and quietened
SHAPES = 128         # distinct clumps in the library
INNER = 12           # magnification each patch is drawn at before averaging

# Fixed-point scale for the shader's atomic log-accumulator. 2^20 gives ~1e-6
# resolution on -log(1-v); a pixel would need ~4000 fully-opaque clumps to
# overflow u32, which cannot happen (see the module docstring).
ACCUM_SCALE = 1 << 20


@dataclass
class GrainV3Look:
    """The five controls the tool exposes.

    stock      which film
    fmt        the format it was shot on. Bigger film is enlarged less, so the
               grain lands both finer AND smoother — the only size control
    strength   0 to 1.5; 1.0 is the emulsion as it comes
    mono       shoot it as black and white (any picture, not just a grey one)
    variation  a different roll of the same film
    """
    stock: str = DEFAULT_STOCK
    fmt: str = DEFAULT_FORMAT
    strength: float = 1.0
    mono: bool = False
    variation: int = 1


def diagonal(h, w):
    return math.hypot(float(h), float(w))


def describe(look, full_diagonal):
    """One line a photographer can read back: what this actually did."""
    clump_px, crystal_px = grain_sizes(look.stock, full_diagonal, look.fmt)
    return (f"{look.stock} on {look.fmt}"
            + (" (black and white)" if look.mono else "")
            + f" — clumps {clump_px:.1f} px across, "
              f"crystals {crystal_px:.2f} px, {look.strength * 100:.0f}% strength")


# =============================================================================
# THE CLUMP LIBRARY  (pure numpy — carried over from clumps.py unchanged)
# =============================================================================

def _draw(size, crystals, radius, spread, rng):
    """One clump patch: crystals drawn big, then averaged down to coverage.

    A crystal finer than the drawing grid is deposited as the fraction of a
    cell it actually covers rather than as a disc — without that, every large
    format (where the crystal really is that fine) collapses onto one patch.
    """
    hi = size * INNER
    canvas = np.zeros((hi, hi), dtype=np.float32)
    centre = hi * 0.5

    for _ in range(crystals):
        ang = rng.random() * 2.0 * math.pi
        rad = math.sqrt(rng.random()) * spread
        cx = centre + rad * math.cos(ang)
        cy = centre + rad * math.sin(ang)
        # A spread of crystal sizes: real emulsions are not uniform.
        rr = max(1e-4, radius * INNER * math.exp(rng.normal(0.0, 0.25)))

        if rr < 0.5:
            iy, ix = int(cy), int(cx)
            if 0 <= iy < hi and 0 <= ix < hi:
                area = min(1.0, math.pi * rr * rr)
                if area > canvas[iy, ix]:
                    canvas[iy, ix] = area
            continue

        y0, y1 = int(max(0, cy - rr)), int(min(hi, cy + rr + 2))
        x0, x1 = int(max(0, cx - rr)), int(min(hi, cx + rr + 2))
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None] + 0.5 - cy
        xx = np.arange(x0, x1)[None, :] + 0.5 - cx
        disc = ((yy * yy + xx * xx) < rr * rr).astype(np.float32)
        np.maximum(canvas[y0:y1, x0:x1], disc, out=canvas[y0:y1, x0:x1])

    return canvas.reshape(size, INNER, size, INNER).mean(axis=(1, 3))


@lru_cache(maxsize=8)
def build_library(clump_px, crystal_px, per_clump, seed=1, samples=SAMPLES,
                  opacity=1.0):
    """Patches for one film at one grain size, as sparse silver only.

    Returns (start, off, val, mean_area, size):
      start      (SHAPES+1,) u32 index into off/val per patch
      off        (N,) u32, packed dy<<16 | dx   (patch-local, both < 65536)
      val        (N,) f32 coverage x opacity
      mean_area  average silver one patch lays down, in canvas pixels — this
                 is what ties the library back to a tone (how many clumps a
                 given density needs)
      size       patch width in canvas pixels

    A patch is mostly empty (a clump nine pixels across typically puts silver
    on seven of the eighty-one), so only the silver is stored. Packing dy/dx
    into one u32 keeps the shader down to three storage buffers.
    """
    clump_px = max(0.05, float(clump_px))
    crystal_px = min(max(float(crystal_px), 1e-4), clump_px)
    spread = clump_px * samples * 0.5
    r = crystal_px * samples * 0.5

    size = int(2 * math.ceil(spread + r) + 3)

    rng = np.random.default_rng((int(seed) * 2654435761) % (2 ** 32))
    counts = np.maximum(1, rng.poisson(max(0.6, per_clump), SHAPES))

    dense = np.empty((SHAPES, size, size), dtype=np.float32)
    for i in range(SHAPES):
        dense[i] = _draw(size, int(counts[i]), r, spread * INNER, rng)
    if opacity < 1.0:
        dense *= float(max(opacity, 1e-3))

    start = np.zeros(SHAPES + 1, dtype=np.uint32)
    offs, vals = [], []
    for i in range(SHAPES):
        yy, xx = np.nonzero(dense[i])
        offs.append((yy.astype(np.uint32) << 16) | xx.astype(np.uint32))
        vals.append(dense[i][yy, xx])
        start[i + 1] = start[i] + yy.size

    return (start,
            np.concatenate(offs).astype(np.uint32) if offs else np.zeros(0, np.uint32),
            np.concatenate(vals).astype(np.float32) if vals else np.zeros(0, np.float32),
            float(dense.sum(axis=(1, 2)).mean()),
            size)


# =============================================================================
# TONE -> SILVER SETUP  (shared by the GPU and numpy paths)
# =============================================================================

def _fades_to(clump_px):
    """How loud grain of this size reads. Rises with size, then levels off."""
    d = max(1e-6, float(clump_px))
    return d / math.sqrt(d * d + 1.0)


def _is_grey(arr):
    if arr.ndim < 3 or arr.shape[2] < 3:
        return True
    step = max(1, arr.shape[0] // 64)
    a = arr[::step, ::step]
    return bool(np.allclose(a[:, :, 0], a[:, :, 1], atol=1e-4)
                and np.allclose(a[:, :, 1], a[:, :, 2], atol=1e-4))


def _pool_max(a, tile):
    """Densest value in each square, for deciding how hard to throw at it."""
    h, w = a.shape
    ph = -(-h // tile) * tile
    pw = -(-w // tile) * tile
    pad = np.zeros((ph, pw), dtype=a.dtype)
    pad[:h, :w] = a
    return pad.reshape(ph // tile, tile, pw // tile, tile).max(axis=(1, 3))


def _library_for(look, h, w):
    """The clump library and derived scalars for a look at a frame size.

    Shared by both back-ends so they cannot drift: (spec, lib, faded, strength).
    """
    spec = STOCKS.get(look.stock, STOCKS[DEFAULT_STOCK])
    clump_px, crystal_px = grain_sizes(look.stock, diagonal(h, w), look.fmt)
    drawn = max(MIN_DRAWN_CLUMP, clump_px)
    faded = _fades_to(clump_px) / _fades_to(drawn)
    lib = build_library(
        round(drawn, 2), round(crystal_px * drawn / clump_px, 4),
        spec["per_clump"], int(look.variation), SAMPLES,
        spec.get("opacity", 1.0),
    )
    strength = (float(np.clip(look.strength, 0.0, 1.5))
                * spec["strength"] * faded)
    return spec, lib, strength


def plan_gpu(look, shape, tile_keep, is_grey):
    """Host-side half of the GPU path.

    ``tile_keep`` and ``is_grey`` come from the tile-reduction pass, so the
    image itself never has to come back across the bus. Everything returned is
    ready to upload: the clump library, this frame's Poisson counts, and the
    composite scalars.
    """
    h, w = shape[:2]
    spec, lib, strength = _library_for(look, h, w)
    start, off, val, mean_area, size = lib

    seed = int(look.variation) * 2654435761
    rng = np.random.default_rng(seed % (2 ** 32))
    per_pixel = -math.log1p(-CEIL) / mean_area          # clumps per canvas px
    counts = rng.poisson(
        per_pixel * np.asarray(tile_keep, dtype=np.float64)
        * (TILE * SAMPLES) ** 2)

    channels = spec["channels"]
    if is_grey:
        channels = (1.0, 1.0, 1.0)      # never tint a picture with no colour

    return dict(
        counts=counts, start=start, off=off, val=val, size=size,
        seed=int(seed) & 0x7FFFFFFF, strength=float(strength),
        channels=tuple(float(c) for c in channels),
        mono=bool(look.mono),
    )


def plan(image, look):
    """Everything both back-ends need, derived from the image and the look.

    Returns a dict — see ``gpu.grain_v3_frame`` for the GPU consumer and
    ``render_v3_cpu`` for the numpy one. Kept in one place so the two paths
    cannot drift on the tone maths.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, ch = arr.shape

    spec, lib, strength = _library_for(look, h, w)
    start, off, val, mean_area, size = lib

    if ch >= 3:
        tone = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1]
                + 0.0722 * arr[:, :, 2])
    else:
        tone = arr[:, :, 0]

    span = CEIL - FLOOR
    seed = int(look.variation) * 2654435761
    need = np.clip(FLOOR + span * tone, FLOOR, CEIL)
    keep = (np.log1p(-need) / math.log1p(-CEIL)).astype(np.float32)
    tile_keep = _pool_max(keep, TILE)

    rng = np.random.default_rng(seed % (2 ** 32))
    per_pixel = -math.log1p(-CEIL) / mean_area          # clumps per canvas px
    counts = rng.poisson(
        per_pixel * tile_keep * (TILE * SAMPLES) ** 2).astype(np.int64)

    channels = spec["channels"]
    if ch < 3 or _is_grey(arr):
        channels = (1.0, 1.0, 1.0)      # never tint a picture with no colour

    return dict(
        arr=arr, tone=tone, keep=keep, tile_keep=tile_keep, counts=counts,
        start=start, off=off, val=val, mean_area=mean_area, size=size,
        seed=int(seed) & 0x7FFFFFFF, strength=strength, channels=channels,
        mono=bool(look.mono or ch == 1), shape=(h, w, ch),
    )


# =============================================================================
# NUMPY FALLBACK  (the oracle, and what runs with no usable GPU)
# =============================================================================

def _mix32(v):
    """Integer hash (murmur3 finalizer variant). Must match mix32() in
    grain_v3_scatter.wgsl exactly — the two back-ends share this RNG."""
    v = np.asarray(v, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(16)
    v = (v * np.uint64(0x7feb352d)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(15)
    v = (v * np.uint64(0x846ca68b)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(16)
    return v & np.uint64(0xFFFFFFFF)


def _tile_seed(ty, tx, seed):
    """Must match tile_seed() in grain_v3_scatter.wgsl."""
    ty = np.asarray(ty, dtype=np.uint64)
    tx = np.asarray(tx, dtype=np.uint64)
    v = ((ty * np.uint64(73856093)) ^ (tx * np.uint64(19349663))
         ^ (np.uint64(seed) * np.uint64(83492791)))
    return (v & np.uint64(0xFFFFFFFF)) | np.uint64(1)


INV32 = 1.0 / 4294967296.0
_CHUNK = 1 << 20      # clumps processed per vectorised batch


def render_v3_cpu(image, look=None):
    """Rebuild `image` out of grain, in numpy. Float 0..1, (H, W) or (H, W, C).

    This is the fallback for machines with no usable GPU, and the oracle the
    shader is checked against. It is *slow* — the model throws tens of millions
    of clumps at a full frame, which is exactly why the real path is the
    compute shader.
    """
    look = look or GrainV3Look()
    p = plan(image, look)
    arr, tone = p['arr'], p['tone']
    h, w, ch = p['shape']
    flat_in = np.asarray(image).ndim == 2

    ch_canvas, cw_canvas = h * SAMPLES, w * SAMPLES
    log_accum = np.zeros(ch_canvas * cw_canvas, dtype=np.float64)

    start, off, val = p['start'], p['off'], p['val']
    keep, tile_keep, counts = p['keep'], p['tile_keep'], p['counts']
    half = p['size'] // 2
    nty, ntx = counts.shape

    # -log(1 - v) per patch sample: the additive form of the clump product.
    log_val = -np.log1p(-np.clip(val, 0.0, 0.999)).astype(np.float64)
    p_dy = (off >> np.uint32(16)).astype(np.int64)
    p_dx = (off & np.uint32(0xFFFF)).astype(np.int64)

    ty_idx, tx_idx = np.nonzero(counts > 0)
    for ty, tx in zip(ty_idx.tolist(), tx_idx.tolist()):
        n = int(counts[ty, tx])
        ceiling = float(tile_keep[ty, tx])
        if n <= 0 or ceiling <= 0.0:
            continue
        base = _tile_seed(ty, tx, p['seed'])

        for lo in range(0, n, _CHUNK):
            k = np.arange(lo, min(lo + _CHUNK, n), dtype=np.uint64)
            s = (base ^ (k * np.uint64(0x9E3779B1))) & np.uint64(0xFFFFFFFF)
            r0 = _mix32(s);  r1 = _mix32(r0)
            r2 = _mix32(r1); r3 = _mix32(r2)
            fx = r0.astype(np.float64) * INV32
            fy = r1.astype(np.float64) * INV32
            fa = r2.astype(np.float64) * INV32
            fp = r3.astype(np.float64) * INV32

            iy = ((ty + fy) * TILE).astype(np.int64)
            ix = ((tx + fx) * TILE).astype(np.int64)
            live = (iy < h) & (ix < w)
            if not live.any():
                continue
            iy, ix = iy[live], ix[live]
            keep_here = keep[iy, ix]
            live2 = (fa[live] * ceiling) <= keep_here
            if not live2.any():
                continue

            y0 = (((ty + fy[live][live2]) * TILE * SAMPLES).astype(np.int64) - half)
            x0 = (((tx + fx[live][live2]) * TILE * SAMPLES).astype(np.int64) - half)
            pat = (fp[live][live2] * SHAPES).astype(np.int64)

            # Expand each surviving clump into its sparse silver samples.
            lo_i = start[pat].astype(np.int64)
            hi_i = start[pat + 1].astype(np.int64)
            lens = hi_i - lo_i
            total = int(lens.sum())
            if total == 0:
                continue
            clump_of = np.repeat(np.arange(lens.size), lens)
            within = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
            sidx = np.repeat(lo_i, lens) + within

            yy = y0[clump_of] + p_dy[sidx]
            xx = x0[clump_of] + p_dx[sidx]
            ok = (yy >= 0) & (yy < ch_canvas) & (xx >= 0) & (xx < cw_canvas)
            if not ok.any():
                continue
            np.add.at(log_accum, yy[ok] * cw_canvas + xx[ok], log_val[sidx[ok]])

    canvas = np.exp(-log_accum).reshape(ch_canvas, cw_canvas)
    silver = (1.0 - canvas).reshape(h, SAMPLES, w, SAMPLES).mean(axis=(1, 3))
    return _composite(arr, tone, silver.astype(np.float32), p, flat_in)


def _composite(arr, tone, silver, p, flat_in):
    """Silver + original -> output. Identical maths to the resolve shader."""
    h, w, ch = p['shape']
    span = CEIL - FLOOR
    developed = np.clip((silver - FLOOR) / span, 0.0, 1.0)
    strength, channels = p['strength'], p['channels']
    base = tone

    if p['mono']:
        grey = base + (developed - base) * strength
        out = np.repeat(grey[:, :, None], ch, axis=2)
    else:
        # Colour is carried THROUGH the silver rather than added on top, so hue
        # and saturation survive; near black it hands over to plain addition,
        # because a share of nothing is nothing.
        out = np.empty((h, w, ch), dtype=np.float32)
        safe = np.maximum(base, 1e-4)
        lean = base / (base + LIFT)
        for c in range(ch):
            lit = base + (developed - base) * (strength * channels[min(c, 2)])
            here = arr[:, :, c]
            out[:, :, c] = (lean * (here * (lit / safe))
                            + (1.0 - lean) * (here + (lit - base)))

    np.clip(out, 0.0, 1.0, out=out)
    return out[:, :, 0] if flat_in else out
