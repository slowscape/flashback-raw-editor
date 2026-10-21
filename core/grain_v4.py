"""Film Grain v4 — the same emulsion, drawn without a lookup table.

v3 rasterises a library of 128 clump patches up front and stamps them; that was
a speed hack from when the crystal-level maths was too slow to do per clump.
Running the scatter as a compute shader removed that constraint, so v4 grows
every clump's crystals where it lands: each clump is unique, not one of 128
shapes reused across the frame.

WHAT IS SHARED WITH v3 (imported, never copied — v3 itself is untouched)

  * the emulsion table (STOCKS) and the formats, so the six films and six
    sizes mean exactly the same thing in both
  * the enlargement maths (grain_sizes), MIN_DRAWN_CLUMP and the size fade
  * the tone model: FLOOR/CEIL, the `keep` inversion, per-tile Poisson counts
  * the composite (_composite), and the GPU tile-reduction and resolve passes

WHAT IS DIFFERENT

  * **No library, no lru_cache, no patch buffers.** Nothing is precomputed and
    nothing is uploaded but the per-tile clump counts. Where v3 needs three
    storage buffers of patch geometry, v4 needs none.
  * **Crystals are grown per clump.** The count varies per clump, and each
    crystal gets its own angle, its own radius within the disc (sqrt-uniform, so
    the disc fills evenly instead of bunching at the centre) and its own size
    from the same lognormal spread v3 draws its patches with. Clump-to-clump
    variety is continuous rather than 128-fold.
  * **mean_area is analytic.** v3 measures the silver its patches actually
    carry; with no patches to measure, v4 computes the expectation directly —
    crystals per clump x mean crystal area x opacity. That number is what ties
    clump count back to tone, so it is what keeps exposure where v3 puts it.

Same five controls, same six films, same six formats.
"""

import math
from dataclasses import dataclass

import numpy as np

from .grain_v3 import (
    # the emulsion, verbatim
    STOCKS, FORMATS, DEFAULT_STOCK, DEFAULT_FORMAT,
    stock_names, format_names, grain_sizes, frame_diagonal, diagonal,
    # the tone model, verbatim
    SAMPLES, TILE, FLOOR, CEIL, LIFT, MIN_DRAWN_CLUMP, ACCUM_SCALE,
    # shared helpers
    _fades_to, _is_grey, _pool_max, _composite, _mix32, _tile_seed, INV32,
)

# Crystal radius is scattered lognormally: rr = r * exp(0.25 * N(0,1)), the same
# spread v3's _draw uses. Area therefore carries E[exp(2 * 0.25 * N)] = e^0.125
# more than the nominal radius alone suggests, and mean_area has to know that or
# the frame lands well off where v3 puts it.
CRYSTAL_SIGMA = 0.25
CRYSTAL_AREA_FACTOR = math.exp(2.0 * CRYSTAL_SIGMA * CRYSTAL_SIGMA)

# Loop bound in the shader. per_clump tops out at 46 (Delta 3200) and the
# Gaussian count would need about +12 sigma to reach this, which will not happen.
MAX_CRYSTALS = 128

# Below this radius (canvas pixels) a crystal is finer than the grid it is drawn
# on, so its whole area drops into the pixel it landed in — the same call v3's
# _draw makes at its own resolution. Above it the area is spread over the
# footprint, so a big crystal reads as a disc and not a hot dot.
SUBPIXEL_R = 0.5


@dataclass
class GrainV4Look:
    """The same five controls v3 exposes.

    stock      which film
    fmt        the format it was shot on — the only grain-size control
    strength   0 to 1.5; 1.0 is the emulsion as it comes
    mono       shoot it as black and white
    variation  a different roll of the same film
    """
    stock: str = DEFAULT_STOCK
    fmt: str = DEFAULT_FORMAT
    strength: float = 1.0
    mono: bool = False
    variation: int = 1


def clump_params(look, h, w):
    """Everything the crystal model needs at this frame size.

    Returns (spec, spread, r, per_clump, opacity, mean_area, strength), with
    `spread` and `r` in canvas pixels — the disc a clump's crystals fall in, and
    the nominal crystal radius. Mirrors v3's _library_for, so the two models
    agree on size for the same film and format.
    """
    spec = STOCKS.get(look.stock, STOCKS[DEFAULT_STOCK])
    clump_px, crystal_px = grain_sizes(look.stock, diagonal(h, w), look.fmt)
    drawn = max(MIN_DRAWN_CLUMP, clump_px)
    faded = _fades_to(clump_px) / _fades_to(drawn)

    # v3 shrinks the crystal by the same factor it grows the clump to the
    # minimum drawn size, so the two stay in proportion.
    crystal_scaled = crystal_px * drawn / clump_px
    spread = drawn * SAMPLES * 0.5
    r = max(1e-4, crystal_scaled * SAMPLES * 0.5)

    per_clump = float(spec["per_clump"])
    opacity = float(spec.get("opacity", 1.0))

    # The silver one clump lays down, in canvas pixels. Analytic where v3
    # measures: crystals x mean crystal area x opacity.
    mean_area = per_clump * math.pi * r * r * CRYSTAL_AREA_FACTOR * opacity

    strength = (float(np.clip(look.strength, 0.0, 1.5))
                * spec["strength"] * faded)
    return spec, spread, r, per_clump, opacity, mean_area, strength


def plan_gpu_v4(look, shape, tile_keep, is_grey):
    """Host-side half of the GPU path — the v4 twin of grain_v3.plan_gpu.

    Far less to hand over than v3: no patch geometry at all, just this frame's
    Poisson clump counts and the handful of scalars the shader grows crystals
    from.
    """
    h, w = shape[:2]
    spec, spread, r, per_clump, opacity, mean_area, strength = clump_params(look, h, w)

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
        counts=counts, seed=int(seed) & 0x7FFFFFFF,
        spread=float(spread), radius=float(r),
        per_clump=float(per_clump), opacity=float(opacity),
        strength=float(strength),
        channels=tuple(float(c) for c in channels),
        mono=bool(look.mono),
    )


# =============================================================================
# NUMPY FALLBACK — the oracle the shader is checked against
# =============================================================================

def _gauss(u1, u2):
    """Box-Muller. Must match gauss() in grain_v4_scatter.wgsl."""
    u1 = np.maximum(u1, 1e-7)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def _u(x):
    """u32 hash output -> uniform [0,1). Matches the shader's INV32 scaling."""
    return x.astype(np.float64) * INV32


def _deposit_point(log_accum, py, px, v, cheight, cwidth):
    """A sub-pixel crystal drops whole into the pixel it fell in."""
    iy = np.floor(py).astype(np.int64)
    ix = np.floor(px).astype(np.int64)
    ok = (iy >= 0) & (iy < cheight) & (ix >= 0) & (ix < cwidth)
    if not ok.any():
        return
    np.add.at(log_accum, iy[ok] * cwidth + ix[ok],
              -np.log1p(-np.clip(v[ok], 0.0, 0.999)))


def _deposit_disc(log_accum, py, px, rr, area, cheight, cwidth):
    """A crystal wider than a pixel, spread over its footprint by coverage."""
    k = int(min(3, max(1, math.ceil(float(rr.max()) + 0.5))))
    cy = np.floor(py).astype(np.int64)
    cx = np.floor(px).astype(np.int64)
    offs = [(dy, dx) for dy in range(-k, k + 1) for dx in range(-k, k + 1)]

    weights = [np.clip(rr + 0.5 - np.hypot((cy + dy + 0.5) - py,
                                           (cx + dx + 0.5) - px), 0.0, 1.0)
               for dy, dx in offs]
    wsum = np.maximum(np.sum(weights, axis=0), 1e-9)

    for (dy, dx), wgt in zip(offs, weights):
        share = area * (wgt / wsum)
        live = share > 1e-9
        if not live.any():
            continue
        iy = cy[live] + dy
        ix = cx[live] + dx
        ok = (iy >= 0) & (iy < cheight) & (ix >= 0) & (ix < cwidth)
        if not ok.any():
            continue
        np.add.at(log_accum, iy[ok] * cwidth + ix[ok],
                  -np.log1p(-np.clip(share[live][ok], 0.0, 0.999)))


def render_v4_cpu(image, look=None):
    """Rebuild `image` out of procedurally grown grain, in numpy.

    The fallback for machines with no usable GPU, and the oracle the shader is
    checked against. Slow by construction — it grows every crystal of every
    clump, which is exactly the work the compute shader exists to absorb.
    """
    look = look or GrainV4Look()
    arr = np.asarray(image, dtype=np.float32)
    flat_in = arr.ndim == 2
    if flat_in:
        arr = arr[:, :, None]
    h, w, ch = arr.shape

    spec, spread, r, per_clump, opacity, mean_area, strength = clump_params(look, h, w)

    tone = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            if ch >= 3 else arr[:, :, 0])

    span = CEIL - FLOOR
    seed = int(look.variation) * 2654435761
    need = np.clip(FLOOR + span * tone, FLOOR, CEIL)
    keep = (np.log1p(-need) / math.log1p(-CEIL)).astype(np.float32)
    tile_keep = _pool_max(keep, TILE)

    rng = np.random.default_rng(seed % (2 ** 32))
    per_pixel = -math.log1p(-CEIL) / mean_area
    counts = rng.poisson(per_pixel * tile_keep * (TILE * SAMPLES) ** 2).astype(np.int64)

    channels = spec["channels"]
    if ch < 3 or _is_grey(arr):
        channels = (1.0, 1.0, 1.0)

    cheight, cwidth = h * SAMPLES, w * SAMPLES
    log_accum = np.zeros(cheight * cwidth, dtype=np.float64)
    seed32 = int(seed) & 0x7FFFFFFF

    ty_idx, tx_idx = np.nonzero(counts > 0)
    for ty, tx in zip(ty_idx.tolist(), tx_idx.tolist()):
        n = int(counts[ty, tx])
        ceiling = float(tile_keep[ty, tx])
        if n <= 0 or ceiling <= 0.0:
            continue

        base = _tile_seed(ty, tx, seed32)
        k = np.arange(n, dtype=np.uint64)
        s = (base ^ (k * np.uint64(0x9E3779B1))) & np.uint64(0xFFFFFFFF)
        r0 = _mix32(s)
        r1 = _mix32(r0)
        r2 = _mix32(r1)
        r3 = _mix32(r2)
        r4 = _mix32(r3)
        fx, fy, fa = _u(r0), _u(r1), _u(r2)

        iy = ((ty + fy) * TILE).astype(np.int64)
        ix = ((tx + fx) * TILE).astype(np.int64)
        live = (iy < h) & (ix < w)
        if not live.any():
            continue
        live &= (fa * ceiling) <= keep[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
        if not live.any():
            continue

        # Clump centres in canvas pixels, kept fractional so clumps land
        # between pixels rather than snapping to the grid.
        ccy = (ty + fy[live]) * TILE * SAMPLES
        ccx = (tx + fx[live]) * TILE * SAMPLES

        # How many crystals this clump grew.
        nz = _gauss(_u(r3[live]), _u(r4[live]))
        ncry = np.clip(np.rint(per_clump + math.sqrt(per_clump) * nz),
                       1, MAX_CRYSTALS).astype(np.int64)

        cry_base = r4[live]
        total = int(ncry.sum())
        if total == 0:
            continue
        clump_of = np.repeat(np.arange(ncry.size), ncry)
        j = (np.arange(total, dtype=np.int64)
             - np.repeat(np.cumsum(ncry) - ncry, ncry)).astype(np.uint64)

        cs = _mix32(cry_base[clump_of] ^ (j * np.uint64(0x85EBCA6B)))
        c0 = _mix32(cs)
        c1 = _mix32(c0)
        c2 = _mix32(c1)
        c3 = _mix32(c2)

        ang = 2.0 * math.pi * _u(c0)
        rad = np.sqrt(_u(c1)) * spread
        rr = np.maximum(1e-4, r * np.exp(CRYSTAL_SIGMA * _gauss(_u(c2), _u(c3))))

        py = ccy[clump_of] + rad * np.sin(ang)
        px = ccx[clump_of] + rad * np.cos(ang)
        area = np.minimum(math.pi * rr * rr * opacity, 0.999)

        small = rr <= SUBPIXEL_R
        if small.any():
            _deposit_point(log_accum, py[small], px[small], area[small],
                           cheight, cwidth)
        big = ~small
        if big.any():
            _deposit_disc(log_accum, py[big], px[big], rr[big], area[big],
                          cheight, cwidth)

    canvas = np.exp(-log_accum).reshape(cheight, cwidth)
    silver = (1.0 - canvas).reshape(h, SAMPLES, w, SAMPLES).mean(axis=(1, 3))

    p = dict(shape=(h, w, ch), strength=strength,
             channels=channels, mono=bool(look.mono or ch == 1))
    return _composite(arr, tone, silver.astype(np.float32), p, flat_in)
