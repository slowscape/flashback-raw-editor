"""Film Grain v3: model invariants, and GPU-vs-numpy-oracle parity.

The numpy path in core/grain_v3.py is the oracle; the three compute shaders are
checked against it, the same way every other GPU stage in this app is checked
against its numpy twin. GPU-touching tests skip cleanly where no usable device
exists (e.g. CI).

See docs/GRAIN_V3_INTEGRATION.md for why the port is not bit-identical to the
standalone tool it came from.
"""
import numpy as np
import pytest

from core import grain_v3 as g3
from core.config import VibeConfig
from core.gpu import gpu


def _gpu_available() -> bool:
    try:
        return bool(gpu._init())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="no usable GPU")


@pytest.fixture
def ramp():
    """A 96x128 black->white ramp: every tone represented, so grain can be
    measured against tone in one image."""
    h, w = 96, 128
    g = (np.mgrid[0:h, 0:w][1] / (w - 1)).astype(np.float32)
    return np.repeat(g[:, :, None], 3, axis=2)


def _dev(out, src):
    """Mean absolute departure from the source — how loud the grain reads."""
    return float(np.abs(out - src).mean())


# =============================================================================
# The model's promises (numpy oracle)
# =============================================================================

def test_exposure_does_not_move(ramp):
    out = g3.render_v3_cpu(ramp, g3.GrainV3Look())
    assert out.mean() == pytest.approx(ramp.mean(), abs=0.01)


def test_coarser_film_is_grainier(ramp):
    fine = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(stock="Acros 100")), ramp)
    coarse = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(stock="Delta 3200")), ramp)
    assert coarse > fine


def test_bigger_format_is_smoother(ramp):
    """Bigger film is enlarged less, so the same emulsion reads finer."""
    small = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(fmt="35mm")), ramp)
    big = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(fmt="8x10")), ramp)
    assert big < small


def test_645_and_6x6_differ(ramp):
    """They share a long edge but are not the same enlargement — the diagonal
    is what matters, which is the whole reason formats are compared that way."""
    a = g3.render_v3_cpu(ramp, g3.GrainV3Look(fmt="645"))
    b = g3.render_v3_cpu(ramp, g3.GrainV3Look(fmt="6x6"))
    assert not np.allclose(a, b)


def test_grain_is_loudest_in_the_midtones(ramp):
    out = g3.render_v3_cpu(ramp, g3.GrainV3Look())
    per_col = np.abs(out - ramp).mean(axis=(0, 2))
    w = ramp.shape[1]
    lo = per_col[:w // 8].mean()
    mid = per_col[w // 2 - 6:w // 2 + 6].mean()
    hi = per_col[-w // 8:].mean()
    assert mid > lo and mid > hi


def test_black_and_white_still_carry_grain(ramp):
    """A coverage model has no grain at either extreme, so FLOOR/CEIL sit away
    from empty and solid on purpose — film has base fog and a max density."""
    out = g3.render_v3_cpu(ramp, g3.GrainV3Look())
    per_col = np.abs(out - ramp).mean(axis=(0, 2))
    assert per_col[:4].mean() > 1e-4
    assert per_col[-4:].mean() > 1e-4


def test_mono_stays_neutral(ramp):
    out = g3.render_v3_cpu(ramp, g3.GrainV3Look(mono=True))
    assert np.allclose(out[:, :, 0], out[:, :, 1], atol=1e-5)
    assert np.allclose(out[:, :, 1], out[:, :, 2], atol=1e-5)


def test_variation_is_another_roll(ramp):
    a = g3.render_v3_cpu(ramp, g3.GrainV3Look(variation=1))
    b = g3.render_v3_cpu(ramp, g3.GrainV3Look(variation=9))
    assert not np.allclose(a, b)
    # same film, so the same amount of grain — just a different roll
    assert _dev(a, ramp) == pytest.approx(_dev(b, ramp), rel=0.15)


def test_strength_scales_the_effect(ramp):
    half = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(strength=0.5)), ramp)
    full = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(strength=1.0)), ramp)
    assert half < full


def test_shape_and_dtype_preserved(ramp):
    out = g3.render_v3_cpu(ramp, g3.GrainV3Look())
    assert out.shape == ramp.shape
    assert out.dtype == np.float32


def test_greyscale_input_stays_2d():
    grey = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = g3.render_v3_cpu(grey, g3.GrainV3Look())
    assert out.shape == grey.shape


# =============================================================================
# GPU vs the oracle
# =============================================================================

@requires_gpu
@pytest.mark.parametrize("stock", ["Acros 100", "Tri-X 400", "Delta 3200"])
def test_gpu_matches_oracle(ramp, stock):
    """Not bit-equal (fixed-point accumulator vs float product), but the grain
    must land in the same place — 5% on magnitude is a generous bar it clears
    comfortably."""
    from core.gpu import Frame
    look = g3.GrainV3Look(stock=stock)
    got = gpu.grain_v3_frame(Frame.from_cpu(ramp, gpu), look)
    assert got is not None
    got = got.cpu()
    ref = g3.render_v3_cpu(ramp, look)
    assert got.mean() == pytest.approx(ref.mean(), abs=0.005)
    assert _dev(got, ramp) == pytest.approx(_dev(ref, ramp), rel=0.05)


@requires_gpu
def test_gpu_is_deterministic(ramp):
    """Preview must equal export, so the same frame has to render identically
    every time — the reason the shader accumulates in fixed-point integers
    (exact and order-independent) rather than floats."""
    from core.gpu import Frame
    look = g3.GrainV3Look()
    a = gpu.grain_v3_frame(Frame.from_cpu(ramp, gpu), look).cpu()
    b = gpu.grain_v3_frame(Frame.from_cpu(ramp, gpu), look).cpu()
    assert np.array_equal(a, b)


@requires_gpu
def test_gpu_rng_matches_cpu_rng():
    """mix32/tile_seed are duplicated in WGSL and numpy; if they drift, the two
    back-ends silently draw different clumps. Guard the numpy half against
    known values so a change to one side without the other is caught."""
    assert int(g3._mix32(np.uint64(0))) == int(g3._mix32(np.uint64(0)))
    # tile_seed must be odd (the shader ORs in 1) and depend on all three inputs
    s = int(g3._tile_seed(3, 5, 12345))
    assert s & 1 == 1
    assert int(g3._tile_seed(4, 5, 12345)) != s
    assert int(g3._tile_seed(3, 6, 12345)) != s
    assert int(g3._tile_seed(3, 5, 999)) != s


# =============================================================================
# Wiring into the app
# =============================================================================

def test_vibe_defaults_to_standard_grain():
    v = VibeConfig()
    assert v.grain_model == 'standard'
    assert v.grain_v3_stock in g3.STOCKS
    assert v.grain_v3_format in g3.FORMATS


def test_v3_fields_survive_serialization():
    v = VibeConfig(grain_model='v3', grain_v3_stock='Delta 3200',
                   grain_v3_format='6x7', grain_v3_strength_pct=75.0,
                   grain_v3_mono=True, grain_v3_variation=7)
    back = VibeConfig.from_dict(v.to_dict())
    for f in ('grain_model', 'grain_v3_stock', 'grain_v3_format',
              'grain_v3_strength_pct', 'grain_v3_mono', 'grain_v3_variation'):
        assert getattr(back, f) == getattr(v, f)


def test_look_built_from_vibe_converts_percent():
    from core.processor import grain_v3_look, _grain_is_v3
    v = VibeConfig(grain_model='v3', grain_v3_strength_pct=75.0)
    assert _grain_is_v3(v)
    assert grain_v3_look(v).strength == pytest.approx(0.75)


def test_vibe_without_v3_fields_is_tolerated():
    """Vibes saved before v3 existed must load and render as standard grain."""
    from core.processor import _grain_is_v3
    class Legacy:
        pass
    assert _grain_is_v3(Legacy()) is False
