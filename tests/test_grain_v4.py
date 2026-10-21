"""Film Grain v4: the emulsion model drawn without a clump library.

v4 shares v3's stock table, size maths and tone model, so most of what is
checked here is that it still behaves like film. What is specific to v4 is that
nothing is precomputed — every clump grows its own crystals — so the two things
worth guarding are that the analytic mean_area keeps exposure where v3 puts it,
and that the shader and the numpy oracle draw the same crystals.

See docs/GRAIN_V3_INTEGRATION.md for the shared architecture.
"""
import math

import numpy as np
import pytest

from core import grain_v3 as g3
from core import grain_v4 as g4
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
    h, w = 96, 128
    g = (np.mgrid[0:h, 0:w][1] / (w - 1)).astype(np.float32)
    return np.repeat(g[:, :, None], 3, axis=2)


def _dev(out, src):
    return float(np.abs(out - src).mean())


# =============================================================================
# The emulsion is the same one
# =============================================================================

def test_same_films_and_formats_as_v3():
    assert g4.stock_names() == g3.stock_names()
    assert g4.format_names() == g3.format_names()
    assert g4.STOCKS is g3.STOCKS       # the table itself, not a copy


def test_no_lookup_table_is_built():
    """The whole point of v4: nothing is precomputed and nothing is cached."""
    assert not hasattr(g4, 'build_library')
    plan = g4.plan_gpu_v4(g4.GrainV4Look(), (96, 128),
                          np.ones((3, 4), dtype=np.float32), False)
    # v3 hands the shader patch geometry; v4 hands it scalars and nothing else.
    assert 'start' not in plan and 'off' not in plan and 'val' not in plan
    assert {'spread', 'radius', 'per_clump', 'opacity'} <= set(plan)


@pytest.mark.parametrize("stock", g3.stock_names())
def test_clump_params_track_v3_sizes(stock):
    """v4 must derive the same clump/crystal geometry v3 does for a given film
    and format, or the two models would not be the same emulsion."""
    look = g4.GrainV4Look(stock=stock)
    _, spread, r, per_clump, opacity, mean_area, _ = g4.clump_params(look, 1544, 2072)
    clump_px, crystal_px = g3.grain_sizes(stock, g3.diagonal(1544, 2072), "35mm")
    drawn = max(g3.MIN_DRAWN_CLUMP, clump_px)
    assert spread == pytest.approx(drawn * g3.SAMPLES * 0.5, rel=1e-6)
    assert per_clump == g3.STOCKS[stock]["per_clump"]
    assert opacity == g3.STOCKS[stock]["opacity"]
    assert mean_area > 0


# =============================================================================
# The model still behaves like film (numpy oracle)
# =============================================================================

def test_exposure_does_not_move(ramp):
    """The analytic mean_area is what ties clump count to tone. If it drifts,
    this is what catches it."""
    out = g4.render_v4_cpu(ramp, g4.GrainV4Look())
    assert out.mean() == pytest.approx(ramp.mean(), abs=0.01)


def test_coarser_film_is_grainier(ramp):
    fine = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(stock="Acros 100")), ramp)
    coarse = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(stock="Delta 3200")), ramp)
    assert coarse > fine


def test_bigger_format_is_smoother(ramp):
    small = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(fmt="35mm")), ramp)
    big = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(fmt="8x10")), ramp)
    assert big < small


def test_grain_is_loudest_in_the_midtones(ramp):
    out = g4.render_v4_cpu(ramp, g4.GrainV4Look())
    per_col = np.abs(out - ramp).mean(axis=(0, 2))
    w = ramp.shape[1]
    assert per_col[w // 2 - 6:w // 2 + 6].mean() > per_col[:w // 8].mean()
    assert per_col[w // 2 - 6:w // 2 + 6].mean() > per_col[-w // 8:].mean()


def test_black_and_white_still_carry_grain(ramp):
    out = g4.render_v4_cpu(ramp, g4.GrainV4Look())
    per_col = np.abs(out - ramp).mean(axis=(0, 2))
    assert per_col[:4].mean() > 1e-4
    assert per_col[-4:].mean() > 1e-4


def test_mono_stays_neutral(ramp):
    out = g4.render_v4_cpu(ramp, g4.GrainV4Look(mono=True))
    assert np.allclose(out[:, :, 0], out[:, :, 1], atol=1e-5)


def test_variation_is_another_roll(ramp):
    a = g4.render_v4_cpu(ramp, g4.GrainV4Look(variation=1))
    b = g4.render_v4_cpu(ramp, g4.GrainV4Look(variation=9))
    assert not np.allclose(a, b)
    assert _dev(a, ramp) == pytest.approx(_dev(b, ramp), rel=0.2)


def test_strength_scales_the_effect(ramp):
    half = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(strength=0.5)), ramp)
    full = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(strength=1.0)), ramp)
    assert half < full


def test_lands_near_v3_for_the_same_film(ramp):
    """Same emulsion, different draw: v4 runs a touch grainier (no union loss
    inside a clump, and continuous shape variety) but must stay in the same
    place, not a different look."""
    for stock in ("Acros 100", "Tri-X 400", "Delta 3200"):
        a = _dev(g3.render_v3_cpu(ramp, g3.GrainV3Look(stock=stock)), ramp)
        b = _dev(g4.render_v4_cpu(ramp, g4.GrainV4Look(stock=stock)), ramp)
        assert b == pytest.approx(a, rel=0.30), stock


def test_shape_and_dtype_preserved(ramp):
    out = g4.render_v4_cpu(ramp, g4.GrainV4Look())
    assert out.shape == ramp.shape and out.dtype == np.float32


def test_greyscale_input_stays_2d():
    grey = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    assert g4.render_v4_cpu(grey, g4.GrainV4Look()).shape == grey.shape


# =============================================================================
# GPU vs the oracle
# =============================================================================

@requires_gpu
@pytest.mark.parametrize("stock", ["Acros 100", "Tri-X 400", "Delta 3200"])
def test_gpu_matches_oracle(ramp, stock):
    from core.gpu import Frame
    look = g4.GrainV4Look(stock=stock)
    got = gpu.grain_v4_frame(Frame.from_cpu(ramp, gpu), look)
    assert got is not None
    got = got.cpu()
    ref = g4.render_v4_cpu(ramp, look)
    assert got.mean() == pytest.approx(ref.mean(), abs=0.005)
    assert _dev(got, ramp) == pytest.approx(_dev(ref, ramp), rel=0.05)


@requires_gpu
def test_gpu_is_deterministic(ramp):
    """Preview must equal export, so a frame has to render identically every
    time — the fixed-point accumulator is what guarantees it."""
    from core.gpu import Frame
    look = g4.GrainV4Look()
    a = gpu.grain_v4_frame(Frame.from_cpu(ramp, gpu), look).cpu()
    b = gpu.grain_v4_frame(Frame.from_cpu(ramp, gpu), look).cpu()
    assert np.array_equal(a, b)


@requires_gpu
def test_v3_is_unaffected_by_v4(ramp):
    """v3 shares the tile-reduction and resolve passes with v4; this catches a
    regression in that shared plumbing."""
    from core.gpu import Frame
    look = g3.GrainV3Look()
    got = gpu.grain_v3_frame(Frame.from_cpu(ramp, gpu), look).cpu()
    ref = g3.render_v3_cpu(ramp, look)
    assert _dev(got, ramp) == pytest.approx(_dev(ref, ramp), rel=0.05)


# =============================================================================
# Wiring into the app
# =============================================================================

def test_model_selection():
    from core.processor import _grain_model, _grain_is_v3, grain_v3_look
    assert _grain_model(VibeConfig()) == 'standard'
    assert _grain_model(VibeConfig(grain_model='v4')) == 'v4'
    assert _grain_is_v3(VibeConfig(grain_model='v4')) is True
    assert isinstance(grain_v3_look(VibeConfig(grain_model='v4')), g4.GrainV4Look)
    assert isinstance(grain_v3_look(VibeConfig(grain_model='v3')), g3.GrainV3Look)


def test_unknown_model_falls_back_to_standard():
    from core.processor import _grain_model
    assert _grain_model(VibeConfig(grain_model='nope')) == 'standard'


def test_v3_and_v4_share_the_film_settings():
    """The five controls describe the film, not the engine — switching model
    must not lose them."""
    from core.processor import grain_v3_look
    v = VibeConfig(grain_model='v3', grain_v3_stock='Delta 3200',
                   grain_v3_format='6x7', grain_v3_strength_pct=80.0,
                   grain_v3_variation=5, grain_v3_mono=True)
    a = grain_v3_look(v)
    v.grain_model = 'v4'
    b = grain_v3_look(v)
    assert (a.stock, a.fmt, a.strength, a.variation, a.mono) == \
           (b.stock, b.fmt, b.strength, b.variation, b.mono)


def test_v4_survives_serialization():
    v = VibeConfig(grain_model='v4', grain_v3_stock='HP5 pushed')
    back = VibeConfig.from_dict(v.to_dict())
    assert back.grain_model == 'v4' and back.grain_v3_stock == 'HP5 pushed'
