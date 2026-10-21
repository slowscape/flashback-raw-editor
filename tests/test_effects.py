"""
Tests for image-effect functions.

These don't check that the result *looks* right — they check shape, dtype,
and the explicit "strength=0 means no-op" contracts that the render
pipeline relies on for short-circuiting.
"""
import numpy as np
import pytest

import math

from core.effects import (
    apply_chromatic_aberration, apply_softness, apply_sharpen,
    apply_vignette, apply_bloom,
    crop_straighten, effective_crop, frame_quad, rect_in_quad,
)


@pytest.fixture
def neutral_gray():
    """64x64 neutral grey float32 image."""
    return np.full((64, 64, 3), 0.5, dtype=np.float32)


# =============================================================================
# No-op contracts
# =============================================================================

def test_vignette_strength_zero_is_noop(neutral_gray):
    out = apply_vignette(neutral_gray, strength=0.0)
    assert np.array_equal(out, neutral_gray)


def test_bloom_strength_zero_is_noop(neutral_gray):
    out = apply_bloom(neutral_gray, strength=0.0)
    assert np.array_equal(out, neutral_gray)


# =============================================================================
# Shape / dtype preservation
# =============================================================================

@pytest.mark.parametrize("fn,args", [
    (apply_chromatic_aberration, (0.005,)),
    (apply_softness, (0.5,)),
    (apply_sharpen, (0.5, 1.0)),
    (apply_vignette, (0.5, 0.05, 1.0)),
    (apply_bloom, (0.3, 0.65)),
])
def test_effect_preserves_shape_and_dtype(neutral_gray, fn, args):
    out = fn(neutral_gray, *args)
    assert out.shape == neutral_gray.shape, f"{fn.__name__} changed shape"
    assert out.dtype == np.float32, f"{fn.__name__} changed dtype"


# =============================================================================
# Output range
# =============================================================================

def test_vignette_never_brightens_pixels(neutral_gray):
    """A vignette darkens edges; no output pixel should exceed input value."""
    out = apply_vignette(neutral_gray, strength=0.5, color_shift=0.0,
                          feather=1.0)
    assert (out <= neutral_gray + 1e-6).all()


def test_vignette_center_unchanged(neutral_gray):
    """The exact image center should be ~untouched by the vignette."""
    out = apply_vignette(neutral_gray, strength=0.5, color_shift=0.05,
                          feather=1.0)
    h, w = neutral_gray.shape[:2]
    center_in = neutral_gray[h // 2, w // 2]
    center_out = out[h // 2, w // 2]
    assert np.allclose(center_out, center_in, atol=1e-3)


# =============================================================================
# Geometry — straighten + crop
# =============================================================================

@pytest.mark.parametrize("w,h", [(4144, 3088), (3088, 4144), (1000, 1000)])
@pytest.mark.parametrize("angle", [0.0, 0.5, 5.0, 15.0, -10.0, 45.0])
@pytest.mark.parametrize("scale,cx,cy", [(1.0, 0.5, 0.5), (0.5, 0.5, 0.5),
                                          (0.6, 0.7, 0.35), (0.3, 0.15, 0.85)])
def test_effective_crop_inside_frame_and_keeps_aspect(w, h, angle, scale, cx, cy):
    x0, y0, x1, y1 = effective_crop(w, h, angle, scale, cx, cy)
    assert rect_in_quad(x0, y0, x1, y1, frame_quad(w, h, angle)), "crop leaves the frame"
    assert (x1 - x0) / (y1 - y0) == pytest.approx(w / h, rel=1e-3), "aspect not preserved"


def test_effective_crop_full_at_zero_is_whole_frame():
    x0, y0, x1, y1 = effective_crop(4144, 3088, 0.0, 1.0, 0.5, 0.5)
    assert (x0, y0, x1, y1) == pytest.approx((0, 0, 4144, 3088))


def test_effective_crop_square_at_45_is_inv_sqrt2():
    """Full crop of a square at 45° shrinks to the largest inscribed square."""
    x0, y0, x1, y1 = effective_crop(1000, 1000, 45.0, 1.0, 0.5, 0.5)
    assert (x1 - x0) == pytest.approx(1000 / math.sqrt(2), rel=1e-3)


def test_straighten_keeps_crop_size_until_edge():
    """BUG #2: a crop that still fits must NOT shrink as the angle changes."""
    widths = [effective_crop(4144, 3088, a, 0.5, 0.5, 0.5)[2] -
              effective_crop(4144, 3088, a, 0.5, 0.5, 0.5)[0]
              for a in (0, 2, 5, 8, 12, 15)]
    assert max(widths) - min(widths) < 1e-6, "0.5 crop should keep its size"
    assert widths[0] == pytest.approx(0.5 * 4144, abs=1.0)


def test_full_crop_shrinks_with_angle():
    """A full-frame crop DOES shrink once its corners hit the rotated edge."""
    w0 = effective_crop(4144, 3088, 0, 1.0, 0.5, 0.5)
    w10 = effective_crop(4144, 3088, 10, 1.0, 0.5, 0.5)
    assert (w10[2] - w10[0]) < (w0[2] - w0[0])


def test_crop_straighten_defaults_are_noop():
    img = np.random.rand(120, 160, 3).astype(np.float32)
    out = crop_straighten(img)  # all defaults
    assert out is img, "neutral geometry should pass the array through untouched"


def test_crop_straighten_no_crop_flag_keeps_full_frame():
    img = np.random.rand(120, 160, 3).astype(np.float32)
    out = crop_straighten(img, angle_deg=8.0, apply_crop=False)
    assert out.shape == img.shape, "apply_crop=False must return the full rotated frame"


def test_crop_straighten_centered_crop_takes_center():
    """A centred half-scale crop (no rotation) returns the middle of the frame."""
    img = np.random.rand(200, 200, 3).astype(np.float32)
    out = crop_straighten(img, scale=0.5)
    assert out.shape[0] == pytest.approx(100, abs=1)
    assert out.shape[1] == pytest.approx(100, abs=1)
    ys, xs = img.shape[0], img.shape[1]
    y0 = (ys - out.shape[0]) // 2
    x0 = (xs - out.shape[1]) // 2
    ref = img[y0:y0 + out.shape[0], x0:x0 + out.shape[1]]
    assert np.array_equal(out, ref)


def test_crop_straighten_offcenter_shifts_slice():
    """Moving crop_cx right selects a region further right than centred."""
    img = np.random.rand(200, 200, 3).astype(np.float32)
    centered = crop_straighten(img, scale=0.5, cx=0.5)
    right = crop_straighten(img, scale=0.5, cx=0.9)
    assert not np.array_equal(centered, right)
    assert right.shape == centered.shape


def test_crop_straighten_preserves_dtype():
    img = np.random.rand(120, 160, 3).astype(np.float32)
    out = crop_straighten(img, angle_deg=6.0, scale=0.8)
    assert out.dtype == np.float32
    assert out.shape[1] / out.shape[0] == pytest.approx(160 / 120, rel=0.02)
