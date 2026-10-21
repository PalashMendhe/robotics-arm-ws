# src/robotic_4dof_arm/test/test_shape_detector.py
"""Unit tests for ShapeDetector pure-Python logic (no ROS graph, no camera).

Covers §7 detector test plan:
  - 2 red + 2 blue synthetic blobs → exactly 4 detections, correct classes
  - Border-touching component → rejected (§3.2 item 4)
  - Touching blobs of the same class → still 2 detections (morph does not bridge)
  - Adaptive area gate (strict) rejects under-/over-sized components
  - Full-mask depth sampling: valid fraction gate + 3×3 fallback
"""
import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# ── Minimal stub so ShapeDetector.__init__ does not need ROS ────────────────
# We only test the pure-Python helper methods; the class is instantiated
# without __init__ via __new__.
from shape_detector import ShapeDetector  # noqa: E402


# ── Build a minimal ShapeDetector with dummy params ──────────────────────────

def _make_detector(img_w=640, img_h=480) -> ShapeDetector:
    """Return a ShapeDetector bypassing ROS __init__."""
    import cv2
    from unittest.mock import MagicMock
    det = ShapeDetector.__new__(ShapeDetector)

    # Mock the ROS logger so helper methods that call self.get_logger() work
    # without an initialised Node.
    det.get_logger = MagicMock(return_value=MagicMock())

    # Intrinsics (enough for area_gate / expected_px)
    det.fx = det.fy = 462.0
    det.cx = img_w / 2.0
    det.cy = img_h / 2.0
    det._camera_info_ready = True

    # Colour thresholds (mirror camera_params.yaml)
    det.red_lo1  = np.array([0,   120,  70], dtype=np.uint8)
    det.red_hi1  = np.array([10,  255, 255], dtype=np.uint8)
    det.red_lo2  = np.array([170, 120,  70], dtype=np.uint8)
    det.red_hi2  = np.array([180, 255, 255], dtype=np.uint8)
    det.blue_lo  = np.array([100, 120,  70], dtype=np.uint8)
    det.blue_hi  = np.array([130, 255, 255], dtype=np.uint8)

    # Morphology kernels
    det.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    det.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Adaptive area gate
    det.object_size_m   = 0.10
    det.nominal_range_m = 0.35
    det.area_scale_min  = 0.4
    det.area_scale_max  = 4.0
    det.min_area_px     = 60.0

    # Border margin
    det.border_margin_px     = 2
    det.depth_valid_frac_min = 0.3

    # Depth range
    det.depth_min   = 0.30
    det.depth_max   = 2.00
    det.depth_scale = 1.0

    return det


def _red_bgr()  -> tuple: return (30, 30, 230)    # red-ish in BGR
def _blue_bgr() -> tuple: return (200, 30,  30)   # blue-ish in BGR


def _paint_blob(img_bgr: np.ndarray, cx: int, cy: int, r: int, colour: tuple):
    """Draw a filled circle (blob) on img_bgr."""
    import cv2
    cv2.circle(img_bgr, (cx, cy), r, colour, -1)


# ── Helper: run segment + components and return per-class results ─────────────

def _detect_all(det: ShapeDetector, img_bgr: np.ndarray) -> dict:
    """
    Returns {'red_cube': [...comps...], 'blue_cylinder': [...comps...]}.
    Mirrors the image_callback pipeline without depth / TF / ROS.
    """
    masks = det.segment_by_colour(img_bgr)
    nom_exp = det.expected_px(det.nominal_range_m)
    results = {}
    for cls, mask in masks.items():
        results[cls] = det.components(mask, cls, nom_exp)
    return results


# ── Synthetic image fixtures ─────────────────────────────────────────────────

IMG_W, IMG_H = 640, 480


def _make_2red_2blue_image():
    """Four well-separated blobs: 2 red at left, 2 blue at right."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    # Red blobs (radius ≈ 40 px → area ≈ 5026 px²)
    _paint_blob(img, 160, 160, 40, _red_bgr())
    _paint_blob(img, 160, 320, 40, _red_bgr())
    # Blue blobs
    _paint_blob(img, 480, 160, 40, _blue_bgr())
    _paint_blob(img, 480, 320, 40, _blue_bgr())
    return img


def _make_border_touching_image():
    """One red blob inside the frame, one touching the left edge → rejected."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    _paint_blob(img, 320, 240, 40, _red_bgr())        # safe
    _paint_blob(img,   1, 240, 40, _red_bgr())        # touches left border
    return img


def _make_touching_pair_image():
    """
    Two red blobs 10 px apart — close but NOT overlapping: morph should NOT
    bridge them (kernel_close=5 px < 10 px gap), so we get 2 components.
    """
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    _paint_blob(img, 250, 240, 40, _red_bgr())
    _paint_blob(img, 350, 240, 40, _red_bgr())   # 100 px centre-to-centre; gap=20px
    return img


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_segment_by_colour_returns_both_classes():
    det = _make_detector()
    img = _make_2red_2blue_image()
    masks = det.segment_by_colour(img)
    assert 'red_cube' in masks
    assert 'blue_cylinder' in masks
    assert masks['red_cube'].dtype  == np.uint8
    assert masks['blue_cylinder'].dtype == np.uint8


def test_two_red_two_blue_detected():
    """2 red + 2 blue well-separated blobs must yield exactly 2 components each."""
    det = _make_detector()
    img = _make_2red_2blue_image()
    res = _detect_all(det, img)
    assert len(res['red_cube'])    == 2, f"Expected 2 red, got {len(res['red_cube'])}"
    assert len(res['blue_cylinder']) == 2, f"Expected 2 blue, got {len(res['blue_cylinder'])}"


def test_colour_decides_class():
    """Red blobs must never appear as blue_cylinder and vice versa."""
    det = _make_detector()
    img = _make_2red_2blue_image()
    res = _detect_all(det, img)
    # No red pixels in the blue mask and vice versa — components must be in
    # their own class bucket
    assert len(res['red_cube'])      > 0
    assert len(res['blue_cylinder']) > 0
    # Centroid x: red at x≈160, blue at x≈480
    red_xs  = [c['cx_px'] for c in res['red_cube']]
    blue_xs = [c['cx_px'] for c in res['blue_cylinder']]
    assert all(x < 320 for x in red_xs),  f"Red blob centroid not on left: {red_xs}"
    assert all(x > 320 for x in blue_xs), f"Blue blob centroid not on right: {blue_xs}"


def test_border_touching_rejected():
    """A blob that touches the frame border must be silently rejected."""
    det = _make_detector()
    img = _make_border_touching_image()
    res = _detect_all(det, img)
    # The blob at x=1 should be rejected; only the centred one kept
    assert len(res['red_cube']) == 1, (
        f"Expected 1 (border blob rejected), got {len(res['red_cube'])}"
    )


def test_touching_pair_yields_two_components():
    """Two blobs that are close but not overlapping must remain two components."""
    det = _make_detector()
    img = _make_touching_pair_image()
    res = _detect_all(det, img)
    assert len(res['red_cube']) == 2, (
        f"Expected 2 (touching pair not merged), got {len(res['red_cube'])}"
    )


def test_component_dict_has_required_keys():
    """Every component dict must carry the keys used downstream."""
    det = _make_detector()
    img = _make_2red_2blue_image()
    res = _detect_all(det, img)
    for cls in ('red_cube', 'blue_cylinder'):
        for comp in res[cls]:
            for key in ('area', 'cx_px', 'cy_px', 'w', 'h', 'angle', 'circularity', 'mask'):
                assert key in comp, f"Missing key '{key}' in {cls} component"


def test_component_angle_is_float():
    """minAreaRect angle must be a float (used for yaw alignment)."""
    det = _make_detector()
    img = _make_2red_2blue_image()
    res = _detect_all(det, img)
    for comp in res['red_cube']:
        assert isinstance(comp['angle'], float)


# ── Adaptive area gate ────────────────────────────────────────────────────────

def test_area_gate_strict_accepts_correct_size():
    """A blob whose area matches the expected object size must pass the strict gate."""
    det = _make_detector()
    exp = det.expected_px(0.35)       # ~132 px at 0.35 m (fx=462, obj=0.10 m)
    lo, hi = det.area_gate(exp, loose=False)
    # Choose radius so area is exactly exp² (the geometric centre of the gate)
    # exp ≈ 132 → area_mid = exp² ≈ 17424, lo = 0.4*exp² ≈ 6970, hi = 4*exp² ≈ 69696
    area_mid = exp ** 2   # representative in-gate area
    assert lo <= area_mid <= hi, (
        f"mid-gate area {area_mid:.0f} outside strict gate [{lo:.0f}, {hi:.0f}]"
    )


def test_area_gate_loose_is_wider():
    det = _make_detector()
    exp = det.expected_px(0.35)
    lo_strict, hi_strict = det.area_gate(exp, loose=False)
    lo_loose,  hi_loose  = det.area_gate(exp, loose=True)
    assert lo_loose  < lo_strict
    assert hi_loose  > hi_strict


# ── Depth sampling ────────────────────────────────────────────────────────────

def test_get_valid_depth_mask_mode():
    """Full-mask depth sampling must return the median of in-mask pixels."""
    det = _make_detector()
    depth = np.full((480, 640), 0.5, dtype=np.float32)  # uniform 0.5 m
    mask  = np.zeros((480, 640), dtype=np.uint8)
    mask[200:260, 200:260] = 255    # 60×60 block
    result = det.get_valid_depth(depth, 230, 230, mask=mask)
    assert result == pytest.approx(0.5, abs=1e-4)


def test_get_valid_depth_fallback_patch():
    """Without a mask the 3×3 patch centroid fallback must work."""
    det = _make_detector()
    depth = np.full((480, 640), 0.75, dtype=np.float32)
    result = det.get_valid_depth(depth, 320, 240)
    assert result == pytest.approx(0.75, abs=1e-4)


def test_get_valid_depth_rejects_low_frac():
    """If < 30% of mask pixels have valid depth the result must be None."""
    det = _make_detector()
    depth = np.zeros((480, 640), dtype=np.float32)   # all zeros = invalid
    mask  = np.ones((480, 640), dtype=np.uint8) * 255
    # Place only 10% valid pixels
    depth[200:212, 200:640] = 0.5  # 12 rows valid out of ~120 total
    result = det.get_valid_depth(depth, 320, 240, mask=mask)
    # valid fraction will be well below 0.3 over the whole mask
    assert result is None


def test_get_valid_depth_out_of_range_returns_none():
    det = _make_detector()
    depth = np.full((480, 640), 5.0, dtype=np.float32)   # 5 m > depth_max 2.0 m
    mask  = np.ones((480, 640), dtype=np.uint8) * 255
    result = det.get_valid_depth(depth, 320, 240, mask=mask)
    assert result is None


def test_get_valid_depth_all_zero_returns_none():
    det = _make_detector()
    depth = np.zeros((480, 640), dtype=np.float32)
    result = det.get_valid_depth(depth, 320, 240)
    assert result is None
