#!/usr/bin/env python3
"""defect_inspector.py  —  Phase 2 §2.2 items 1-2: defect vision core.

Pure-OpenCV geometric profiler + multi-view disambiguation engine.  Importing
this module pulls in NOTHING ROS-specific (only cv2 + numpy), so the unit
tests exercise it without a sourced workspace.

Pipeline (per ROI, grayscale):
  bilateral filter → adaptive-Gaussian threshold → open → close →
  connectedComponents → largest component → skeletonize (Zhang–Suen, vendored:
  opencv-contrib's ximgproc.thinning ships no wheels for OpenCV 5 aarch64,
  and the Gazebo/CI images must stay contrib-free) → spine walk (length,
  max width via distanceTransform, aspect ratio) → K^-1 projection to mm.

The DNN hook (§2.2 item 1, ONNX via cv2.dnn) is a deliberate seam, not a
dependency: attach_onnx_model(path) registers a graph, and every analyze_roi
verdict carries model_confidence=None until a model is attached — the
geometric path never requires one.
"""

import math
import os

import cv2
import numpy as np


# ── Tunable defaults (overridden per-call from scene_config.INSPECTION) ──────
ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = 7.0
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 50.0
BILATERAL_SIGMA_SPACE = 7.0
CRACK_ASPECT_MIN = 4.0
MIN_SPINE_LENGTH_MM = 1.0
ANCHOR_RADIUS_MM = 2.0

# Attached ONNX graph (None until attach_onnx_model is called).
_onnx_net = None
_onnx_path = None


def attach_onnx_model(path):
    """Attach an ONNX defect model via the cv2.dnn graph engine (§2.2 item 1).

    Returns the loaded net; raises FileNotFoundError for a missing path so a
    bad model path fails loudly instead of silently degrading to geometry.
    """
    global _onnx_net, _onnx_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f'ONNX defect model not found: {path}')
    _onnx_net = cv2.dnn.readNetFromONNX(path)
    _onnx_path = path
    return _onnx_net


def model_confidence(roi_gray):
    """DNN confidence for an ROI, or None when no model is attached."""
    if _onnx_net is None:
        return None
    blob = cv2.dnn.blobFromImage(roi_gray, 1.0 / 255.0, (224, 224),
                                 (0, 0, 0), swapRB=False, crop=False)
    _onnx_net.setInput(blob)
    out = _onnx_net.forward().ravel()
    return float(np.max(out))


def skeletonize(mask):
    """Zhang–Suen thinning of a binary (0/255) mask → 1-px spine (0/255).

    Vendored (two-pass, vectorised with numpy) because cv2.ximgproc.thinning
    lives in opencv-contrib, which ships no OpenCV-5 wheels for aarch64 —
    Phase 3's Graviton image cannot depend on it.
    """
    img = (mask > 0).astype(np.uint8)
    changed = True
    rows, cols = img.shape
    while changed:
        changed = False
        for step in (0, 1):
            # 8-neighbours p2..p9 clockwise from north.
            p2 = img[0:rows - 2, 1:cols - 1]
            p3 = img[0:rows - 2, 2:cols]
            p4 = img[1:rows - 1, 2:cols]
            p5 = img[2:rows, 2:cols]
            p6 = img[2:rows, 1:cols - 1]
            p7 = img[2:rows, 0:cols - 2]
            p8 = img[1:rows - 1, 0:cols - 2]
            p9 = img[0:rows - 2, 0:cols - 2]
            nbrs = np.stack([p2, p3, p4, p5, p6, p7, p8, p9], axis=0)
            n_count = nbrs.sum(axis=0)
            # 0→1 transitions in the ordered ring p2..p9,p2.
            ring = np.concatenate([nbrs, nbrs[0:1]], axis=0)
            transitions = ((ring[1:] - ring[:-1]) == 1).sum(axis=0)
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            kill = (img[1:rows - 1, 1:cols - 1] == 1) & (n_count >= 2) \
                & (n_count <= 6) & (transitions == 1) & cond
            if kill.any():
                img[1:rows - 1, 1:cols - 1][kill] = 0
                changed = True
    out = np.zeros_like(mask)
    out[img > 0] = 255
    return out


def spine_metrics(skeleton, dist, mm_per_pixel):
    """Walk the 1-px spine → (length_mm, max_width_mm, aspect_ratio).

    Length = 8-connected walk (orthogonal 1, diagonal √2).  Width at each
    spine pixel = 2·distanceTransform value.  Aspect = length / width.
    """
    ys, xs = np.nonzero(skeleton > 0)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    if len(xs) == 1:
        # Degenerate spine (e.g. a filled disc thins to one pixel): no walk
        # is possible; report zero length with the distance-peak width so the
        # aspect ratio → 0 (blob, never crack-like) instead of a bogus walk.
        w = float(dist[ys[0], xs[0]]) * 2.0 * mm_per_pixel
        return 0.0, w, 0.0
    pts = set(zip(xs.tolist(), ys.tolist()))
    start = None
    for (x, y) in pts:
        n = sum((x + dx, y + dy) in pts
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0))
        if n == 1:
            start = (x, y)
            break
    if start is None:  # closed loop: no endpoint, start anywhere
        start = next(iter(pts))
    visited, cur, prev, steps = {start}, start, None, 0.0
    max_w_px = float(dist[start[1], start[0]]) * 2.0
    while True:
        x, y = cur
        # At junctions (Y-shapes from the open/close rounding of a filled
        # blob) several unvisited neighbours exist: follow the branch whose
        # distanceTransform ridge runs deepest, i.e. the widest continuation.
        # A true crack spine has no junctions (single neighbour), so this
        # choice only affects blob artefacts — and there it yields the full
        # disc diameter as the width instead of a truncated walk.
        best, best_d = None, -1.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                cand = (x + dx, y + dy)
                if cand in pts and cand not in visited \
                        and (prev is None or cand != prev):
                    d = float(dist[cand[1], cand[0]])
                    if d > best_d:
                        best, best_d = cand, d
        nxt = best
        if nxt is None:
            break
        steps += math.sqrt(2.0) if (nxt[0] != x and nxt[1] != y) else 1.0
        max_w_px = max(max_w_px, float(dist[nxt[1], nxt[0]]) * 2.0)
        visited.add(nxt)
        prev, cur = cur, nxt
    length_mm = steps * mm_per_pixel
    width_mm = max_w_px * mm_per_pixel
    return length_mm, width_mm, length_mm / max(width_mm, 1e-9)


def analyze_roi(roi_gray, fx, depth_m, block_size=ADAPTIVE_BLOCK_SIZE,
                c=ADAPTIVE_C, bilateral_d=BILATERAL_D,
                bilateral_sig_color=BILATERAL_SIGMA_COLOR,
                bilateral_sig_space=BILATERAL_SIGMA_SPACE,
                crack_aspect_min=CRACK_ASPECT_MIN,
                min_spine_length_mm=MIN_SPINE_LENGTH_MM):
    """Geometric profile of one grayscale ROI → metrics dict (§2.2 item 1).

    K^-1 projection: mm_per_pixel = depth_m / fx · 1000.  confidence fuses
    aspect evidence (0.7) with spine-length evidence (0.3); a sub-threshold
    spine (dust/speckle) scores 0.0 and is never crack-like.
    """
    if roi_gray is None or roi_gray.size == 0:
        raise ValueError('analyze_roi received an empty ROI')
    gray = roi_gray if roi_gray.ndim == 2 else \
        cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    filt = cv2.bilateralFilter(gray, bilateral_d,
                               bilateral_sig_color, bilateral_sig_space)
    th = cv2.adaptiveThreshold(filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block_size, c)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    mm_per_pixel = float(depth_m) / float(fx) * 1000.0
    empty = {'length_mm': 0.0, 'max_width_mm': 0.0, 'aspect_ratio': 0.0,
             'is_crack_like': False, 'confidence': 0.0,
             'model_confidence': model_confidence(gray),
             'mm_per_pixel': mm_per_pixel, 'area_px': 0}
    if n <= 1:
        return empty
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1  # largest component = candidate defect
    area_px = int(stats[idx, cv2.CC_STAT_AREA])
    comp = np.zeros_like(closed)
    comp[labels == idx] = 255
    skel = skeletonize(comp)
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    length_mm, width_mm, aspect = spine_metrics(skel, dist, mm_per_pixel)
    if length_mm < min_spine_length_mm:
        empty['area_px'] = area_px
        return empty
    aspect_ev = min(1.0, aspect / (2.0 * crack_aspect_min))
    length_ev = min(1.0, length_mm / (5.0 * min_spine_length_mm))
    confidence = round(0.7 * aspect_ev + 0.3 * length_ev, 3)
    return {'length_mm': round(length_mm, 3),
            'max_width_mm': round(width_mm, 3),
            'aspect_ratio': round(aspect, 2),
            'is_crack_like': bool(aspect >= crack_aspect_min),
            'confidence': confidence,
            'model_confidence': model_confidence(gray),
            'mm_per_pixel': round(mm_per_pixel, 4), 'area_px': area_px}


def disambiguate(survey_metrics, oblique_metrics,
                 anchor_radius_mm=ANCHOR_RADIUS_MM):
    """Multi-view verdict (§2.2 item 2): anchored crack vs shifting glare.

    Centroids are (x, y) mm in CAD-datum coordinates.  Anchored (shift ≤
    radius) + oblique crack-like → crack; shifting/vanishing → benign;
    anchored but oblique-clean → inconclusive (one more viewpoint, bounded
    by max_viewpoints in the MCP policy layer).
    """
    s = survey_metrics.get('centroid_mm', (0.0, 0.0))
    o = oblique_metrics.get('centroid_mm', (0.0, 0.0))
    shift = math.dist((float(s[0]), float(s[1])), (float(o[0]), float(o[1])))
    anchored = shift <= anchor_radius_mm
    oblique_crack = bool(oblique_metrics.get('is_crack_like', False))
    survey_crack = bool(survey_metrics.get('is_crack_like', False))
    p_s = float(survey_metrics.get('confidence', 0.0))
    p_o = float(oblique_metrics.get('confidence', 0.0))
    if anchored and oblique_crack:
        return 'crack', round(0.5 * (p_s + p_o), 3)
    if not anchored:
        return 'benign', round(0.5 * ((1.0 - p_s) + (1.0 - p_o)), 3)
    if survey_crack and not oblique_crack:
        return 'inconclusive', round(0.5 * (p_s + (1.0 - p_o)), 3)
    return 'benign', round(0.5 * ((1.0 - p_s) + (1.0 - p_o)), 3)
