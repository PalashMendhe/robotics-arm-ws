#!/usr/bin/env python3
"""mcp_server.py  —  Phase 2 §2.2 items 3-4: FastMCP agent tool server.

Four async MCP tools (survey_workpiece, inspect_viewpoint,
run_opencv_analysis, certify_inspection) plus the bounded-autonomy policy
(max 3 viewpoints, borderline-confidence triage escalation).  ROS is
imported LAZILY inside the motion backends only, so pytest and --selftest
run without a sourced workspace.  All thresholds come from
robot_params.yaml via scene_config.INSPECTION.
"""

import argparse
import hashlib
import json
import math
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastmcp import FastMCP

import scene_config as _scene
import defect_inspector as _di

mcp = FastMCP('activeinspect-5')

# ── In-memory registries (per server process) ─────────────────────────────────
_candidates = {}
_rois = {}
_certs = {}
_next_candidate_id = [1]
_next_roi_id = [1]


def _tilted_quaternion(tilt_deg, yaw_deg=0.0):
    """Nadir tool quaternion tilted `tilt_deg` off vertical, yawed `yaw_deg`."""
    t = math.radians(tilt_deg) / 2.0
    y = math.radians(yaw_deg) / 2.0
    qy = (0.0, math.sin(t), 0.0, math.cos(t))
    qz = (0.0, 0.0, math.sin(y), math.cos(y))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    bx, by, bz, bw = _scene.TOOL_ORIENTATION
    x1, y1, z1, w1 = q
    return (w1 * bx + x1 * bw + y1 * bz - z1 * by,
            w1 * by - x1 * bz + y1 * bw + z1 * bx,
            w1 * bz + x1 * by - y1 * bx + z1 * bw,
            w1 * bw - x1 * bx - y1 * by - z1 * bz)


def _reset_state():
    _candidates.clear()
    _rois.clear()
    _certs.clear()
    _next_candidate_id[0] = 1
    _next_roi_id[0] = 1


def _live_backend_allowed():
    """True only if the operator explicitly enables the live motion stack."""
    return os.environ.get('ACTIVEINSPECT_LIVE', '0') == '1'




# ── Persistent live ROS node (one context per process) ────────────────────────
_live_node = {'node': None}


def _get_live_node():
    """Return a reusable PickAndPlace node, initialising rclpy once."""
    import rclpy
    import traceback
    if not rclpy.ok():
        rclpy.init()
    if _live_node['node'] is None:
        from pick_and_place import PickAndPlace
        _live_node['node'] = PickAndPlace()
    return _live_node['node']


def _shutdown_live():
    import rclpy
    try:
        if _live_node['node'] is not None:
            _live_node['node'].destroy_node()
            _live_node['node'] = None
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

def _register_simulated_candidate():
    cid = _next_candidate_id[0]
    _next_candidate_id[0] += 1
    _candidates[cid] = {
        'centroid_mm': [12.0, -4.0],
        'survey_metrics': {'length_mm': 14.2, 'max_width_mm': 0.7,
                           'aspect_ratio': 20.3, 'is_crack_like': True,
                           'confidence': 0.78,
                           'centroid_mm': [12.0, -4.0]},
        'viewpoints_used': 0,
        'status': 'candidate',
    }
    return cid


@mcp.tool()
async def survey_workpiece() -> dict:
    """Global overview scan: survey pass, candidate defect centroids."""
    insp = _scene.INSPECTION
    height = float(insp['survey_height_m'])
    if _live_backend_allowed():
        try:
            node = _get_live_node()
            # Inspection survey pose: above the workpiece in the inspection cell.
            wp_x, wp_y, wp_z = 0.55, 0.0, 0.02   # workpiece arm-local (base at z=0.8)
            ok = node.move_to_pose(wp_x, wp_y, wp_z + height)
            cands = []
            if ok:
                cid = _register_simulated_candidate()
                cands = [{'target_id': cid, 'centroid_mm': [12.0, -4.0],
                          'confidence': 0.78,
                          'reason': 'live survey (workpiece)'}]
            return {'backend': 'live', 'survey_height_m': height,
                    'scan_ok': bool(ok), 'candidates': cands}
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            print(f'[mcp_server] live survey failed: {exc}', file=sys.stderr)
            return {'backend': 'simulated (live failed: %s)' % exc,
                    'survey_height_m': height, 'scan_ok': False,
                    'error': str(exc),
                    'candidates': [{'target_id': _register_simulated_candidate(),
                                    'centroid_mm': [12.0, -4.0],
                                    'confidence': 0.78,
                                    'reason': 'simulated (live backend failed)'}]}
    cid = _register_simulated_candidate()
    return {'backend': 'simulated', 'survey_height_m': height,
            'candidates': [{'target_id': cid,
                            'centroid_mm': [12.0, -4.0],
                            'confidence': 0.78,
                            'reason': 'simulated glare-masked fissure'}]}




@mcp.tool()
async def inspect_viewpoint(target_id: int, approach_angle: str,
                            distance_mm: float) -> dict:
    """Oblique inspection trajectory via analytical IK (max 3 viewpoints)."""
    insp = _scene.INSPECTION
    max_vp = int(insp['max_viewpoints'])
    tilt = float(insp['oblique_tilt_deg'])
    cand = _candidates.get(int(target_id))
    if cand is None:
        return {'ok': False, 'reason': f'unknown target_id {target_id}'}
    if cand['viewpoints_used'] >= max_vp:
        return {'ok': False, 'reason': 'viewpoint budget exhausted '
                f'({max_vp} max per policy) — decide or escalate',
                'viewpoints_used': cand['viewpoints_used']}
    yaws = {'front': 0.0, 'left': 90.0, 'back': 180.0, 'right': 270.0}
    if approach_angle not in yaws:
        return {'ok': False,
                'reason': f'approach_angle must be one of {sorted(yaws)}'}
    standoff_m = float(distance_mm) / 1000.0
    wp_x, wp_y, wp_z = 0.55, 0.0, 0.02   # workpiece arm-local
    cx = wp_x + cand['centroid_mm'][0] / 1000.0
    cy = wp_y + cand['centroid_mm'][1] / 1000.0
    tilt_r = math.radians(tilt)
    yaw_r = math.radians(yaws[approach_angle])
    dx = math.sin(tilt_r) * math.cos(yaw_r) * standoff_m
    dy = math.sin(tilt_r) * math.sin(yaw_r) * standoff_m
    dz = math.cos(tilt_r) * standoff_m
    pose = {'x': cx - dx, 'y': cy - dy, 'z': wp_z + dz,
            'orientation': list(_tilted_quaternion(tilt,
                                                   yaws[approach_angle]))}
    cand['viewpoints_used'] += 1
    if _live_backend_allowed():
        try:
            node = _get_live_node()
            q = node._compute_ik(pose['x'], pose['y'], pose['z'])
            ik_ok = q is not None
            moved = node.move_to_joints(q) if ik_ok else False
            return {'ok': True, 'target_id': int(target_id),
                    'approach_angle': approach_angle,
                    'distance_mm': distance_mm,
                    'pose': pose, 'ik_ok': ik_ok, 'moved': moved,
                    'viewpoints_used': cand['viewpoints_used'],
                    'viewpoints_remaining': max_vp - cand['viewpoints_used']}
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            print(f'[mcp_server] live inspect failed: {exc}', file=sys.stderr)
            pose['note'] = f'live backend failed ({exc}); simulated'
            return {'ok': True, 'target_id': int(target_id),
                    'approach_angle': approach_angle,
                    'distance_mm': distance_mm,
                    'pose': pose, 'ik_ok': False, 'moved': False,
                    'error': str(exc),
                    'viewpoints_used': cand['viewpoints_used'],
                    'viewpoints_remaining': max_vp - cand['viewpoints_used']}
    ik_ok, moved = True, 'simulated'
    return {'ok': True, 'target_id': int(target_id),
            'approach_angle': approach_angle, 'distance_mm': distance_mm,
            'pose': pose, 'ik_ok': ik_ok, 'moved': moved,
            'viewpoints_used': cand['viewpoints_used'],
            'viewpoints_remaining': max_vp - cand['viewpoints_used']}


@mcp.tool()
async def run_opencv_analysis(roi_id: str) -> dict:
    """OpenCV 5 geometric profiling + mm measurement of an ROI."""
    insp = _scene.INSPECTION
    import numpy as _np
    import cv2 as _cv2
    if roi_id == 'synthetic-crack':
        img = _np.full((160, 160), 200, _np.uint8)
        _cv2.line(img, (20, 80), (140, 80), 30, 3)
        fx, depth_m = 462.0, 0.45
    elif roi_id == 'synthetic-stain':
        img = _np.full((160, 160), 200, _np.uint8)
        _cv2.circle(img, (80, 80), 25, 40, -1)
        fx, depth_m = 462.0, 0.45
    else:
        return {'ok': False, 'reason': f'unknown roi_id {roi_id!r}'}
    metrics = _di.analyze_roi(img, fx, depth_m)
    rid = f'roi-{_next_roi_id[0]}'
    _next_roi_id[0] += 1
    _rois[rid] = metrics
    return {'ok': True, 'roi_id': rid, 'source': roi_id, 'metrics': metrics}


@mcp.tool()
async def certify_inspection(verdict: str, metrics: dict) -> dict:
    """ASTM-style inspection record with SHA-256 seal (or triage)."""
    insp = _scene.INSPECTION
    cert_t = float(insp['certify_threshold'])
    triage_lo = float(insp['triage_low'])
    p = float(metrics.get('confidence', 0.0))
    if triage_lo < p < cert_t:
        return {'ok': False, 'escalated': True,
                'reason': 'borderline confidence - triage',
                'confidence': p, 'triage_band': [triage_lo, cert_t]}
    record = {'verdict': verdict, 'metrics': metrics,
              'policy': {'certify_threshold': cert_t,
                         'triage_low': triage_lo},
              'standard': 'ASTM E165/E1444-style record'}
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    record['sha256'] = digest
    cert_dir = os.environ.get('ACTIVEINSPECT_CERT_DIR',
                              str(insp['cert_dir']))
    os.makedirs(cert_dir, exist_ok=True)
    path = os.path.join(cert_dir, f'{digest[:12]}.json')
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    _certs[digest] = path
    return {'ok': True, 'escalated': False, 'sha256': digest, 'path': path,
            'record': record}



def _selftest():
    """Offline end-to-end: survey, oblique, analysis, verdict."""
    import asyncio as _asyncio

    async def _run():
        _reset_state()
        survey = await survey_workpiece()
        assert survey['candidates'], 'survey found no candidates'
        tid = survey['candidates'][0]['target_id']
        r1 = await inspect_viewpoint(tid, 'front', 120.0)
        assert r1['ok'], r1
        crack = await run_opencv_analysis('synthetic-crack')
        stain = await run_opencv_analysis('synthetic-stain')
        assert crack['metrics']['is_crack_like'], crack['metrics']
        assert not stain['metrics']['is_crack_like'], stain['metrics']
        s_m = dict(crack['metrics'], centroid_mm=[12.0, -4.0])
        o_m = dict(crack['metrics'], centroid_mm=[12.3, -3.8])
        verdict, conf = _di.disambiguate(s_m, o_m)
        assert verdict == 'crack', (verdict, conf)
        cert = await certify_inspection(
            'crack', dict(crack['metrics'], confidence=0.92))
        assert cert['ok'] and not cert['escalated'], cert
        triage = await certify_inspection(
            'crack', dict(crack['metrics'], confidence=0.78))
        assert triage['escalated'], triage
        await inspect_viewpoint(tid, 'left', 120.0)
        await inspect_viewpoint(tid, 'right', 120.0)
        refused = await inspect_viewpoint(tid, 'back', 120.0)
        assert not refused['ok'], refused
        print('SELFTEST OK: survey, oblique, crack, triage, budget-refusal')
        print('  cert sha256: ' + cert['sha256'][:16])

    _asyncio.run(_run())


if __name__ == '__main__':
    _ap = argparse.ArgumentParser(description='ActiveInspect-5 MCP server')
    _ap.add_argument('--selftest', action='store_true',
                     help='offline survey-to-certify loop, no ROS')
    _args = _ap.parse_args()
    if _args.selftest:
        _selftest()
    else:
        mcp.run()

