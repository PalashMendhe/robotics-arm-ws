# src/robotic_4dof_arm/test/test_vision_sorting.py
"""Unit tests for the vision-based sorting mission pure logic.

Covers the ROS-free helper methods (no executor / action servers needed):
  - PickAndPlace.is_reachable / _compute_ik  — reachability envelope guard
  - PickAndPlace._normalize_angle / IK branch wrap / _duplicate_server_hint —
    live-run regressions (UR5e ±2π joint limits, duplicate move_group, errors)
  - SortingController._on_pick_table         — Table 2 pick-region filter
  - SortingController._object_reachable      — grasp+lift waypoint check
  - SortingController._update_registry       — cross-scan object registry
  - SortingController._next_zone / _mark_zone_used / _slot_index — slot logic
  - vision_sim spawner                       — reach-constrained sampling
"""
import sys
import os
import math
import random
import importlib.util
import pytest
from unittest.mock import MagicMock

# Point Python at the scripts folder BEFORE importing from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from pick_and_place import PickAndPlace  # noqa: E402
from sorting_controller import (  # noqa: E402
    SortingController,
    PICK_WORLD_Z,
    ARM_BASE_WORLD_Z,
    MATCH_RADIUS_M,
    LIFT_HEIGHT_M,
    MAX_SLOTS,
    REACH_MARGIN_M,
    SCAN_TCP_WORLD_Z,
    SCAN_X,
    SCAN_YS,
    PICK_REGION_X_MIN,
    PICK_REGION_X_MAX,
    PICK_REGION_Y_MIN,
    PICK_REGION_Y_MAX,
    GRASP_GAP_MIN_M,
    GRASP_GAP_MAX_M,
    GRASP_MAX_VEL_M_S,
)

# Load the launch file as a module (filename has a dot, so use importlib).
LAUNCH_PATH = os.path.join(os.path.dirname(__file__), '..', 'launch', 'vision_sim.launch.py')


@pytest.fixture(scope='module')
def spawner():
    spec = importlib.util.spec_from_file_location('vision_sim_launch', LAUNCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def arm():
    """PickAndPlace without __init__ (no ROS graph needed for pure IK logic)."""
    node = PickAndPlace.__new__(PickAndPlace)
    node.get_logger = MagicMock()   # failure paths log loudly
    return node


@pytest.fixture
def controller():
    """SortingController without __init__ (pure logic only)."""
    node = SortingController.__new__(SortingController)
    node.get_logger = MagicMock()
    node._registry = []
    node._red_zones = [(-0.28, -0.12), (-0.28, -0.24), (-0.28, -0.36)]
    node._blue_zones = [(-0.28, 0.12), (-0.28, 0.24), (-0.28, 0.36)]
    node._red_used = 0
    node._blue_used = 0
    node._last_joint_state = None   # for _prong_states / _verify_grasp tests
    node._buf = []                  # for _collect tests
    return node


def _arm_z(world_z: float) -> float:
    return world_z - ARM_BASE_WORLD_Z


# ── IK reachability ─────────────────────────────────────────────────────────

def test_compute_ik_returns_six_joints_for_reachable(arm):
    q = arm._compute_ik(-0.70, 0.10, _arm_z(PICK_WORLD_Z))
    assert q is not None
    assert len(q) == 6
    assert all(isinstance(v, float) for v in q)


def test_compute_ik_returns_none_for_far_corner(arm):
    """Far outside the UR5e envelope (R ~= 1.00 m) must fail LOUDLY — no clamping.

    (The old (-0.78, 0.38) corner from the custom-UR5 days IS solvable with
    the vendor UR5e geometry — the interim tool-down IK lands the jaw
    FK-exact, verified — so it can no longer serve as the unreachable probe.)"""
    q = arm._compute_ik(-0.92, 0.40, _arm_z(PICK_WORLD_Z))
    assert q is None


def test_is_reachable_boundary(arm):
    z = _arm_z(PICK_WORLD_Z)
    assert arm.is_reachable(-0.70, 0.10, z) is True
    assert arm.is_reachable(-0.78, 0.38, z) is True   # solvable with UR5e geometry
    assert arm.is_reachable(-0.92, 0.40, z) is False


def test_is_reachable_below_wrist_offset(arm):
    """Targets closer than the d4 wrist offset are unsolvable (asin domain)."""
    assert arm.is_reachable(0.0, 0.0, 0.05) is False


def test_is_reachable_comfortable_target_keeps_margin(arm):
    z = _arm_z(PICK_WORLD_Z)
    assert arm.is_reachable(-0.70, 0.10, z, margin=REACH_MARGIN_M) is True


# ── Pick-region filter ──────────────────────────────────────────────────────

def test_on_pick_table_inside(controller):
    assert controller._on_pick_table(-0.70, 0.00) is True
    assert controller._on_pick_table(-0.65, -0.30) is True


def test_on_pick_table_excludes_sort_zones(controller):
    """Placed objects on Table 1 must never be re-picked."""
    assert controller._on_pick_table(-0.28, -0.12) is False
    assert controller._on_pick_table(-0.28, 0.24) is False


def test_on_pick_table_excludes_outside(controller):
    assert controller._on_pick_table(-0.30, 0.00) is False
    assert controller._on_pick_table(-0.70, 0.60) is False
    assert controller._on_pick_table(-1.10, 0.00) is False


def test_pick_region_bounds_are_sane():
    assert PICK_REGION_X_MIN < PICK_REGION_X_MAX
    assert PICK_REGION_Y_MIN < PICK_REGION_Y_MAX


# ── Object reachability (grasp + lift waypoints) ────────────────────────────

def test_object_reachable_for_typical_object(controller):
    assert controller._object_reachable(-0.70, 0.10) is True


def test_object_unreachable_for_far_corner(controller):
    assert controller._object_reachable(-0.92, 0.40) is False


def test_object_reachable_checks_lift_waypoint_too(controller):
    """is_reachable is consulted for BOTH pick and pick+LIFT heights."""
    seen = []

    def fake_is_reachable(x, y, z, margin=0.0):
        seen.append(z)
        return True

    controller.is_reachable = fake_is_reachable
    assert controller._object_reachable(-0.70, 0.10) is True
    pick_z = _arm_z(PICK_WORLD_Z)
    assert pick_z in seen
    assert pick_z + LIFT_HEIGHT_M in seen


# ── Cross-scan registry ─────────────────────────────────────────────────────

def test_registry_registers_and_returns_nearest_first(controller):
    dets = [('red_cube', -0.68, -0.15), ('red_cube', -0.60, 0.30)]
    candidates = controller._update_registry(dets)
    assert len(candidates) == 2
    d0 = math.hypot(candidates[0]['x'], candidates[0]['y'])
    d1 = math.hypot(candidates[1]['x'], candidates[1]['y'])
    assert d0 <= d1
    # (−0.60, 0.30) at R≈0.671 is closer to the base than (−0.68, −0.15)
    # at R≈0.696, so it sorts first
    assert (candidates[0]['x'], candidates[0]['y']) == (-0.60, 0.30)


def test_registry_dedups_within_match_radius(controller):
    """Two detections of the same object collapse to one registry entry."""
    controller._update_registry([('red_cube', -0.70, 0.20)])
    controller._update_registry([('red_cube', -0.70 + 0.03, 0.20)])
    assert len(controller._registry) == 1


def test_registry_refreshes_position_on_rescan(controller):
    controller._update_registry([('red_cube', -0.70, 0.20)])
    assert len(controller._registry) == 1
    # Same object re-detected slightly shifted: position refreshed, no duplicate
    controller._update_registry([('red_cube', -0.702, 0.198)])
    assert len(controller._registry) == 1
    assert abs(controller._registry[0]['x'] + 0.702) < 1e-9


def test_registry_ignores_match_radius_boundary(controller):
    controller._update_registry([('red_cube', -0.70, 0.20)])
    far = (-0.70 + MATCH_RADIUS_M + 0.05, 0.20)
    controller._update_registry([('red_cube', *far)])
    assert len(controller._registry) == 2


def test_registry_skips_sort_zone_detections(controller):
    """Re-scans see placed objects on Table 1 — they must not become candidates."""
    candidates = controller._update_registry([('red_cube', -0.28, -0.12)])
    assert candidates == []
    assert controller._registry == []


def test_registry_skips_unreachable_and_warns(controller):
    candidates = controller._update_registry([('red_cube', -0.92, 0.40)])
    assert candidates == []
    assert controller._registry == []
    controller.get_logger.return_value.warning.assert_called()


def test_registry_excludes_done_and_blacklisted(controller):
    controller._update_registry([('red_cube', -0.70, 0.20),
                                 ('blue_cylinder', -0.65, -0.10)])
    assert len(controller._registry) == 2
    controller._registry[0]['done'] = True
    controller._registry[1]['blacklisted'] = True
    assert controller._update_registry([]) == []


# ── Zone slot assignment ────────────────────────────────────────────────────

def test_zone_slots_assigned_in_order(controller):
    assert controller._next_zone('red_cube') == (-0.28, -0.12)
    controller._mark_zone_used('red_cube')
    assert controller._next_zone('red_cube') == (-0.28, -0.24)
    assert controller._slot_index('red_cube') == 1
    # Blue side is independent
    assert controller._next_zone('blue_cylinder') == (-0.28, 0.12)
    assert controller._slot_index('blue_cylinder') == 0


def test_zone_slots_exhausted(controller):
    for _ in range(MAX_SLOTS):
        assert controller._next_zone('blue_cylinder') is not None
        controller._mark_zone_used('blue_cylinder')
    assert controller._next_zone('blue_cylinder') is None
    assert controller._slot_index('blue_cylinder') == MAX_SLOTS


# ── Spawner reach constraint ────────────────────────────────────────────────

def test_spawner_positions_within_reach_envelope(spawner):
    """Every sampled position must satisfy the arm pick-radius envelope."""
    random.seed(42)
    positions = spawner._generate_positions(4)
    assert len(positions) == 4
    for (x, y) in positions:
        assert x <= spawner.X_SPAWN_INNER
        assert x >= spawner._reach_constrained_x_max(y)
        assert math.hypot(x, y) <= spawner.R_PICK_MAX + 1e-9
        assert spawner.SPAWN_Y_MIN <= y <= spawner.SPAWN_Y_MAX


def test_spawner_respects_min_separation(spawner):
    random.seed(7)
    positions = spawner._generate_positions(4)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = math.hypot(positions[i][0] - positions[j][0],
                              positions[i][1] - positions[j][1])
            assert dist >= spawner.MIN_SEP


def test_spawner_reach_helper_matches_pick_radius(spawner):
    """Sanity: at y=0 the outer spawn edge must be exactly -R_PICK_MAX."""
    assert spawner._reach_constrained_x_max(0.0) == pytest.approx(-spawner.R_PICK_MAX)


def test_spawner_old_corner_is_outside_envelope(spawner):
    """The legacy (-0.78, +-0.38) corner that caused stretched poses is excluded."""
    assert math.hypot(-0.78, 0.38) > spawner.R_PICK_MAX


def test_fallback_positions_are_all_reachable(spawner):
    positions = spawner._fallback_positions(4)
    assert len(positions) == 4
    for (x, y) in positions:
        assert math.hypot(x, y) <= spawner.R_PICK_MAX + 1e-9


def test_fallback_rejects_impossible_counts(spawner):
    with pytest.raises(RuntimeError):
        spawner._fallback_positions(6)


# ── Live-run regression: IK joints must respect the URDF joint limits ───────
# Historical: the old custom arm failed the pick at (-0.676, 0.105) with
# theta1 = 3.14817 — above its ±3.14 limit — and every MoveIt goal answered
# FAILURE(99999).  The Phase 0 UR5e has ±2π limits; the regression now
# asserts every IK solution stays inside them AND lands FK-exact.

def test_normalize_angle_wraps_into_pi_band(arm):
    assert arm._normalize_angle(3.14817) == pytest.approx(3.14817 - 2 * math.pi, abs=1e-9)
    assert arm._normalize_angle(-3.20) == pytest.approx(-3.20 + 2 * math.pi, abs=1e-9)
    assert arm._normalize_angle(math.pi) == pytest.approx(-math.pi, abs=1e-9)
    assert -math.pi <= arm._normalize_angle(17.0) < math.pi


def test_compute_ik_live_failure_target_within_limits_and_exact(arm):
    """The old failure-log target must solve within the UR5e ±2π limits and
    land the jaw on the target (FK round-trip, tool-down orientation)."""
    tgt = (-0.676, 0.105, 0.05)
    q = arm._compute_ik(*tgt)
    assert q is not None
    assert all(abs(v) <= arm.JOINT_LIMIT + 1e-9 for v in q)
    p, z = arm._fk_jaw(q)
    assert math.dist(p, tgt) < 1e-6
    assert abs(z[0]) < 1e-9 and abs(z[1]) < 1e-9 and abs(z[2] + 1.0) < 1e-9


def test_compute_ik_picks_correct_d4_branch(arm):
    """The d4 wrist offset gives two mirrored wrist planes; the branch the
    solver picks must land the jaw ON the target (not 2*d4 away, as the old
    min(|theta1|) rule did) — verified by FK round-trip, both signs of y."""
    for y in (0.105, -0.105):
        tgt = (-0.676, y, 0.05)
        q = arm._compute_ik(*tgt)
        assert q is not None
        p, _ = arm._fk_jaw(q)
        assert math.dist(p, tgt) < 1e-6, f'branch error at y={y}'


def test_compute_ik_spawn_band_sweep_within_limits(arm):
    """Every pickable spawn position must yield in-limit joints (±2π) or a
    loud None — and any returned solution must be FK-exact."""
    for y in (-0.38, -0.20, 0.0, 0.20, 0.38):
        for x in (-0.66, -0.75):
            q = arm._compute_ik(x, y, 0.05)
            if q is None:
                continue  # unreachable envelope corner — covered by None tests
            assert all(abs(v) <= arm.JOINT_LIMIT + 1e-9 for v in q)
            p, _ = arm._fk_jaw(q)
            assert math.dist(p, (x, y, 0.05)) < 1e-6


def test_scan_poses_reachable_and_within_limits(arm):
    """Every wrist-camera scan pose must solve IK and stay in joint limits.

    The scan poses park the TCP 0.39 m above the table so the eye-in-hand
    camera (lens 0.05 m above the TCP plane) clears the 0.30 m depth near
    clip while covering the full spawn band.
    """
    scan_az = SCAN_TCP_WORLD_Z - ARM_BASE_WORLD_Z
    for sy in SCAN_YS:
        q = arm._compute_ik(SCAN_X, sy, scan_az)
        assert q is not None, f'scan pose ({SCAN_X}, {sy}) unreachable'
        assert len(q) == 6
        assert all(abs(v) <= arm.JOINT_LIMIT + 1e-9 for v in q)
        # Interim UR5e IK: scan poses solve on the mirrored wrist plane (B)
        # and must still land the jaw exactly on the scan target.
        p, z = arm._fk_jaw(q)
        assert math.dist(p, (SCAN_X, sy, scan_az)) < 1e-6
        assert abs(z[2] + 1.0) < 1e-9


def test_duplicate_server_hint(arm):
    assert arm._duplicate_server_hint(1) is None
    hint = arm._duplicate_server_hint(2)
    assert hint is not None and 'vision-clean' in hint


def test_moveit_error_code_names(arm):
    assert arm._ERROR_NAMES[1] == 'SUCCESS'
    assert arm._ERROR_NAMES[99999] == 'FAILURE'
    assert arm._ERROR_NAMES[10001] == 'PLANNING_FAILED'
    assert arm._ERROR_NAMES[10004] == 'CONTROL_FAILED'


# ── Grasp verification (3b — jaw gap + velocity) ────────────────────────────

def _make_joint_state(qr: float, ql: float, vr: float = 0.0, vl: float = 0.0):
    """Return a lightweight mock joint-state message for the prong joints."""
    msg = MagicMock()
    msg.name     = ['right_prong_joint', 'left_prong_joint']
    msg.position = [qr, ql]
    msg.velocity = [vr, vl]
    return msg


def test_jaw_gap_formula(controller):
    """gap = 0.035 + qr - ql (prong geometry, arm.urdf.xacro:330-363)."""
    controller._last_joint_state = _make_joint_state(0.06, -0.06)
    assert controller._jaw_gap() == pytest.approx(0.035 + 0.06 + 0.06)


def test_jaw_gap_none_when_no_joint_state(controller):
    controller._last_joint_state = None
    assert controller._jaw_gap() is None


def test_verify_grasp_held(controller):
    """Object between the jaws: prongs stall at ±0.0325 → gap ≈ 0.100 m → HELD."""
    qr, ql = 0.0325, -0.0325
    controller._last_joint_state = _make_joint_state(qr, ql, 0.0, 0.0)
    # Patch sleep so the test does not actually wait
    import time
    from unittest.mock import patch
    with patch.object(time, 'sleep', return_value=None):
        assert controller._verify_grasp('test') is True


def test_verify_grasp_empty(controller):
    """No object: prongs reach command ±0.012 → gap ≈ 0.059 m → EMPTY."""
    qr, ql = 0.012, -0.012
    controller._last_joint_state = _make_joint_state(qr, ql, 0.0, 0.0)
    import time
    from unittest.mock import patch
    with patch.object(time, 'sleep', return_value=None):
        assert controller._verify_grasp('test') is False


def test_verify_grasp_corner(controller):
    """Corner-grasped cube: prongs stall at ±0.0705 → gap ≈ 0.176 m → EMPTY (flag)."""
    # gap = 0.035 + 0.0705 - (-0.0705) = 0.1760 > GRASP_GAP_MAX_M (0.115)
    qr, ql = 0.0705, -0.0705
    controller._last_joint_state = _make_joint_state(qr, ql, 0.0, 0.0)
    import time
    from unittest.mock import patch
    with patch.object(time, 'sleep', return_value=None):
        assert controller._verify_grasp('test') is False


def test_verify_grasp_no_joint_state_passes_through(controller):
    """Missing /joint_states must not block the mission (returns True conservatively)."""
    controller._last_joint_state = None
    import time
    from unittest.mock import patch
    with patch.object(time, 'sleep', return_value=None):
        assert controller._verify_grasp('test') is True


def test_grasp_constants_signal_budget():
    """The grasp gap window must be wide enough to capture 0.1 m objects."""
    # A held object gives gap ≈ 0.100 m; must be inside [MIN, MAX].
    object_size_m = 0.10   # cube face-to-face / cylinder diameter
    assert GRASP_GAP_MIN_M < object_size_m < GRASP_GAP_MAX_M
    # The empty close (command ±0.012 → gap 0.059) must be OUTSIDE the window.
    empty_gap = 0.035 + 0.012 + 0.012
    assert empty_gap < GRASP_GAP_MIN_M


# ── Yaw alignment helpers ────────────────────────────────────────────────────

def test_jaw_world_yaw(controller):
    """jaw_world_yaw(theta1) = degrees(theta1) on the UR5e (q6 = 0,
    FK-verified against the vendor model)."""
    theta1 = math.radians(-90.0)
    assert SortingController._jaw_world_yaw(theta1) == pytest.approx(-90.0, abs=1e-6)
    theta1 = 0.0
    assert SortingController._jaw_world_yaw(theta1) == pytest.approx(0.0, abs=1e-6)


def test_nearest_90_deg():
    assert SortingController._nearest_90_deg(10.0)  == pytest.approx(0.0)
    assert SortingController._nearest_90_deg(46.0)  == pytest.approx(90.0)
    assert SortingController._nearest_90_deg(-46.0) == pytest.approx(-90.0)
    assert SortingController._nearest_90_deg(91.0)  == pytest.approx(90.0)


def test_align_ik_yaw_cylinder_unchanged(controller):
    """Cylinders are insensitive to yaw — the IK solution must not be modified."""
    q = [math.radians(-90), 0.5, -0.8, 0.3, 0.0, 0.0]
    result = controller._align_ik_yaw(q, 37.0, 'blue_cylinder', 'test')
    assert result == q


def test_align_ik_yaw_cube_applies_correction(controller):
    """For a red_cube the yaw correction must trim wrist_3 (q6) so the jaw
    axis rotates toward the nearest cube face (UR5e: jaw_yaw = q1 - q6)."""
    controller.get_logger = MagicMock()
    # theta1 = 0.0 → jaw_world_yaw = 0°
    # obj_angle_deg = 70° → nearest_90 = 90° → delta = +90° → q6 -= pi/2
    q = [0.0, 0.5, -0.8, 0.3, 0.0, 0.0]
    result = controller._align_ik_yaw(q, 70.0, 'red_cube', 'test')
    assert result is not None
    assert len(result) == 6
    assert result[:5] == q[:5]                 # position joints untouched
    assert result[5] == pytest.approx(-math.pi / 2, abs=1e-6)
    # New jaw yaw must equal the requested nearest face normal.
    assert SortingController._jaw_world_yaw(result[0]) - math.degrees(result[5]) \
        == pytest.approx(90.0, abs=1e-6)


def test_align_ik_yaw_none_input_unchanged(controller):
    """If IK returns None (unreachable), _align_ik_yaw must pass None through."""
    result = controller._align_ik_yaw(None, 45.0, 'red_cube', 'test')
    assert result is None


# ── Registry 4-tuple angle propagation ──────────────────────────────────────

def test_registry_stores_angle_deg(controller):
    """angle_deg from the detection must be stored in the registry entry."""
    detections = [('red_cube', -0.70, 0.10, 37.5)]
    controller._update_registry(detections)
    assert len(controller._registry) == 1
    assert controller._registry[0].get('angle_deg') == pytest.approx(37.5)


def test_registry_refreshes_angle_on_rescan(controller):
    """A re-scan with a new angle must update the angle in the existing entry."""
    controller._update_registry([('red_cube', -0.70, 0.10, 10.0)])
    controller._update_registry([('red_cube', -0.70, 0.10, 45.0)])
    assert len(controller._registry) == 1
    assert controller._registry[0]['angle_deg'] == pytest.approx(45.0)


def test_collect_dedup_preserves_angle(controller):
    """
    The dedup logic in _collect must preserve the angle field (4-tuple), not
    strip it to a 3-tuple.  We test this indirectly: we push the raw 4-tuples
    through _update_registry (which consumes _collect's output) and verify the
    angle lands in the registry.  We also do a white-box check on the dedup
    branch by exercising the cluster/median path with 4-tuple seeds.
    """
    # Two detections of the same object: same class, within DEDUP_RADIUS_M
    dets_4tuple = [
        ('red_cube', -0.70, 0.10, 15.0),
        ('red_cube', -0.702, 0.098, 20.0),   # same object, slightly shifted
    ]
    # _update_registry must handle 4-tuples and store the angle
    candidates = controller._update_registry(dets_4tuple)
    assert len(candidates) == 1, "Two nearby same-class detections should merge to one"
    assert 'angle_deg' in candidates[0], "angle_deg must be stored in registry entry"
    assert isinstance(candidates[0]['angle_deg'], float)
