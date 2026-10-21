#!/usr/bin/env python3
"""scene_config.py — single source of truth for geometry & motion constants.

Every number that used to be hardcoded inside ``pick_and_place.py`` /
``planning_scene_manager.py`` / ``sorting_controller.py`` (waypoints, table
geometry, object poses and sizes, grasp commands, world z of the work surface)
now lives in ``arm_moveit_config/config/robot_params.yaml`` and is exposed here
under the names those modules use.  Nothing in this package may re-type them.

Resolution order (first hit wins):
  1. ``<ament share>/arm_moveit_config/config/robot_params.yaml``  (installed WS)
  2. ``<repo>/src/arm_moveit_config/config/robot_params.yaml``     (bare pytest / dev)

Importing this module never raises because of a missing ament index: the
repo-relative fallback is always tried, so the unit tests can import the
scripts without a sourced workspace.

``test_params.py`` asserts this file's values against the URDF xacro
properties, ``worlds/vision_sorting.sdf`` and ``sort_zones`` so they can never
silently diverge from the model again.
"""

import os

import yaml

# scripts/ -> robotic_4dof_arm/ -> src/ -> arm_moveit_config/config/
_REPO_YAML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', 'arm_moveit_config', 'config', 'robot_params.yaml',
))


def _candidate_paths():
    """Installed share path first (if ament is importable/available), then repo."""
    paths = []
    try:
        from ament_index_python.packages import get_package_share_directory
        paths.append(os.path.join(
            get_package_share_directory('arm_moveit_config'),
            'config', 'robot_params.yaml',
        ))
    except Exception:          # no ament env (bare pytest) — repo fallback wins
        pass
    paths.append(_REPO_YAML)
    return paths


def load_path():
    """Absolute path of the robot_params.yaml actually loaded."""
    for path in _candidate_paths():
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        'robot_params.yaml not found; tried: ' + ', '.join(_candidate_paths())
    )


def load_robot_params():
    """The parsed robot_params.yaml as a plain dict."""
    with open(load_path(), 'r') as handle:
        return yaml.safe_load(handle)


PARAMS = load_robot_params()

# ── Blocks ────────────────────────────────────────────────────────────────────
KINEMATICS = PARAMS['kinematics']
GRASP      = PARAMS['grasp']
MISSION    = PARAMS['mission']
SCENE      = PARAMS['scene']
VISION     = PARAMS['vision']
SORT_ZONES = PARAMS['sort_zones']

# ── World-frame anchors ───────────────────────────────────────────────────────
ARM_BASE_WORLD_Z  = float(SCENE['arm_base_world_z'])   # robot spawn height
TABLE_TOP_WORLD_Z = float(SCENE['table_top_world_z'])  # work surface
# Objects spawn resting on the table: centre = table top + half the object.
HALF_OBJECT_M     = float(SCENE['objects']['red_cube']['size'][2]) / 2.0
PICK_WORLD_Z      = float(MISSION['pick_sequence']['grasp_world_z'])
# Nadir tool orientation (x, y, z, w) used by Cartesian goals.
TOOL_ORIENTATION  = tuple(float(v) for v in SCENE['tool_orientation'])

# ── Grasp commands (prong joint targets) ──────────────────────────────────────
GRIPPER_OPEN  = [float(v) for v in GRASP['open']]
GRIPPER_CLOSE = [float(v) for v in GRASP['close']]

# ── Arm joint position limit (tightest value in joints.arm.*) ─────────────────
def _arm_joint_limit() -> float:
    """Tightest |position limit| across every arm joint (URDF: ±3.14)."""
    limits = [
        abs(float(bound))
        for joint in PARAMS['joints']['arm'].values()
        for bound in (joint['min_position'], joint['max_position'])
    ]
    return min(limits) if limits else 3.14


JOINT_LIMIT = _arm_joint_limit()

# ── Vision filter (detection → pickable) ─────────────────────────────────────
_FILTER = VISION['pick_filter']
PICK_REGION_X_MIN = float(_FILTER['x_min'])
PICK_REGION_X_MAX = float(_FILTER['x_max'])
PICK_REGION_Y_MIN = float(_FILTER['y_min'])
PICK_REGION_Y_MAX = float(_FILTER['y_max'])

# ── MoveIt2 orientation tolerance (arm.urdf.xacro JA 4-5 degree rule of thumb).
# test_params.py::test_moveit_tolerances_json_keys asserts this block exists and
# would flag a missing value loudly (as if someone deleted grasp.open and left
# the code body defaulting to ±0.06 by accident).
MOVEIT = PARAMS['moveit']

# Orientation tolerance applied to Cartesian goals in move_to_pose().
ORIENT_TOLERANCE = float(MOVEIT['orientation_tolerance'])
# Joint-position tolerance applied to every joint constraint in move_to_joints().
JOINT_TOLERANCE  = float(MOVEIT['joint_tolerance'])

# ── Detection aggregation tuning ──────────────────────────────────────────────
CLUSTER_RADIUS_M = float(VISION['cluster_radius_m'])   # same-class cluster → one target
DEDUP_RADIUS_M   = float(VISION['dedup_radius_m'])     # in-scan dedup within this

# ── Phase 2: ActiveInspect defect-inspection policy ────────────────────────────
# mcp_server.py / defect_inspector.py read INSPECTION (never re-type a
# threshold); test file asserts every key below exists in robot_params.yaml.
INSPECTION = PARAMS['inspection']
INSP_SURVEY_HEIGHT_M     = float(INSPECTION['survey_height_m'])
INSP_OBLIQUE_DISTANCE_M  = float(INSPECTION['oblique_distance_m'])
INSP_OBLIQUE_TILT_DEG    = float(INSPECTION['oblique_tilt_deg'])
INSP_MAX_VIEWPOINTS      = int(INSPECTION['max_viewpoints'])
INSP_CERTIFY_THRESHOLD   = float(INSPECTION['certify_threshold'])
INSP_TRIAGE_LOW          = float(INSPECTION['triage_low'])
INSP_ADAPTIVE_BLOCK      = int(INSPECTION['adaptive_block_size'])
INSP_ADAPTIVE_C          = float(INSPECTION['adaptive_c'])
INSP_BILATERAL_D         = int(INSPECTION['bilateral_d'])
INSP_BILATERAL_SIG_COLOR = float(INSPECTION['bilateral_sigma_color'])
INSP_BILATERAL_SIG_SPACE = float(INSPECTION['bilateral_sigma_space'])
INSP_CRACK_ASPECT_MIN    = float(INSPECTION['crack_aspect_min'])
INSP_MIN_SPINE_MM        = float(INSPECTION['min_spine_length_mm'])
INSP_ANCHOR_RADIUS_MM    = float(INSPECTION['anchor_radius_mm'])
INSP_CERT_DIR            = str(INSPECTION['cert_dir'])

# ── Mission waypoints (world z, conversion happens at call time) ──────────────
PICK_SEQUENCE = {
    str(k): float(v) for k, v in MISSION['pick_sequence'].items()
}

# ── Scan geometry (TCP world targets for the wrist-camera scan pass) ──────────
SCAN = PARAMS['scan']
SCAN_TCP_WORLD_Z = float(SCAN['tcp_world_z'])
SCAN_X           = float(SCAN['x'])
SCAN_YS          = [float(v) for v in SCAN['ys']]
SCAN_SETTLE_SECS = float(SCAN['settle_secs'])

# ── Sorting mission tunables ───────────────────────────────────────────────────
SORTING = PARAMS['sorting']
SORT_MATCH_RADIUS_M               = float(SORTING['match_radius_m'])
SORT_EMPTY_SCANS_TO_FINISH        = int(SORTING['empty_scans_to_finish'])
SORT_MAX_ATTEMPTS_PER_OBJECT      = int(SORTING['max_attempts_per_object'])
SORT_REACH_MARGIN_M               = float(SORTING['reach_margin_m'])
SORT_CONSECUTIVE_FAIL_ABORT       = int(SORTING['consecutive_fail_abort'])
SORT_LIFT_HEIGHT_M                = float(SORTING['lift_height_m'])
SORT_GRASP_GAP_MIN_M              = float(SORTING['grasp_gap_min_m'])
SORT_GRASP_GAP_MAX_M              = float(SORTING['grasp_gap_max_m'])
SORT_GRASP_MAX_VEL_M_S            = float(SORTING['grasp_max_vel_m_s'])
SORT_GRASP_SETTLE_SECS            = float(SORTING['grasp_settle_secs'])
# objects lie on table_top_world_z; their stated z must map to that plane.
class PickTableZ:
    """World z for an object resting on the pick table (gazebo frame)."""
    @staticmethod
    def surface() -> float:
        return TABLE_TOP_WORLD_Z

    @classmethod
    def centre(cls) -> float:
        """Centre of an object that rests on the surface (size/0.2)."""
        return TABLE_TOP_WORLD_Z + HALF_OBJECT_M


def arm_z(world_z: float) -> float:
    """Convert a Gazebo world-frame z to the arm-local IK z."""
    return float(world_z) - ARM_BASE_WORLD_Z


def object_size(class_id: str):
    """Collision/geometry size of a detected class, or the demo box size."""
    spec = SCENE['objects'].get(class_id) or SCENE['target_box']
    return tuple(float(v) for v in spec['size'])


def in_pick_filter(x: float, y: float) -> bool:
    """True when (x, y) is inside the detection pick filter (Table 2 only)."""
    return (PICK_REGION_X_MIN <= x <= PICK_REGION_X_MAX and
            PICK_REGION_Y_MIN <= y <= PICK_REGION_Y_MAX)


def place_slot(class_id: str = 'red_cube', index: int = 0):
    """(x, y) of a sort-zone slot, or None when the grid has no such slot."""
    zones = SORT_ZONES.get('red' if class_id == 'red_cube' else 'blue') or []
    if index >= len(zones):
        return None
    slot = zones[index]
    return (float(slot[0]), float(slot[1]))


def place_pose(class_id: str = 'red_cube', index: int = 0):
    """(x, y) drop pose for a class — the sort-zone slot the sorter would use.

    Raises RuntimeError when the configured grid has no slot, because that is a
    configuration error (a mission with nowhere to put the object), not a state
    the arm should silently work around.
    """
    slot = place_slot(class_id, index)
    if slot is None:
        raise RuntimeError(
            f'sort_zones has no slot {index} for {class_id!r} — fix '
            f'robot_params.yaml (zones: red={len(SORT_ZONES.get("red") or [])}, '
            f'blue={len(SORT_ZONES.get("blue") or [])})'
        )
    return slot
