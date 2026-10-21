import math
import os
import re
import sys
import yaml
import pytest


PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'arm_moveit_config', 'config', 'robot_params.yaml')
URDF_PATH   = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'arm.urdf.xacro')

@pytest.fixture
def params():
    with open(PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)

def test_params_load(params):
    assert params is not None

def test_robot_section(params):
    assert 'use_sim_time' in params['robot']
    assert isinstance(params['robot']['use_sim_time'], bool)

def test_motion_section(params):
    required = ['max_velocity_scaling_factor', 'max_acceleration_scaling_factor', 'allowed_planning_time', 'num_planning_attempts', 'replan_attempts', 'max_retries', 'position_tolerance']

    for key in required:
        assert key in params['motion'], f"Missing key '{key}' in motion section"
    assert 0.0 < params['motion']['max_velocity_scaling_factor'] <= 1.0
    assert params['motion']['max_retries'] >= 1

def test_workspace_section(params):
    ws = params['workspace']
    assert len(ws['min_corner']) == 3
    assert len(ws['max_corner']) == 3
    for i in range(3):
        assert ws['min_corner'][i] < ws['max_corner'][i], f"min_corner[{i}] should be less than max_corner[{i}]"

def test_joint_names(params):
    arm_joints = list(params['joints']['arm'].keys())
    assert len(arm_joints) == 6, "Expected 6 arm joints"
    gripper_joints = list(params['joints']['gripper'].keys())
    assert len(gripper_joints) == 2, "Expected 2 gripper joints"

def test_home_position(params):
    home = params['joints']['home_position']
    arm_joints = list(params['joints']['arm'].keys())
    gripper_joints = list(params['joints']['gripper'].keys())
    for joint in arm_joints + gripper_joints:
        assert joint in home, f"Home position missing for joint '{joint}'"


# ── Kinematics block presence ────────────────────────────────────────────────

def test_kinematics_section_present(params):
    """robot_params.yaml must contain a 'kinematics' block (§6 drift-test anchor)."""
    assert 'kinematics' in params, "Missing 'kinematics' block in robot_params.yaml"


def test_kinematics_keys_present(params):
    kin = params['kinematics']
    for key in ('d1', 'a2', 'a3', 'd4', 'd5', 'd6', 'jaw_centre_offset',
                'flange_adapter', 'flange_to_jaw', 'tcp_grasp_reference',
                'dz_correction'):
        assert key in kin, f"Missing kinematics.{key}"


def test_kinematics_tcp_grasp_reference_formula(params):
    """tcp_grasp_reference must equal d6 + flange_adapter + jaw_centre_offset
    (the full wrist_1-pivot → jaw-centre stack of the Phase 0 UR5e model),
    and flange_to_jaw must equal flange_adapter + jaw_centre_offset."""
    kin = params['kinematics']
    expected = kin['d6'] + kin['flange_adapter'] + kin['jaw_centre_offset']
    assert abs(kin['flange_to_jaw']
               - (kin['flange_adapter'] + kin['jaw_centre_offset'])) < 1e-6, (
        f"kinematics.flange_to_jaw={kin['flange_to_jaw']} != "
        f"flange_adapter({kin['flange_adapter']}) + "
        f"jaw_centre_offset({kin['jaw_centre_offset']})"
    )
    assert abs(kin['tcp_grasp_reference'] - expected) < 1e-6, (
        f"kinematics.tcp_grasp_reference={kin['tcp_grasp_reference']} "
        f"!= d6({kin['d6']}) + flange_adapter({kin['flange_adapter']}) + "
        f"jaw_centre_offset({kin['jaw_centre_offset']}) = {expected}"
    )


def test_kinematics_tcp_grasp_reference_matches_pick_and_place():
    """pick_and_place.TCP_Z_OFFSET must equal robot_params.yaml tcp_grasp_reference."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from pick_and_place import PickAndPlace
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
    ref = params['kinematics']['tcp_grasp_reference']
    assert abs(PickAndPlace.TCP_Z_OFFSET - ref) < 1e-6, (
        f"PickAndPlace.TCP_Z_OFFSET={PickAndPlace.TCP_Z_OFFSET} "
        f"!= robot_params.kinematics.tcp_grasp_reference={ref}"
    )

def test_grasp_block_matches_scene_config(params):
    """grasp.open / grasp.close must equal what scene_config exposes to the nodes."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import scene_config as _scene
    assert [float(v) for v in params['grasp']['open']] == _scene.GRIPPER_OPEN
    assert [float(v) for v in params['grasp']['close']] == _scene.GRIPPER_CLOSE


def test_mission_block_present(params):
    """robot_params.yaml must contain the scripted-demo waypoint block."""
    assert 'mission' in params, "Missing 'mission' block in robot_params.yaml"
    seq = params['mission']['pick_sequence']
    for key in ('pre_pick_world_z', 'grasp_world_z', 'lift_world_z',
                'transport_world_z', 'place_world_z', 'retreat_world_z'):
        assert key in seq, f"Missing mission.pick_sequence.{key}"
        assert isinstance(seq[key], (int, float))


def test_mission_waypoint_heights(params):
    """Waypoints must sit above the table in the right order (world z)."""
    top = float(params['scene']['table_top_world_z'])
    seq = params['mission']['pick_sequence']
    assert seq['grasp_world_z'] == pytest.approx(top + 0.05)
    assert seq['pre_pick_world_z'] > seq['grasp_world_z']
    assert seq['lift_world_z'] > seq['pre_pick_world_z']


def test_mission_place_matches_sort_zones(params):
    """mission.*_world_z that reuse sort-zone heights must stay in sync."""
    seq = params['mission']['pick_sequence']
    sz = params['sort_zones']
    assert seq['place_world_z'] == pytest.approx(float(sz['place_height']))
    assert seq['transport_world_z'] == pytest.approx(float(sz['pre_place_height']))
    assert seq['retreat_world_z'] == pytest.approx(float(sz['pre_place_height']))


def test_tool_orientation_is_unit_quaternion(params):
    """scene.tool_orientation must be a unit (x, y, z, w) quaternion."""
    q = params['scene']['tool_orientation']
    assert len(q) == 4
    assert math.isclose(sum(float(v) ** 2 for v in q), 1.0, abs_tol=1e-3)


def test_scene_block_present(params):
    """robot_params.yaml must contain the world-geometry ('scene') block."""
    scene = params['scene']
    for key in ('arm_base_world_z', 'table_top_world_z', 'table_top_clearance',
                'tables', 'obstacle', 'target_box', 'objects', 'tool_orientation'):
        assert key in scene, f"Missing scene.{key}"
    assert len(scene['tables']) >= 2, 'Expected at least Table 1 + Table 2'
    for table in scene['tables']:
        assert len(table['size']) == 3 and len(table['centre']) == 3


def test_vision_block_present(params):
    """robot_params.yaml must wire use_vision + detection classes + pick filter."""
    vision = params['vision']
    for key in ('use_vision', 'detections_topic', 'detection_timeout_secs',
                'detection_classes', 'prefer_class', 'pick_filter', 'collect_secs'):
        assert key in vision, f"Missing vision.{key}"
    assert isinstance(vision['use_vision'], bool)
    for cls in vision['detection_classes']:
        assert cls in params['scene']['objects'], (
            f'vision.detection_classes has {cls!r} with no scene.objects entry')
    filt = vision['pick_filter']
    for key in ('x_min', 'x_max', 'y_min', 'y_max'):
        assert key in filt, f'Missing vision.pick_filter.{key}'
    assert filt['x_min'] < filt['x_max'] and filt['y_min'] < filt['y_max']


def test_vision_pick_filter_inside_pick_table(params):
    """The vision pick filter must sit inside Table 2's footprint (tighter than the
    physical edges so edge-clipped / partial detections are never treated as
    pickable), and x_max must stay clear of Table 1 (ends at x = -0.50).

    With the current table (-1.20 … -0.60 in x, ±0.40 in y) and filter
    (-1.00 … -0.55 in x, ±0.40 in y): the filter is inset from the far edge by
    20 cm (so far-edge objects are not picked) and extends 5 cm past the near
    edge toward Table 1 but stays at least 5 cm clear of it.
    """
    filt = params['vision']['pick_filter']
    table2 = next(t for t in params['scene']['tables'] if t['name'] == 'table_2')
    cx, cy = float(table2['centre'][0]), float(table2['centre'][1])
    sx, sy = float(table2['size'][0]), float(table2['size'][1])
    table_far, table_near = cx - sx / 2, cx + sx / 2   # -1.20, -0.60
    fmin, fmax = float(filt['x_min']), float(filt['x_max'])
    fymin, fymax = float(filt['y_min']), float(filt['y_max'])

    # x: filter is inside (or gently touching) the table and clear of Table 1.
    assert table_far - 0.05 <= fmin <= table_near, (
        f"x_min {fmin} must be inside Table 2's x range ({table_far}..{table_near})"
    )
    assert table_far <= fmax <= -0.50, (
        f"x_max {fmax} must sit on/near Table 2 and stay clear of Table 1 (-0.50)"
    )
    # y: filter must not exceed the table's y bounds.
    assert cy - sy / 2 - 1e-6 <= fymin, (
        f"y_min {fymin} must be >= table y_min {cy - sy / 2}"
    )
    assert fymax <= cy + sy / 2 + 1e-6, (
        f"y_max {fymax} must be <= table y_max {cy + sy / 2}"
    )


def test_scene_config_pick_sequence_matches_mission(params):
    """scene_config.PICK_SEQUENCE must expose mission.pick_sequence 1:1."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import scene_config as _scene
    seq = params['mission']['pick_sequence']
    assert set(_scene.PICK_SEQUENCE) == set(seq)
    for key, value in seq.items():
        assert _scene.PICK_SEQUENCE[key] == pytest.approx(float(value))


def test_scene_config_place_slot_matches_sort_zones(params):
    """scene_config.place_slot() must return the configured sort-zone slots."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import scene_config as _scene
    for cls in ('red_cube', 'blue_cylinder'):
        zones = params['sort_zones']['red' if cls == 'red_cube' else 'blue']
        for i, slot in enumerate(zones):
            assert _scene.place_slot(cls, i) == (float(slot[0]), float(slot[1]))
        assert _scene.place_slot(cls, len(zones)) is None


def test_scan_block_present(params):
    """robot_params.yaml must expose the wrist-camera scan poses."""
    scan = params['scan']
    for key in ('tcp_world_z', 'x', 'ys', 'settle_secs'):
        assert key in scan, f'Missing scan.{key}'
    assert len(scan['ys']) >= 1


# ── URDF vs YAML drift test (Phase 0: xacro-processed vendor model) ─────────

def _process_cell_xacro():
    """Process urdf/arm.urdf.xacro (ur_type:=ur5e) into URDF XML.

    Skips the xacro-based tests when the ROS environment is not sourced
    (bare pytest without ur_description in the ament index).
    """
    xacro = pytest.importorskip('xacro')
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory('ur_description')
    except Exception:
        pytest.skip('ur_description not in the ament index (source ROS first)')
    return xacro.process_file(URDF_PATH, mappings={'ur_type': 'ur5e'}).toxml()


def _urdf_joint_table(urdf_xml: str) -> dict:
    """{joint_name: dict(type, parent, child, xyz, rpy, axis)} of every joint."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(urdf_xml)
    table = {}
    for j in root.findall('joint'):
        o = j.find('origin')
        xyz = tuple(float(v) for v in o.attrib.get('xyz', '0 0 0').split()) \
            if o is not None else (0.0, 0.0, 0.0)
        rpy = tuple(float(v) for v in o.attrib.get('rpy', '0 0 0').split()) \
            if o is not None else (0.0, 0.0, 0.0)
        axis_e = j.find('axis')
        axis = tuple(float(v) for v in axis_e.attrib.get('xyz', '0 0 1').split()) \
            if axis_e is not None else (0.0, 0.0, 1.0)
        table[j.attrib['name']] = dict(
            type=j.attrib['type'], parent=j.find('parent').attrib['link'],
            child=j.find('child').attrib['link'], xyz=xyz, rpy=rpy, axis=axis)
    return table


def _urdf_fk_jaw(urdf_xml: str, q, flange_to_jaw: float):
    """FK of the generated URDF at joint vector q (six arm joints in
    base_link → tool0 order).  Returns (jaw_position, tool_z_axis); the jaw
    centre sits flange_to_jaw below the flange along the tool axis."""
    import numpy as np
    table = _urdf_joint_table(urdf_xml)
    by_child = {j['child']: j for j in table.values()}
    chain, cur = [], 'tool0'
    while cur != 'base_link':
        j = by_child[cur]
        chain.append(j)
        cur = j['parent']
    chain.reverse()

    def rpy_R(r, p, y):
        cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                  math.sin(p), math.cos(y), math.sin(y))
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]])

    def axis_R(axis, ang):
        ax = np.array(axis, float)
        x, y, z = ax / np.linalg.norm(ax)
        c, s, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])

    T = np.eye(4)
    qi = 0
    for j in chain:
        M = np.eye(4)
        M[:3, :3] = rpy_R(*j['rpy'])
        M[:3, 3] = j['xyz']
        T = T @ M
        if j['type'] in ('revolute', 'continuous'):
            R = np.eye(4)
            R[:3, :3] = axis_R(j['axis'], q[qi])
            T = T @ R
            qi += 1
    z = T[:3, 2]
    jaw = T[:3, 3] + z * flange_to_jaw
    return jaw, z


def test_kinematics_match_urdf(params):
    """
    Phase 0 drift test: process the cell wrapper xacro with ur_type:=ur5e and
    assert the generated URDF encodes exactly the kinematics.* scalars of
    robot_params.yaml (shoulder z, elbow |x|, wrist_1 |x|/z, wrist_2 |y|,
    wrist_3 |y|, flange-adapter height).  If ur_description is upgraded and
    these diverge, the IK silently mis-picks — this must fail loudly first.
    """
    xml = _process_cell_xacro()
    origins = _urdf_joint_table(xml)
    kin = params['kinematics']
    for joint, idx, key in (('shoulder_pan_joint', 2, 'd1'),
                            ('elbow_joint', 0, 'a2'),
                            ('wrist_1_joint', 0, 'a3'),
                            ('wrist_1_joint', 2, 'd4'),
                            ('wrist_2_joint', 1, 'd5'),
                            ('wrist_3_joint', 1, 'd6')):
        got = abs(origins[joint]['xyz'][idx])
        assert got == pytest.approx(kin[key], abs=1e-9), (
            f"URDF {joint} xyz[{idx}]={got} != kinematics.{key}={kin[key]} "
            "(drift detected — update robot_params.yaml or ur_description)"
        )
    assert origins['gripper_mount_joint']['xyz'][2] == pytest.approx(
        kin['flange_adapter'], abs=1e-9), 'flange adapter height drifted'


def test_pick_and_place_fk_matches_urdf(params):
    """The runtime FK (PickAndPlace._fk_jaw) must reproduce the vendor URDF
    FK at randomized joint sets to < 1e-9 m (guard for the interim IK)."""
    xml = _process_cell_xacro()
    import random as _rng
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from pick_and_place import PickAndPlace
    kin = params['kinematics']
    rng = _rng.Random(7)
    for _ in range(10):
        q = [rng.uniform(-math.pi, math.pi) for _ in range(6)]
        p_ref, z_ref = _urdf_fk_jaw(xml, q, kin['flange_to_jaw'])
        p, z = PickAndPlace._fk_jaw(q)
        for got, want in zip(p, p_ref):
            assert got == pytest.approx(float(want), abs=1e-9)
        for got, want in zip(z, z_ref):
            assert got == pytest.approx(float(want), abs=1e-9)


def test_home_pose_fk_tool_down(params):
    """The configured home must spawn the arm standing with the tool pointing
    straight down toward the pick table (-X) — Phase 0 gate item."""
    xml = _process_cell_xacro()
    kin = params['kinematics']
    home = params['joints']['home_position']
    q = [float(home[j]) for j in params['joints']['arm']]
    p, z = _urdf_fk_jaw(xml, q, kin['flange_to_jaw'])
    assert abs(z[0]) < 1e-9 and abs(z[1]) < 1e-9
    assert z[2] == pytest.approx(-1.0, abs=1e-9)
    # FK reference from the vendor model: jaw over the -X side of the base.
    assert p[0] == pytest.approx(-0.2925, abs=1e-3)
    assert p[1] == pytest.approx(-0.1333, abs=1e-3)
    assert p[2] == pytest.approx(0.4149, abs=1e-3)
