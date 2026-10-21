#!/usr/bin/env python3
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    BoundingVolume,
)
import yaml

# Ensure script directory is in path so we can import PlanningSceneManager
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from planning_scene_manager import PlanningSceneManager
import scene_config as _scene


class PickAndPlace(Node):
    # ── IK geometry & grasp commands: single source is robot_params.yaml ──────
    # Exposed by scene_config (which resolves the YAML via ament, with a
    # repo-relative fallback).  test_params.py asserts these match the
    # generated UR5e URDF (xacro-processed, FK-verified) and the grasp block,
    # so they can never drift silently.
    D1_SHOULDER  = float(_scene.KINEMATICS['d1'])   # shoulder height above base
    L1_LENGTH    = float(_scene.KINEMATICS['a2'])   # upper arm length (|a2|)
    L2_LENGTH    = float(_scene.KINEMATICS['a3'])   # forearm length   (|a3|)
    D4_OFFSET    = float(_scene.KINEMATICS['d4'])   # wrist_1 lateral offset
    D5_OFFSET    = float(_scene.KINEMATICS['d5'])   # wrist_2 in-plane offset
    D6_OFFSET    = float(_scene.KINEMATICS['d6'])   # flange offset along tool
    # Flange (tool0) → jaw centre along the tool axis: ISO adapter + jaw.
    FLANGE_TO_JAW = float(_scene.KINEMATICS['flange_to_jaw'])
    # Wrist_1 pivot height above the shoulder =
    #   z_jaw + TCP_Z_OFFSET - DZ_CORRECTION.
    TCP_Z_OFFSET = float(_scene.KINEMATICS['tcp_grasp_reference'])
    DZ_CORRECTION = float(_scene.KINEMATICS['dz_correction'])
    # Tightest arm-joint position limit from joints.arm.* (UR5e ±2π).
    JOINT_LIMIT      = _scene.JOINT_LIMIT

    # ── UR5e structural transforms (ur_description, ur_type:=ur5e) ───────────
    # Fixed rpy values from the vendor URDF chain base_link → tool0; the scalar
    # offsets above come from robot_params.yaml.  test_params.py::
    # test_kinematics_match_urdf processes the cell xacro and asserts this FK
    # matches the generated URDF to < 1e-9 m.
    _RPY_BASE_INERTIA = (0.0, 0.0, math.pi)                # base_link → inertia
    _RPY_UPPER_ARM    = (math.pi / 2, 0.0, 0.0)            # → upper_arm_link
    _RPY_WRIST_2      = (math.pi / 2, 0.0, 0.0)            # → wrist_2_link
    _RPY_WRIST_3      = (math.pi / 2, math.pi, math.pi)    # → wrist_3_link
    _RPY_FLANGE       = (0.0, -math.pi / 2, -math.pi / 2)  # wrist_3 → flange
    _RPY_TOOL0        = (math.pi / 2, 0.0, math.pi / 2)    # flange → tool0

    def __init__(self):
        params_path = _scene.load_path()
        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)
        super().__init__(
            'pick_and_place_autonomous',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, self.params['robot']['use_sim_time'])
            ]
        )
        # 1. Action client for MoveIt MoveGroup (arm motion planning)
        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')

        # 2. Action client for Gripper controller
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory'
        )

        # 2b. Action client for the arm controller (preflight health check —
        #     a missing arm controller otherwise only surfaces as mid-mission
        #     motion failures).
        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory'
        )

        self.joint_names = list(self.params['joints']['arm'].keys())
        self.gripper_joint_names = list(self.params['joints']['gripper'].keys())

        # 3. Initialize PlanningSceneManager
        self.scene = PlanningSceneManager(node=self)

        # 4. Vision: the scripted demo takes the grasp pose from
        #    /vision/detections when vision.use_vision is true, and only falls
        #    back to the configured scene.target_box pose when nothing arrives
        #    (the fallback is logged loudly at pick time, never silent).
        self._vision = dict(_scene.VISION)
        self._detections = []
        self._detection_sub = None
        if self._vision.get('use_vision', False):
            try:
                from vision_msgs.msg import Detection2DArray
                self._detection_sub = self.create_subscription(
                    Detection2DArray,
                    self._vision['detections_topic'],
                    self._detection_cb,
                    10,
                )
                self.get_logger().info(
                    f'Vision enabled — subscribing to '
                    f'{self._vision["detections_topic"]} '
                    f'(prefer_class={self._vision["prefer_class"]})'
                )
            except Exception as exc:      # vision_msgs missing → fallback pose
                self.get_logger().warning(
                    f'vision.use_vision is true but {type(exc).__name__}: {exc} '
                    f'— falling back to scene.target_box'
                )
                self._vision['use_vision'] = False

        self.get_logger().info('Waiting for MoveGroup and Gripper action servers...')
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('MoveGroup action server not available')
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('Gripper action server not available')
        self.get_logger().info('Action servers connected successfully!')

    # ── Vision input (pick_and_place demo) ───────────────────────────────────
    def _detection_cb(self, msg):
        """Cache the latest detections published by shape_detector.py.

        Only classes listed in vision.detection_classes and inside
        vision.pick_filter are kept, so objects already placed in the sort
        zones (Table 1) are never re-picked.
        """
        allowed = set(self._vision.get('detection_classes', []))
        for det in msg.detections:
            if not det.results:
                continue
            cls = det.results[0].hypothesis.class_id
            if cls not in allowed:
                continue
            pos = det.results[0].pose.pose.position
            if not _scene.in_pick_filter(pos.x, pos.y):
                continue
            self._detections.append({
                'cls': cls,
                'x': float(pos.x),
                'y': float(pos.y),
                'z': float(pos.z),
            })

    def _pick_target(self):
        """(x, y, class, source) for the next grasp.

        Vision first: spin until /vision/detections delivers a candidate of
        vision.prefer_class inside the pick filter (nearest first), then take
        the mean of the samples.  Only when nothing arrives within
        vision.detection_timeout_secs does the configured scene.target_box
        pose get used — and that fallback is logged, never silent.
        """
        box = _scene.SCENE['target_box']
        fallback = (
            float(box['position'][0]), float(box['position'][1]), box['id'],
            'scene.target_box (vision fallback)',
        )
        if not self._vision.get('use_vision', False) or self._detection_sub is None:
            self.get_logger().warning(
                f'vision.use_vision is false — using the configured pose '
                f'({fallback[0]:.2f}, {fallback[1]:.2f})'
            )
            return fallback

        prefer = self._vision.get('prefer_class')
        timeout = float(self._vision.get('detection_timeout_secs', 6.0))
        self._detections.clear()
        self.get_logger().info(
            f'Waiting up to {timeout:.1f} s for /vision/detections '
            f'({prefer} inside the pick filter)...'
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if any(d['cls'] == prefer for d in self._detections):
                break

        candidates = [d for d in self._detections if d['cls'] == prefer]
        if not candidates:
            self.get_logger().warning(
                f'No {prefer} detection within {timeout:.1f} s — FALLING BACK to '
                f'the configured pose ({fallback[0]:.2f}, {fallback[1]:.2f}). '
                f'This is NOT a perceived object.'
            )
            return fallback

        candidates.sort(key=lambda d: math.hypot(d['x'], d['y']))
        best = candidates[0]
        cluster = [d for d in candidates
                   if math.hypot(d['x'] - best['x'], d['y'] - best['y']) < _scene.CLUSTER_RADIUS_M]
        x = sum(d['x'] for d in cluster) / len(cluster)
        y = sum(d['y'] for d in cluster) / len(cluster)
        self.get_logger().info(
            f'Pick target from vision: {best["cls"]} at ({x:.3f}, {y:.3f}) '
            f'[{len(cluster)} samples of {len(candidates)} candidate(s)]'
        )
        return (x, y, best['cls'], 'vision')

    @staticmethod
    def _normalize_angle(a: float) -> float:
        """Wrap an angle into [-pi, pi)."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    # ── UR5e forward kinematics (compact, from robot_params.yaml scalars) ─────
    @classmethod
    def _fk_jaw(cls, q):
        """FK base_link → tool0 for the six arm joints, plus the fixed gripper.

        Returns (jaw_position_xyz, tool_z_axis); the jaw centre sits
        FLANGE_TO_JAW below the flange along the tool axis.  Pure python — no
        URDF needed at runtime; parity with the vendor model is enforced by
        test_params.py::test_kinematics_match_urdf.
        """
        def T(xyz, rpy):
            cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
            cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
            cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
            return [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
                [-sp, cp * sr, cp * cr, xyz[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]

        def mul(A, B):
            return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
                    for i in range(4)]

        def rotz(a):
            c, s = math.cos(a), math.sin(a)
            return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

        M = T((0.0, 0.0, 0.0), cls._RPY_BASE_INERTIA)
        M = mul(M, T((0.0, 0.0, cls.D1_SHOULDER), (0.0, 0.0, 0.0)))
        M = mul(M, rotz(q[0]))
        M = mul(M, T((0.0, 0.0, 0.0), cls._RPY_UPPER_ARM))
        M = mul(M, rotz(q[1]))
        M = mul(M, T((-cls.L1_LENGTH, 0.0, 0.0), (0.0, 0.0, 0.0)))
        M = mul(M, rotz(q[2]))
        M = mul(M, T((-cls.L2_LENGTH, 0.0, cls.D4_OFFSET), (0.0, 0.0, 0.0)))
        M = mul(M, rotz(q[3]))
        M = mul(M, T((0.0, -cls.D5_OFFSET, 0.0), cls._RPY_WRIST_2))
        M = mul(M, rotz(q[4]))
        M = mul(M, T((0.0, cls.D6_OFFSET, 0.0), cls._RPY_WRIST_3))
        M = mul(M, T((0.0, 0.0, 0.0), cls._RPY_FLANGE))
        M = mul(M, T((0.0, 0.0, 0.0), cls._RPY_TOOL0))
        z = (M[0][2], M[1][2], M[2][2])
        jaw = (M[0][3] + z[0] * cls.FLANGE_TO_JAW,
               M[1][3] + z[1] * cls.FLANGE_TO_JAW,
               M[2][3] + z[2] * cls.FLANGE_TO_JAW)
        return jaw, z

    def is_reachable(self, x: float, y: float, z: float, margin: float = 0.0) -> bool:
        """
        Checks whether TCP position (x, y, z) lies inside the interim UR5e IK
        envelope (either wrist plane). `margin` (metres) shrinks the usable
        reach so targets are solvable with the arm not fully stretched.
        """
        R = math.sqrt(x**2 + y**2)
        if R <= self.D4_OFFSET:
            return False  # wrist offset makes asin(d4/R) undefined
        dz = z + self.TCP_Z_OFFSET - self.DZ_CORRECTION
        reach = self.L1_LENGTH + self.L2_LENGTH - margin
        inner = abs(self.L1_LENGTH - self.L2_LENGTH)
        r_sqrt = math.sqrt(R**2 - self.D4_OFFSET**2)
        for rho in (r_sqrt + self.D5_OFFSET, -r_sqrt + self.D5_OFFSET):
            if inner <= math.sqrt(rho**2 + dz**2) <= reach:
                return True
        return False

    def _planar_seed(self, x: float, y: float, z: float, branch: str) -> list | None:
        """Planar (q1, q2, q3) seed for the tool-down IK (elbow-up).

        Branch A: wrist plane on the near side (rho = +sqrt(R²-d4²) + d5),
        Branch B: mirrored plane (rho = -sqrt(R²-d4²) + d5).  Joint conventions
        are FK-pinned: phi2 = -q2, phi3 = -q3 (see phased_implementation.md §0).
        """
        R = math.sqrt(x**2 + y**2)
        if R <= self.D4_OFFSET:
            return None
        a0 = math.atan2(y, x)
        alpha = math.asin(min(1.0, self.D4_OFFSET / R))
        r_sqrt = math.sqrt(R**2 - self.D4_OFFSET**2)
        if branch == 'A':
            t1 = a0 - alpha
            rho = r_sqrt + self.D5_OFFSET
        else:
            t1 = a0 - math.pi + alpha
            rho = -r_sqrt + self.D5_OFFSET
        dz = z + self.TCP_Z_OFFSET - self.DZ_CORRECTION  # wrist_1 above shoulder
        D = math.sqrt(rho**2 + dz**2)
        if D > self.L1_LENGTH + self.L2_LENGTH or \
                D < abs(self.L1_LENGTH - self.L2_LENGTH):
            return None
        cos_phi3 = ((D**2 - self.L1_LENGTH**2 - self.L2_LENGTH**2)
                    / (2.0 * self.L1_LENGTH * self.L2_LENGTH))
        phi3 = -math.acos(max(-1.0, min(1.0, cos_phi3)))   # elbow-up
        phi2 = math.atan2(dz, rho) - math.atan2(
            self.L2_LENGTH * math.sin(phi3),
            self.L1_LENGTH + self.L2_LENGTH * math.cos(phi3))
        return [t1, -phi2, -phi3]

    def _newton_tool_down(self, x: float, y: float, z: float,
                          seed: list) -> list | None:
        """3-DOF Newton polish on the exact tool-down manifold.

        Tool-down poses are exactly {(q1, q2, q3): q4 = pi/2 - (q2 + q3),
        q5 = pi/2, q6 = 0} (FK-verified), so the jaw position map is 3→3 and
        converges to machine precision from the planar seed.
        """
        import numpy as np

        def full(q3d):
            q4 = self._normalize_angle(math.pi / 2.0 - float(q3d[1])
                                       - float(q3d[2]))
            return [float(q3d[0]), float(q3d[1]), float(q3d[2]), q4,
                    math.pi / 2.0, 0.0]

        q3d = np.array(seed, float)
        h, tol, iters = 1e-7, 1e-9, 15
        for _ in range(iters):
            p, _ = self._fk_jaw(full(q3d))
            if math.dist(p, (x, y, z)) < tol:
                break
            J = np.zeros((3, 3))
            for i in range(3):
                dq = np.zeros(3)
                dq[i] = h
                pp, _ = self._fk_jaw(full(q3d + dq))
                pm, _ = self._fk_jaw(full(q3d - dq))
                J[:, i] = [(pp[k] - pm[k]) / (2.0 * h) for k in range(3)]
            e = np.array([x - p[0], y - p[1], z - p[2]])
            try:
                step = np.linalg.solve(J, e)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(J, e, rcond=None)[0]
            norm = float(np.linalg.norm(step))
            if norm > 0.5:
                step *= 0.5 / norm
            q3d = q3d + step
            q3d[1:] = np.clip(q3d[1:], -2.0 * math.pi, 2.0 * math.pi)
            q3d[0] = float(self._normalize_angle(q3d[0]))
        q = full(q3d)
        p, za = self._fk_jaw(q)
        if math.dist(p, (x, y, z)) > 1e-6 or abs(za[0]) > 1e-9 \
                or abs(za[1]) > 1e-9 or abs(za[2] + 1.0) > 1e-9:
            return None
        # Wrap into (-pi, pi]: equivalent poses, shorter MoveIt joint paths.
        return [self._normalize_angle(v) for v in q]

    def _ik_branches(self, x: float, y: float, z: float) -> list:
        """Return all FK-exact tool-down IK solutions (8 max) for the target."""
        R = math.hypot(x, y)
        if R <= self.D4_OFFSET:
            return []
        a0 = math.atan2(y, x)
        alpha = math.asin(min(1.0, self.D4_OFFSET / R))
        r_sqrt = math.sqrt(R * R - self.D4_OFFSET * self.D4_OFFSET)
        dz = z + self.TCP_Z_OFFSET - self.DZ_CORRECTION
        L1, L2 = self.L1_LENGTH, self.L2_LENGTH
        # Base (shoulder) angle solutions: plane A / B.
        plane_defs = (
            (a0 - alpha, 1.0),    # A:  t1, base rho = +r_sqrt
            (a0 - math.pi + alpha, -1.0),  # B: t1, base rho = -r_sqrt
        )
        out = []
        for t1, base_sign in plane_defs:
            for flip in (1, -1):
                rho = base_sign * r_sqrt + flip * self.D5_OFFSET
                D = math.hypot(rho, dz)
                if D > L1 + L2 or D < abs(L1 - L2):
                    continue
                cos_phi3 = ((D * D - L1 * L1 - L2 * L2)
                            / (2.0 * L1 * L2))
                cos_phi3 = max(-1.0, min(1.0, cos_phi3))
                acos_phi3 = math.acos(cos_phi3)
                for phi3 in (acos_phi3, -acos_phi3):
                    phi2 = math.atan2(dz, rho) - math.atan2(
                        L2 * math.sin(phi3), L1 + L2 * math.cos(phi3))
                    q1 = self._normalize_angle(t1)
                    q2 = self._normalize_angle(-phi2)
                    q3 = self._normalize_angle(-phi3)
                    q4 = self._normalize_angle(
                        flip * (math.pi / 2.0) - q2 - q3)
                    q5 = flip * (math.pi / 2.0)
                    q = [q1, q2, q3, q4, q5, 0.0]
                    p, za = self._fk_jaw(q)
                    if math.dist(p, (x, y, z)) > 1e-6:
                        continue
                    if abs(za[0]) > 1e-9 or abs(za[1]) > 1e-9 \
                            or abs(za[2] + 1.0) > 1e-9:
                        continue
                    out.append([float(v) for v in q])
        return out

    def _compute_ik(self, x: float, y: float, z: float) -> list | None:
        """
        Closed-form UR5e IK (8 branches): analytical 2-link planar solve for
        (q1, q2, q3) on both wrist planes (A/B) and both elbow signs, times
        both wrist (q5) flips for the tool-down orientation.  Returns the first
        solution that is FK-exact and inside the UR5e joint limits, or None.
        """
        branches = self._ik_branches(x, y, z)
        in_limit = [q for q in branches
                    if all(abs(v) <= self.JOINT_LIMIT + 1e-9 for v in q)]
        if in_limit:
            return in_limit[0]
        if branches:
            self.get_logger().warning(
                f'IK ({x:.3f}, {y:.3f}, {z:.3f}): solution exceeds joint limits'
            )
            return branches[0]
        if math.hypot(x, y) <= self.D4_OFFSET:
            self.get_logger().warning(
                f'IK ({x:.3f}, {y:.3f}, {z:.3f}): target inside wrist offset'
            )
        else:
            self.get_logger().warning(
                f'IK ({x:.3f}, {y:.3f}, {z:.3f}): no tool-down solution in '
                'either wrist plane'
            )
        return None

    def _ik_or_abort(self, x: float, y: float, z: float, step: str) -> list:
        """
        IK wrapper for mission steps: aborts loudly when the target is
        unreachable instead of sending a clamped (bogus) joint goal.
        """
        q = self._compute_ik(x, y, z)
        if q is None:
            self._abort(f'{step}: target ({x:.2f}, {y:.2f}, {z:.2f}) is unreachable by IK')
        return q


    def _is_within_workspace(self, x: float, y: float, z: float) -> bool:
        """
        Checks if the given Cartesian coordinates are within the defined workspace.
        """
        ws = self.params['workspace']
        return (
            ws['min_corner'][0] <= x <= ws['max_corner'][0] and
            ws['min_corner'][1] <= y <= ws['max_corner'][1] and
            ws['min_corner'][2] <= z <= ws['max_corner'][2]
        )

    # Symbolic names for the moveit_msgs/MoveItErrorCodes enum.
    _ERROR_NAMES = {
        1: 'SUCCESS',
        99999: 'FAILURE',
        10001: 'PLANNING_FAILED',
        10002: 'INVALID_MOTION_PLAN',
        10003: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
        10004: 'CONTROL_FAILED',
        10005: 'UNACHIEVABLE',
        10007: 'TIMED_OUT',
        10008: 'PREEMPTED',
        10009: 'START_STATE_IN_COLLISION',
        10010: 'START_STATE_VIOLATES_START_STATE_CONSTRAINTS',
        -31: 'NO_IK_SOLUTION',
    }

    def _execute_move_group_goal(self, goal: MoveGroup.Goal, label: str = '') -> bool:
        future = self._move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error(f'Goal rejected: {label}')
            return False

        result_future = goal_handle.get_result_async()
        # NOTE: rclpy.spin_until_future_complete() returns None in some rclpy
        # versions — never test its return value; test future.done() instead.
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        if not result_future.done():
            self.get_logger().error(
                f'Motion timed out after 60 s: {label} — cancelling.'
            )
            goal_handle.cancel_goal()
            return False
        result = result_future.result().result
        if result.error_code.val == 1:  # 1 = SUCCESS
            self.get_logger().info(f'{label} motion succeeded.')
            time.sleep(0.5)
            return True
        code = result.error_code.val
        name = self._ERROR_NAMES.get(code, f'UNKNOWN({code})')
        self.get_logger().error(f'Motion failed [{name} code {code}]: {label}')
        if code == 99999:
            self.get_logger().error(
                '  FAILURE(99999) hints: a stale second move_group owns '
                '/move_action (run `make vision-clean`), the simulation is '
                'paused, or the controllers are not spawned.'
            )
        return False

    def move_to_pose(self, x: float, y: float, z: float, orientation: tuple = None, link_name: str = 'gripper_tcp') -> bool:
        """
        Plans and executes a collision-free path to the target Cartesian pose using MoveIt 2.
        Validates workspace bounds before sending. Retries on failure.

        orientation: (x, y, z, w) quaternion; defaults to the configured
        scene.tool_orientation (nadir / pointing down) — no literal quaternion.
        """
        qx, qy, qz, qw = orientation or _scene.TOOL_ORIENTATION
        if not self._is_within_workspace(x, y, z):
            self.get_logger().error(
                f'Target ({x:.2f}, {y:.2f}, {z:.2f}) is OUTSIDE workspace bounds. Refusing goal.'
            )
            return False
        self.get_logger().info(f'Planning collision-aware path to: ({x:.2f}, {y:.2f}, {z:.2f})')

        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.pipeline_id = 'ompl'
        goal.request.planner_id = 'RRTConnectkConfigDefault'
        m = self.params['motion']
        goal.request.num_planning_attempts = m['num_planning_attempts']
        goal.request.allowed_planning_time = m['allowed_planning_time']
        goal.request.max_velocity_scaling_factor = m['max_velocity_scaling_factor']
        goal.request.max_acceleration_scaling_factor = m['max_acceleration_scaling_factor']
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = m['replan_attempts']
        goal.request.start_state.is_diff = True


        # Workspace parameters
        ws = self.params['workspace']
        goal.request.workspace_parameters.header.frame_id = 'world'
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        goal.request.workspace_parameters.min_corner.x = ws['min_corner'][0]
        goal.request.workspace_parameters.min_corner.y = ws['min_corner'][1]
        goal.request.workspace_parameters.min_corner.z = ws['min_corner'][2]
        goal.request.workspace_parameters.max_corner.x = ws['max_corner'][0]
        goal.request.workspace_parameters.max_corner.y = ws['max_corner'][1]
        goal.request.workspace_parameters.max_corner.z = ws['max_corner'][2]

        # Target Pose Constraints
        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        target_pose.orientation.x = float(qx)
        target_pose.orientation.y = float(qy)
        target_pose.orientation.z = float(qz)
        target_pose.orientation.w = float(qw)

        # Position constraint
        tol = self.params['motion']['position_tolerance']
        pcm = PositionConstraint()
        pcm.header.frame_id = 'world'
        pcm.header.stamp = self.get_clock().now().to_msg()
        pcm.link_name = link_name
        bv = BoundingVolume()
        box_prim = SolidPrimitive()
        box_prim.type = SolidPrimitive.BOX
        box_prim.dimensions = [tol, tol, tol]  # tolerance box
        bv.primitives = [box_prim]
        bv.primitive_poses = [target_pose]
        pcm.constraint_region = bv
        pcm.weight = 1.0

        # Orientation constraint
        ocm = OrientationConstraint()
        ocm.header.frame_id = 'world'
        ocm.header.stamp = self.get_clock().now().to_msg()
        ocm.link_name = link_name
        ocm.orientation = target_pose.orientation
        ocm.absolute_x_axis_tolerance = _scene.ORIENT_TOLERANCE
        ocm.absolute_y_axis_tolerance = _scene.ORIENT_TOLERANCE
        ocm.absolute_z_axis_tolerance = _scene.ORIENT_TOLERANCE
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints = [pcm]
        constraints.orientation_constraints = [ocm]
        goal.request.goal_constraints = [constraints]

        return self._execute_move_group_goal(goal, label=f'Cartesian ({x:.2f}, {y:.2f}, {z:.2f})')

    def move_to_joints(self, joint_positions: list) -> bool:
        """
        Plans and executes a collision-free motion to target joint positions using MoveIt 2.
        """
        if joint_positions is None:
            self.get_logger().error(
                'move_to_joints received None (unreachable IK target). Refusing goal.'
            )
            return False
        self.get_logger().info(f'Planning collision-aware path to joints: {joint_positions}')

        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.pipeline_id = 'ompl'
        goal.request.planner_id = 'RRTConnectkConfigDefault'
        m = self.params['motion']
        goal.request.num_planning_attempts = m['num_planning_attempts']
        goal.request.allowed_planning_time = m['allowed_planning_time']
        goal.request.max_velocity_scaling_factor = m['max_velocity_scaling_factor']
        goal.request.max_acceleration_scaling_factor = m['max_acceleration_scaling_factor']
        goal.planning_options.plan_only = False
        goal.request.start_state.is_diff = True

        constraints = Constraints()
        for name, pos in zip(self.joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = _scene.JOINT_TOLERANCE
            jc.tolerance_below = _scene.JOINT_TOLERANCE
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal.request.goal_constraints = [constraints]

        return self._execute_move_group_goal(goal, label='Joint-space')

    def move_gripper(self, positions: list, duration_sec: float = 1.5):
        """
        Commands the gripper controller to open or close prongs.
        """
        self.get_logger().info(f'Moving gripper to prongs: {positions}')
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error(f'Gripper goal REJECTED: {positions}')
        else:
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
            if not result_future.done():
                self.get_logger().error('Gripper motion timed out — cancelling.')
                goal_handle.cancel_goal()
            else:
                res = result_future.result().result
                if res.error_code != 0:  # FollowJointTrajectoryResult.SUCCESS = 0
                    self.get_logger().error(
                        f'Gripper motion failed [code {res.error_code}]: {positions}'
                    )
        time.sleep(0.5)

    # ── MoveGroup stack health ────────────────────────────────────────────
    def _move_group_server_count(self) -> int:
        """
        Every live MoveGroup action server publishes one /move_action feedback
        endpoint, so the feedback publisher count equals the server count.
        Returns 1 when discovery reports nothing (single-server default).
        """
        try:
            count = self.count_publishers('/move_action/_action/feedback')
        except Exception:
            return 1
        return count if count > 0 else 1

    def _duplicate_server_hint(self, server_count: int) -> str | None:
        """Pure helper (unit-tested): actionable message when >1 server."""
        if server_count <= 1:
            return None
        return (
            f'{server_count} MoveGroup action servers own /move_action! A stale '
            'stack answers goals with FAILURE(99999). Close old ROS terminals '
            'or run `make vision-clean`, then restart vision-sim + vision-sort.'
        )

    def preflight_check(self) -> bool:
        self.get_logger().info('Running preflight checks...')
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('PREFLIGHT FAIL: MoveGroup server not available.')
            return False
        hint = self._duplicate_server_hint(self._move_group_server_count())
        if hint:
            self.get_logger().error(f'PREFLIGHT FAIL: {hint}')
            return False
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('PREFLIGHT FAIL: Gripper server not available.')
            return False
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'PREFLIGHT FAIL: /arm_controller/follow_joint_trajectory not '
                'available — was `make vision-sim` started (controllers spawned)?'
            )
            return False
        from sensor_msgs.msg import JointState
        received = [False]
        def _cb(_msg): received[0] = True
        sub = self.create_subscription(JointState, '/joint_states', _cb, 1)
        deadline = time.time() + 3.0
        while not received[0] and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        if not received[0]:
            self.get_logger().error('PREFLIGHT FAIL: /joint_states not publishing.')
            return False
        self.get_logger().info('Preflight checks passed ✓')
        return True

    def _abort(self, step: str):
        """
        Called when a mission step fails.
        Attempts a safe return to home, then raises RuntimeError to halt the mission.
        """
        self.get_logger().error(f'══ MISSION ABORT at step: [{step}] ══')
        self.get_logger().warning('Attempting emergency return to home pose...')
        home = list(self.params['joints']['home_position'].values())
        self.move_to_joints(home)
        raise RuntimeError(f'Mission aborted at: {step}')

    def run(self):
        """
        Executes the complete collision-aware pick and place cycle.
        """
        if not self.preflight_check():
            self.get_logger().error('Preflight checks failed. Aborting mission.')
            return
        self.get_logger().info('========================================================')
        self.get_logger().info('   Starting Autonomous Collision-Aware Pick and Place   ')
        self.get_logger().info('========================================================')

        home = list(self.params['joints']['home_position'].values())

        # ── Waypoints: robot_params.yaml mission.pick_sequence, converted to
        #    arm-local IK z once (arm_z = world_z - scene.arm_base_world_z).
        seq = {name: _scene.arm_z(world_z)
               for name, world_z in _scene.PICK_SEQUENCE.items()}

        # 1. Initialize MoveIt Planning Scene (tables + pick object) from YAML
        self.get_logger().info('\n[1/11] Initializing Planning Scene from robot_params.yaml...')
        self.scene.init_scene_from_params()
        time.sleep(1.0)

        # 2. Return to Home pose (upright safe stance)
        self.get_logger().info('\n[2/11] Moving to Safe Home Pose...')
        if not self.move_to_joints(home):
            self._abort('Step 2: Home Pose')

        # 3. Open Gripper (grasp.open from robot_params.yaml)
        self.get_logger().info('\n[3/11] Opening Gripper...')
        self.move_gripper(_scene.GRIPPER_OPEN)

        # 4. Perception decides the grasp pose — no pose is baked into this
        #    script any more.  _pick_target() asks /vision/detections for the
        #    nearest object inside the pick filter and only uses the configured
        #    fallback pose (logging it) when vision delivers nothing.
        pick_x, pick_y, obj_cls, obj_source = self._pick_target()
        obj_size = _scene.object_size(obj_cls)
        obj_id = obj_cls
        table_top = _scene.TABLE_TOP_WORLD_Z
        self.get_logger().info(
            f'Grasp pose source: {obj_source} → {obj_cls} at '
            f'({pick_x:.3f}, {pick_y:.3f}), size {obj_size}'
        )
        # Put the perceived object into the planning scene so the planner sees it
        self.scene.add_object_from_params(
            obj_id, obj_cls, (pick_x, pick_y, table_top)
        )

        # 5. Move to Pre-Pick Pose (waypoint from mission.pick_sequence)
        self.get_logger().info(
            f'\n[4/11] Planning path to Pre-Pick Pose '
            f'({pick_x:.2f}, {pick_y:.2f}, world z={_scene.PICK_SEQUENCE["pre_pick_world_z"]:.2f})...'
        )
        if not self.move_to_joints(
                self._ik_or_abort(pick_x, pick_y, seq['pre_pick_world_z'],
                                  'Step 4: Pre-Pick Pose')):
            self._abort('Step 4: Pre-Pick Pose')

        # 6. Descend to Grasp Pose (surrounding the perceived object)
        self.get_logger().info('\n[5/11] Descending to Grasp Height...')
        if not self.move_to_joints(
                self._ik_or_abort(pick_x, pick_y, seq['grasp_world_z'],
                                  'Step 5: Grasp Height')):
            self._abort('Step 5: Grasp Height')

        # 7. Close Gripper & Attach the object in MoveIt
        self.get_logger().info('\n[6/11] Closing Gripper & Attaching Object to MoveIt...')
        self.move_gripper(_scene.GRIPPER_CLOSE)
        self.scene.attach_object_to_gripper(
            obj_id, obj_size, link_name='gripper_base_link'
        )
        time.sleep(0.5)

        # 8. Lift Object (Vertical Retreat)
        self.get_logger().info('\n[7/11] Lifting Object vertically...')
        if not self.move_to_joints(
                self._ik_or_abort(pick_x, pick_y, seq['lift_world_z'],
                                  'Step 7: Lift')):
            self._abort('Step 7: Lift')

        # 9. Collision-Aware Transport to the drop pose.  The drop pose is the
        #    first sort-zone slot for the object's class (robot_params.yaml
        #    sort_zones) — exactly where the vision sorter puts that class.
        drop_x, drop_y = _scene.place_pose(obj_cls)
        self.get_logger().info('\n[8/11] Planning Collision-Free Path to Pre-Place...')
        if not self.move_to_joints(
                self._ik_or_abort(drop_x, drop_y, seq['transport_world_z'],
                                  'Step 8: Transport')):
            self._abort('Step 8: Transport')

        # 10. Descend to Place Pose (onto the destination marker)
        self.get_logger().info('\n[9/11] Lowering to Place Pose...')
        if not self.move_to_joints(
                self._ik_or_abort(drop_x, drop_y, seq['place_world_z'],
                                  'Step 9: Place Height')):
            self._abort('Step 9: Place Height')

        # 11. Open Gripper & Detach Object in MoveIt
        self.get_logger().info('\n[10/11] Opening Gripper & Detaching Object...')
        self.move_gripper(_scene.GRIPPER_OPEN)
        self.scene.detach_object_from_gripper(
            obj_id, (drop_x, drop_y, table_top), obj_size
        )
        time.sleep(0.5)

        # 12. Retreat Upwards & Return to Home
        self.get_logger().info('\n[11/11] Retreating Upwards & Returning Home...')
        if not self.move_to_joints(
                self._ik_or_abort(drop_x, drop_y, seq['retreat_world_z'],
                                  'Step 11: Retreat')):
            self._abort('Step 11: Retreat')
        if not self.move_to_joints(home):
            self._abort('Step 11: Return Home')

        self.get_logger().info('========================================================')
        self.get_logger().info('   Pick and Place Mission Completed Successfully! ✓     ')
        self.get_logger().info('========================================================')


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    try:
        node.run()
    except RuntimeError as e:
        node.get_logger().error(f'Mission failed: {e}')
    except KeyboardInterrupt:
        node.get_logger().info('Pick and place interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
