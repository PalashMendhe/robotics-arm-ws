#!/usr/bin/env python3
"""
sorting_controller.py  —  Week 3: Autonomous Vision-Based Sorting
------------------------------------------------------------------
Perception-driven sorting for the KNOWN 2+2 scene (2 red cubes + 2 blue
cylinders; MAX_SLOTS = 2 per class — the mission can sort at most 4):

  home → [ scan → pick → lift → traverse → place → retreat ] × N → home

The controller never assumes how many objects the world contains.  It scans
Table 2, sorts the nearest reachable object, then re-scans.  The mission ends
only when EMPTY_SCANS_TO_FINISH consecutive scans find nothing new — so the
cycle count always equals the number of objects actually on the table.

COORDINATE FIX
--------------
The shape_detector publishes world-frame (x, y, z) but TF calibration may
drift.  Since all objects spawn at world z=1.00 (table surface 0.95 + half
object 0.05), the grasp uses the fixed PICK_WORLD_Z; only (x, y) from the
detector steer the arm.  Detected z is logged for TF validation only.

IK z conversion:  arm_z = world_z - ARM_BASE_WORLD_Z (0.95)
  pick:        arm_z = 0.05   (at object centre)
  pre-pick:    arm_z = 0.15   (+10 cm above object)
  lift:        arm_z = 0.15   (straight up after grasp)
  traverse:    lifted pose → above the zone at pre-place height
  place:       arm_z = 0.10
  pre-place:   arm_z = 0.25

MOTION SEQUENCE (per object) — no home visit anywhere inside the loop
---------------------------------------------------------------------
  home (once at mission start)
  loop (per object):
    1. open gripper
    2. pre-pick   (above object)    MoveIt
    3. grasp      (descend)         MoveIt
    4. close gripper
    5. lift       (straight up)     MoveIt   ← short vertical leg
    6. traverse   (above the zone)  MoveIt   ← plans FROM the lifted pose,
                                                 so the path is a short high
                                                 swing — the arm can no
                                                 longer fold back to home
    7. place      (descend)         MoveIt
    8. open gripper
    9. retreat    (straight up)     MoveIt   ← ready for the next pick
  home (once at mission end)

FAILURE HANDLING
----------------
  * Pre-pick / grasp failure → object untouched; the cycle is retried later.
  * Lift / traverse failure  → retried from the lifted pose; if retries are
    exhausted the object is lowered back onto the pick table (never dropped
    mid-swing) and the cycle reports failure.
  * An object that fails MAX_ATTEMPTS_PER_OBJECT times is blacklisted so the
    mission can never loop forever.
"""

import math
import os
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from vision_msgs.msg import Detection2DArray

current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pick_and_place import PickAndPlace   # noqa: E402
import scene_config as _scene              # noqa: E402


# ── Coordinate constants ──────────────────────────────────────────────────────
# Single source = robot_params.yaml (via scene_config).  Nothing below re-types
# a world offset, a grasp command or a pick-region bound: the same YAML block is
# what planning_scene_manager publishes and what test_params.py cross-checks
# against the URDF / world SDF.
ARM_BASE_WORLD_Z = _scene.ARM_BASE_WORLD_Z   # arm spawn height (gazebo world)
PICK_WORLD_Z     = _scene.PICK_WORLD_Z       # mission.pick_sequence.grasp_world_z

# ── Detection tuning ──────────────────────────────────────────────────────────
DETECTIONS_TOPIC       = _scene.VISION['detections_topic']
DETECTION_COLLECT_SECS = float(_scene.VISION.get('collect_secs', 5.0))
DEDUP_RADIUS_M         = 0.10  # in-scan: detections within 10 cm → same object
MATCH_RADIUS_M         = 0.07  # cross-scan registry match (< MIN_SEP / 2)
# Grasp commands: robot_params.yaml grasp.open / grasp.close (single source).
GRIPPER_OPEN           = list(_scene.GRIPPER_OPEN)    # 0.155 m gap clears a 45-deg cube
GRIPPER_CLOSE          = list(_scene.GRIPPER_CLOSE)   # 20.5 mm grasp signal
# Placement capacity is a property of the sort_zones grid, never a literal.
MAX_SLOTS              = min(len(_scene.SORT_ZONES.get('red') or [0]),
                             len(_scene.SORT_ZONES.get('blue') or [0]))


# ── Grasp verification (idea 3b): jaw gap + settled velocity ────────────────
# Jaw geometry (arm.urdf.xacro:330-363): prong origins at y = +/-0.02,
# axis +Y on BOTH joints.  Inner-face gap = 0.035 + qr - ql.
# A 0.1 m object stalls the jaws at gap = 0.10 (qr = +0.0325, ql = -0.0325).
# With GRIPPER_CLOSE = [0.012, -0.012] the command (gap 0.059) can never
# reach the object faces, so 3b compares the MEASURED gap against the
# object size: HELD = gap in [0.085, 0.115] with both jaw velocities ~0.
# A grasp is accepted when the measured jaw gap (from /joint_states via
# the inherited node) sits in [GRASP_GAP_MIN_M, GRASP_GAP_MAX_M].
GRASP_GAP_MIN_M     = 0.085   # 0.10 m object minus tolerance / slight rotation
GRASP_GAP_MAX_M     = 0.115   # open-gap value; also catches 0.141 m corner grasps
GRASP_MAX_VEL_M_S   = 0.01    # jaws must be settled
GRASP_SETTLE_SECS   = 0.4     # wait after close before sampling

# ── Wrist-camera scan poses (TCP world targets) ──────────────────────────────
# The eye-in-hand camera looks straight down the tool axis, so the image
# footprint is centred under the TCP.  Two poses cover the spawn band
# (x ∈ [-0.78, -0.63], y ∈ [-0.38, +0.38]):
#   TCP world z = 1.34 → lens 0.39 m above the table → objects 0.34 m from the
#   lens (outside the 0.30 m near clip; jaws/prongs are culled by the clip);
#   footprint ≈ 0.47 (world y) × 0.35 (world x) m per pose → y-coverage
#   ±0.435 with a 0.07 m overlap strip (registry merges via MATCH_RADIUS_M);
#   x-coverage [-0.806, -0.454] (the extreme far-corner sliver x < -0.806 is
#   clipped but the object centre still lands in frame, so it is detected).
# Reach check (is_reachable at the parked pose): D = 0.811 ≤ L1+L2 = 0.817 ✓.
# E6: the lens is ALSO ~5 cm sideways of the tool axis (+X in the flange
# frame, camera_params.yaml wrist_camera.xyz) and that offset ROTATES with
# theta1, so the true nadir point is not exactly (SCAN_X, SCAN_YS).  The
# margin absorbs it: at h ≈ 0.39 m a 5 cm shift is ~15 % of the half-width.
# If SCAN_X/YS are ever re-tuned, recompute from the TF-derived LENS pose
# (gripper_base_link -> d415_wrist_link from robot_state_publisher), never
# from the TCP alone.
#
# Single source = robot_params.yaml `scan:` (via scene_config).  These module
# names stay so the existing tests and _scan_pass() need no API change.
_SCAN = _scene.PARAMS['scan']
SCAN_TCP_WORLD_Z = float(_SCAN['tcp_world_z'])
SCAN_X           = float(_SCAN['x'])
SCAN_YS          = [float(v) for v in _SCAN['ys']]
SCAN_SETTLE_SECS = float(_SCAN['settle_secs'])

# ── Mission loop tuning ───────────────────────────────────────────────────────
LIFT_HEIGHT_M           = 0.10  # vertical clearance for pre-pick / lift legs
EMPTY_SCANS_TO_FINISH   = 2     # consecutive empty scans that end the mission
MAX_ATTEMPTS_PER_OBJECT = 2     # failed cycles before an object is blacklisted
REACH_MARGIN_M          = 0.02  # IK envelope margin used to filter candidates
CONSECUTIVE_FAIL_ABORT  = 3     # consecutive failed cycles → systemic MoveIt failure

# ── Pick-region filter (Table 2 only — placed objects live on Table 1) ───────
# Values come from robot_params.yaml vision.pick_filter via scene_config, so the
# same bounds steer the sorting registry, the scripted demo grasp and the tests.
# PICK_REGION_* stay as importable names because test_vision_sorting.py imports
# them (with tight bounds deliberately narrower than the physical Table 2
# footprint x: -1.20 … -0.60, whose inner edge is x = -0.60; Table 1 ends at x = -0.50).
PICK_REGION_X_MIN = _scene.PICK_REGION_X_MIN
PICK_REGION_X_MAX = _scene.PICK_REGION_X_MAX
PICK_REGION_Y_MIN = _scene.PICK_REGION_Y_MIN
PICK_REGION_Y_MAX = _scene.PICK_REGION_Y_MAX



def _arm_z(world_z: float) -> float:
    """Convert Gazebo world-frame z to arm-local IK z."""
    return world_z - ARM_BASE_WORLD_Z


class SortingController(PickAndPlace):
    """
    Extends PickAndPlace. Overrides run() with a vision-driven sorting mission.
    Inherits: _compute_ik, is_reachable, move_to_joints, move_gripper,
    preflight_check, _abort.
    """

    def __init__(self):
        super().__init__()
        self.get_logger().info(
            f'SortingController ready  |  PICK_WORLD_Z={PICK_WORLD_Z}  '
            f'ARM_BASE_WORLD_Z={ARM_BASE_WORLD_Z}'
        )

        # ── Detection buffer ─────────────────────────────────────────────
        self._buf: list = []
        self._det_sub = self.create_subscription(
            Detection2DArray, DETECTIONS_TOPIC, self._detection_cb, 10
        )
        from sensor_msgs.msg import JointState
        self._last_joint_state = None
        self._js_sub = self.create_subscription(
            JointState, '/joint_states',
            lambda msg: setattr(self, '_last_joint_state', msg), 10
        )

        # ── Sort zones from YAML (world z applied at call time) ───────────
        sz = self.params['sort_zones']
        self._place_wz     = float(sz['place_height'])      # e.g. 1.05
        self._pre_place_wz = float(sz['pre_place_height'])  # e.g. 1.20
        self._red_zones    = [(float(p[0]), float(p[1])) for p in sz['red']]
        self._blue_zones   = [(float(p[0]), float(p[1])) for p in sz['blue']]

        # ── Mission state: registry of tracked objects (count unknown) ────
        self._registry: list = []   # [{'cls', 'x', 'y', 'attempts', 'done', 'blacklisted'}]
        self._red_used  = 0         # placement slots consumed in the red zone
        self._blue_used = 0         # placement slots consumed in the blue zone

    # ── Detection callback ────────────────────────────────────────────────
    def _detection_cb(self, msg: Detection2DArray):
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            cls = hyp.hypothesis.class_id
            x   = hyp.pose.pose.position.x
            y   = hyp.pose.pose.position.y
            # bbox.center.theta carries the minAreaRect angle (degrees)
            # published by shape_detector — used for jaw yaw alignment (§5.2).
            angle_deg = det.bbox.center.theta
            if self._allowed_class(cls):
                self._buf.append((cls, x, y, angle_deg))

    # ── Snapshot + dedup ──────────────────────────────────────────────────
    def _collect(self) -> list:
        """
        Spin for DETECTION_COLLECT_SECS, then return deduplicated list of
        (class_id, x, y, angle_deg) using XY-only deduplication (z ignored).

        angle_deg is the minAreaRect angle from bbox.center.theta — the most
        recent sample in the cluster is kept so the angle is fresh.
        """
        self.get_logger().info(
            f'Collecting detections for {DETECTION_COLLECT_SECS:.0f} s ...'
        )
        self._buf.clear()
        deadline = time.time() + DETECTION_COLLECT_SECS
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

        # Deduplicate: cluster by (x, y) within DEDUP_RADIUS_M
        remaining = list(self._buf)
        unique = []
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            outliers = []
            for item in remaining:
                if (item[0] == seed[0] and
                        math.hypot(item[1]-seed[1], item[2]-seed[2]) < DEDUP_RADIUS_M):
                    cluster.append(item)
                else:
                    outliers.append(item)
            remaining = outliers
            # Representative: median x, y; latest angle (last in cluster)
            xs = sorted(c[1] for c in cluster)
            ys = sorted(c[2] for c in cluster)
            mid = len(cluster) // 2
            # Angle: use the last sample in the cluster (most recent frame)
            latest_angle = cluster[-1][3] if len(cluster[0]) > 3 else 0.0
            unique.append((seed[0], xs[mid], ys[mid], latest_angle))

        return unique

    # ── Pick-region + reachability filters ────────────────────────────────
    def _on_pick_table(self, x: float, y: float) -> bool:
        """True when (x, y) is inside the Table 2 pick filter.

        Delegates to scene_config.in_pick_filter (robot_params.yaml
        vision.pick_filter) so the sorting registry, the scripted demo and the
        unit tests all share one definition.  This excludes anything the camera
        sees outside Table 2 — in particular objects already placed in the
        Table 1 sort zones — so placed objects are never re-picked.
        """
        return _scene.in_pick_filter(x, y)

    def _allowed_class(self, cls: str) -> bool:
        """True when the detector class is enabled in vision.detection_classes."""
        allowed = _scene.VISION.get('detection_classes') or (
            'red_cube', 'blue_cylinder')
        return cls in allowed

    def _object_reachable(self, x: float, y: float) -> bool:
        """
        True when the grasp, pre-pick AND lift waypoints are all solvable
        with a safety margin.  Guards against the old failure mode where an
        out-of-reach target made IK silently clamp to a bogus stretched pose.
        """
        pick_az = _arm_z(PICK_WORLD_Z)
        return all(
            self.is_reachable(x, y, az, margin=REACH_MARGIN_M)
            for az in (pick_az, pick_az + LIFT_HEIGHT_M)
        )

    # ── Cross-scan object registry (tracks an unknown number of objects) ──
    def _update_registry(self, detections: list) -> list:
        """
        Matches fresh detections to already-tracked objects (within
        MATCH_RADIUS_M), refreshes their positions and yaw, registers new
        objects, and returns the sortable candidates — nearest to the arm first.
        Candidates must be: on the pick table, reachable, not done and not
        blacklisted.

        detections: list of (cls, x, y, angle_deg) from _collect/_scan_pass.
        """
        for det in detections:
            cls, x, y = det[0], det[1], det[2]
            angle_deg = det[3] if len(det) > 3 else 0.0
            if not self._on_pick_table(x, y):
                continue  # e.g. objects already placed on Table 1
            best, best_d = None, MATCH_RADIUS_M
            for e in self._registry:
                if e['cls'] != cls or e['done'] or e['blacklisted']:
                    continue
                d = math.hypot(x - e['x'], y - e['y'])
                if d < best_d:
                    best, best_d = e, d
            if best is not None:
                best['x'], best['y'] = x, y        # refresh tracked position
                best['angle_deg'] = angle_deg       # refresh yaw from latest scan
                continue
            if not self._object_reachable(x, y):
                self.get_logger().warning(
                    f'Ignoring {cls} at ({x:.2f}, {y:.2f}): outside arm reach'
                )
                continue
            self._registry.append({
                'cls': cls, 'x': x, 'y': y, 'angle_deg': angle_deg,
                'attempts': 0, 'done': False, 'blacklisted': False,
            })

        candidates = [
            e for e in self._registry if not e['done'] and not e['blacklisted']
        ]
        candidates.sort(key=lambda e: math.hypot(e['x'], e['y']))
        return candidates

    # ── Zone slot assignment (by placement counter, not detection index) ──
    def _next_zone(self, cls: str):
        """Next free placement slot for a class, or None when all are full."""
        zones = self._red_zones if cls == 'red_cube' else self._blue_zones
        used  = self._red_used  if cls == 'red_cube' else self._blue_used
        if used >= min(MAX_SLOTS, len(zones)):
            return None
        return zones[used]

    def _mark_zone_used(self, cls: str):
        if cls == 'red_cube':
            self._red_used += 1
        else:
            self._blue_used += 1

    def _slot_index(self, cls: str) -> int:
        return self._red_used if cls == 'red_cube' else self._blue_used

    # ── Wrist-camera multi-pose scan (look-then-move) ─────────────────────
    def _scan_pass(self) -> list:
        """
        Park the TCP directly over each scan pose and collect detections with
        the eye-in-hand camera.  Because the camera looks straight down the
        tool axis, the image footprint is centred under the TCP — parking is
        all the aiming the scan needs.

        Per pose: settle (SCAN_SETTLE_SECS), then _collect() spins for
        DETECTION_COLLECT_SECS while the arm is stationary, so every
        back-projected sample uses a frozen TF (look-then-move; the stale
        motion frames queued before the settle are discarded by _collect's
        buffer clear).  Returns one deduplicated detection list with
        cross-pose duplicates merged (same class within DEDUP_RADIUS_M).
        """
        merged: list = []
        scan_az = _arm_z(SCAN_TCP_WORLD_Z)
        for sy in SCAN_YS:
            q = self._compute_ik(SCAN_X, sy, scan_az)
            if q is None:
                self.get_logger().warning(
                    f'[scan] pose ({SCAN_X}, {sy}, {scan_az:.3f}) unreachable '
                    f'— skipped (spawn band partially uncovered)'
                )
                continue
            if not self.move_to_joints(q):
                self.get_logger().warning(
                    f'[scan] move to ({SCAN_X}, {sy}) FAILED — partial scan'
                )
                continue
            time.sleep(SCAN_SETTLE_SECS)
            merged.extend(self._collect())

        # Cross-pose merge: an object in the overlap strip is reported by
        # both poses; keep the first representative within DEDUP_RADIUS_M.
        unique: list = []
        for det in merged:
            cls, x, y = det[0], det[1], det[2]
            angle_deg = det[3] if len(det) > 3 else 0.0
            if not any(
                c == cls and math.hypot(x - ux, y - uy) < DEDUP_RADIUS_M
                for (c, ux, uy, *_) in unique
            ):
                unique.append((cls, x, y, angle_deg))
        return unique


    # ── Grasp verification helpers ───────────────────────────────────
    def _prong_states(self):
        """Latest (pos, vel) of both prong joints from /joint_states."""
        qr = ql = vr = vl = None
        msg = getattr(self, '_last_joint_state', None)
        if msg is not None:
            for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
                if name == 'right_prong_joint':
                    qr, vr = pos, vel
                elif name == 'left_prong_joint':
                    ql, vl = pos, vel
        return qr, ql, vr, vl

    def _jaw_gap(self):
        """Measured jaw gap in metres, or None when no joint state yet."""
        qr, ql, _, _ = self._prong_states()
        if qr is None or ql is None:
            return None
        return 0.035 + qr - ql   # 0.035 m rest gap + symmetric travel

    def _verify_grasp(self, label):
        """True when the jaws stalled on an object after closing.

        Verifies IN PLACE: the arm stays exactly where the close finished
        (descended on the object) while the prong positions settle, then
        the measured jaw gap is gated (E2 — no motion is commanded here;
        the pre-E2 version needlessly re-drove to the grasp joints first).
        """
        _ = label  # kept for log context at the call site
        time.sleep(GRASP_SETTLE_SECS)   # let prong positions settle (no motion)
        gap = self._jaw_gap()
        qr, ql, vr, vl = self._prong_states()
        if gap is None:
            self.get_logger().warning(f'  [{label}] grasp unverified (no /joint_states yet)')
            return True   # do not block the mission on a missing sample
        held = (GRASP_GAP_MIN_M <= gap <= GRASP_GAP_MAX_M
                and abs(vr or 0.0) <= GRASP_MAX_VEL_M_S
                and abs(vl or 0.0) <= GRASP_MAX_VEL_M_S)
        self.get_logger().info(
            f'  [{label}] grasp check: gap={gap:.4f} m '
            f'(qr={qr:+.4f}, ql={ql:+.4f}) -> {"HELD" if held else "EMPTY"}'
        )
        return held

    # ── Jaw yaw alignment (§5.2, Finding C) ──────────────────────────────
    @staticmethod
    def _jaw_world_yaw(theta1: float) -> float:
        """
        Compute the jaw's current yaw in the world frame from the base joint
        angle θ1 (radians).

        On the UR5e (Phase 0 baseline) the jaw travel axis is the gripper
        base's Y axis.  FK-verified against the vendor URDF: with wrist_3
        (q6) zero the jaw axis sits exactly θ1 from world +X, and for any
        q6 the relation is jaw_yaw = degrees(θ1 - q6) while the tool z-axis
        stays (0, 0, -1) — so q6 is a pure yaw trim at tool-down poses.

        Returned in degrees (consistent with minAreaRect convention).
        """
        return math.degrees(theta1)

    @staticmethod
    def _nearest_90_deg(angle_deg: float) -> float:
        """Round angle to the nearest multiple of 90°, result in [0, 360)."""
        return round(angle_deg / 90.0) * 90.0

    def _align_ik_yaw(self, q: list, obj_angle_deg: float,
                      class_id: str, label: str) -> list:
        """
        Apply yaw alignment to an IK joint solution by adding a wrist-roll
        correction Δθ5 (joint index 4 in the 6-DOF list) so the jaw faces
        align with the nearest cube face (§5.2, Finding C).

        Cylinders are insensitive to yaw → returned unchanged.
        The correction is clamped to the ±3.14 rad joint limit.
        TCP position is unchanged (verified: 0.00 mm shift at all sampled poses).

        q         : 6-DOF joint list from _compute_ik
        obj_angle : minAreaRect angle in degrees (0° = camera col, measured CW)
        class_id  : 'red_cube' or 'blue_cylinder'
        """
        if class_id != 'red_cube':
            return q   # cylinder: any yaw is valid
        if q is None:
            return q

        jaw_yaw = self._jaw_world_yaw(q[0])           # world yaw of the jaw axis
        # Object angle from minAreaRect: 0° = first axis of the enclosing rect.
        # Nearest face normal for a cube is at the nearest 90° from the jaw yaw.
        nearest = self._nearest_90_deg(obj_angle_deg)
        delta_deg = nearest - jaw_yaw
        # Normalise to (-180, 180]
        delta_deg = (delta_deg + 180.0) % 360.0 - 180.0
        delta_rad = math.radians(delta_deg)

        new_q = list(q)
        # jaw_yaw = degrees(θ1 - θ6) on the UR5e (tool-down, FK-verified), so
        # rotating the jaw axis by +delta means trimming wrist_3 by -delta.
        new_q[5] = self._normalize_angle(q[5] - delta_rad)
        if abs(new_q[5]) > self.JOINT_LIMIT:
            # Try ±90° neighbours (the cube has 4-fold symmetry)
            for alt in (delta_rad + math.pi / 2, delta_rad - math.pi / 2,
                        delta_rad + math.pi):
                candidate = self._normalize_angle(q[5] - alt)
                if abs(candidate) <= self.JOINT_LIMIT:
                    new_q[5] = candidate
                    break
            else:
                self.get_logger().warning(
                    f'  [{label}] yaw alignment Δθ6={math.degrees(delta_rad):.1f}° '
                    'out of range — picking without yaw correction'
                )
                return q

        self.get_logger().info(
            f'  [{label}] yaw alignment: jaw={jaw_yaw:.1f}°  '
            f'obj={obj_angle_deg:.1f}°  nearest={nearest:.1f}°  '
            f'Δ={delta_deg:.1f}°  θ6: {math.degrees(q[5]):.1f}° → {math.degrees(new_q[5]):.1f}°'
        )
        return new_q

    # ── Single pick-place cycle (no home visit, no drop mid-swing) ────────
    def _pick_and_place(self, entry: dict, zone_x: float, zone_y: float,
                        label: str) -> bool:
        """
        Pick the tracked object `entry` and place it at (zone_x, zone_y).
        Sequence: pre-pick → grasp → lift → traverse → place → retreat.

        Yaw alignment (§5.2): before each IK call θ5 is adjusted to align
        the jaw faces with the nearest cube face normal derived from the
        object's minAreaRect angle stored in `entry['angle_deg']`.  The
        TCP position is unchanged (Finding C: 0.00 mm shift).

        The lift leg breaks the long grasp→place move into two short legs, so
        MoveIt plans a direct high swing and can no longer fold the arm back
        through its home configuration.

        Returns True on success.
        """
        ox, oy = entry['x'], entry['y']
        obj_angle = entry.get('angle_deg', 0.0)
        cls       = entry['cls']
        pick_az      = _arm_z(PICK_WORLD_Z)           # 0.05
        pre_pick_az  = pick_az + LIFT_HEIGHT_M        # 0.15
        place_az     = _arm_z(self._place_wz)         # 0.10
        pre_place_az = _arm_z(self._pre_place_wz)     # 0.25

        self.get_logger().info(
            f'[{label}] pick=({ox:.3f},{oy:.3f}) arm_pick_z={pick_az:.3f}  '
            f'→ zone=({zone_x:.3f},{zone_y:.3f}) arm_place_z={place_az:.3f}  '
            f'obj_angle={obj_angle:.1f}°'
        )

        # 1. Open gripper
        self.move_gripper(GRIPPER_OPEN)

        # 2. Pre-pick: 10 cm above the object, with jaw yaw aligned
        q_pre = self._compute_ik(ox, oy, pre_pick_az)
        q_pre = self._align_ik_yaw(q_pre, obj_angle, cls, label)
        if not self.move_to_joints(q_pre):
            self.get_logger().error(f'  [{label}] Pre-pick FAILED')
            return False                              # object untouched

        # 3. Descend to grasp height (same yaw alignment as pre-pick)
        q_grasp = self._compute_ik(ox, oy, pick_az)
        q_grasp = self._align_ik_yaw(q_grasp, obj_angle, cls, label)
        if not self.move_to_joints(q_grasp):
            self.get_logger().error(f'  [{label}] Grasp descent FAILED')
            return False                              # still at safe height

        # 4. Close gripper
        self.move_gripper(GRIPPER_CLOSE)
        time.sleep(0.3)
        if not self._verify_grasp(label):
            self.get_logger().warning(f'  [{label}] EMPTY grasp — opening and aborting cycle')
            self.move_gripper(GRIPPER_OPEN)
            return False

        # 5. Lift straight up — keeps the transport path short and high
        if not self.move_to_joints(self._compute_ik(ox, oy, pre_pick_az)):
            self.get_logger().error(
                f'  [{label}] Lift FAILED — object left at grasp pose'
            )
            return False

        # 5b. Object still held after lift? (catches slip/drop)
        gap = self._jaw_gap()
        if gap is not None and not (GRASP_GAP_MIN_M <= gap <= GRASP_GAP_MAX_M):
            self.get_logger().warning(
                f'  [{label}] object lost on lift (gap={gap:.4f}) — setting down'
            )
            self._set_down_at(ox, oy, pick_az, pre_pick_az)
            return False

        # 6. Traverse to above the zone (plans FROM the lifted pose)
        q_traverse = self._compute_ik(zone_x, zone_y, pre_place_az)
        if q_traverse is None or not self.move_to_joints(q_traverse):
            self.get_logger().warning(
                f'  [{label}] Traverse FAILED — retrying from lifted pose'
            )
            if q_traverse is None or not self.move_to_joints(q_traverse):
                self.get_logger().error(
                    f'  [{label}] Traverse FAILED twice — setting object back down'
                )
                self._set_down_at(ox, oy, pick_az, pre_pick_az)
                return False

        # 7. Descend to place height
        if not self.move_to_joints(self._compute_ik(zone_x, zone_y, place_az)):
            self.get_logger().error(
                f'  [{label}] Place descent FAILED — releasing above zone'
            )
            self.move_gripper(GRIPPER_OPEN)           # drops into/near the zone
            self._retreat(zone_x, zone_y, pre_place_az)
            return False

        # 8. Release
        self.move_gripper(GRIPPER_OPEN)
        time.sleep(0.3)

        # 9. Retreat straight up — arm ready for the next pick
        self._retreat(zone_x, zone_y, pre_place_az)

        self.get_logger().info(f'  [{label}] ✓ Complete')
        return True

    # ── Recovery helpers ──────────────────────────────────────────────────
    def _set_down_at(self, x: float, y: float, down_az: float, up_az: float):
        """Best-effort recovery while holding an object: lower it back onto
        the table, release, and lift clear.  Never drops it mid-swing."""
        if self.move_to_joints(self._compute_ik(x, y, down_az)):
            self.move_gripper(GRIPPER_OPEN)
            time.sleep(0.2)
        else:
            self.move_gripper(GRIPPER_OPEN)           # unavoidable drop
        self.move_to_joints(self._compute_ik(x, y, up_az))

    def _retreat(self, x: float, y: float, up_az: float):
        """Best-effort vertical retreat after a place/release."""
        q = self._compute_ik(x, y, up_az)
        if q is not None:
            self.move_to_joints(q)

    # ── Main sorting mission (perception-driven, unknown object count) ────
    def run(self):
        if not self.preflight_check():
            self.get_logger().error('Preflight failed.')
            return

        self.get_logger().info('=' * 60)
        self.get_logger().info('   Vision Sorting Mission — START')
        self.get_logger().info('=' * 60)

        home = list(self.params['joints']['home_position'].values())

        # Phase 1: home (mission start only)
        self.get_logger().info('[Phase 1] Home pose...')
        if not self.move_to_joints(home):
            self._abort('Phase 1: Home')

        # Phase 2: gripper ready to grasp
        self.get_logger().info('[Phase 2] Gripper open — ready to sort')
        self.get_logger().info(
            f'[Phase 2] IK z: pick={_arm_z(PICK_WORLD_Z):.3f}  '
            f'place={_arm_z(self._place_wz):.3f}  '
            f'pre_place={_arm_z(self._pre_place_wz):.3f}'
        )
        self.move_gripper(GRIPPER_OPEN)

        # Phase 3: scan → sort nearest → re-scan loop.
        #   The object count is never assumed; the mission ends only when
        #   EMPTY_SCANS_TO_FINISH consecutive scans find nothing new.
        self.get_logger().info(
            '[Phase 3] Sorting loop (re-scanning after every object)...'
        )
        red_ok = blue_ok = 0
        empty_scans = 0
        scan_no = 0
        motion_fail_streak = 0   # consecutive failed cycles (circuit breaker)
        while empty_scans < EMPTY_SCANS_TO_FINISH:
            scan_no += 1
            # Wrist-camera scan: park over each pose, collect while settled,
            # merge across poses, then update the tracked-object registry.
            detections = self._scan_pass()
            candidates = self._update_registry(detections)

            self.get_logger().info(
                f'[Scan {scan_no}] {len(candidates)} sortable object(s) on Table 2'
            )
            if not candidates:
                empty_scans += 1
                if empty_scans < EMPTY_SCANS_TO_FINISH:
                    self.get_logger().info(
                        f'[Scan {scan_no}] Pick table empty — confirming with '
                        f'{EMPTY_SCANS_TO_FINISH - empty_scans} more scan(s)...'
                    )
                continue
            empty_scans = 0

            entry = candidates[0]                   # nearest reachable object
            cls = entry['cls']
            zone = self._next_zone(cls)
            if zone is None:
                self.get_logger().warning(
                    f'All {MAX_SLOTS} slots for {cls} are full — '
                    f'ignoring further {cls} detections.'
                )
                entry['blacklisted'] = True
                continue

            zx, zy = zone
            label = f'{cls.split("_")[0]}_{self._slot_index(cls)}'
            ok = self._pick_and_place(entry, zx, zy, label)

            if ok:
                motion_fail_streak = 0
                entry['done'] = True
                self._mark_zone_used(cls)
                if cls == 'red_cube':
                    red_ok += 1
                else:
                    blue_ok += 1
            else:
                motion_fail_streak += 1
                entry['attempts'] += 1
                ox, oy = entry['x'], entry['y']
                if entry['attempts'] >= MAX_ATTEMPTS_PER_OBJECT:
                    entry['blacklisted'] = True
                    self.get_logger().warning(
                        f'{cls} at ({ox:.2f},{oy:.2f}) failed '
                        f'{entry["attempts"]}x — blacklisting it.'
                    )
                else:
                    self.get_logger().warning(
                        f'{cls} at ({ox:.2f},{oy:.2f}) failed — retrying '
                        f'(attempt {entry["attempts"] + 1}/{MAX_ATTEMPTS_PER_OBJECT}).'
                    )
                if motion_fail_streak >= CONSECUTIVE_FAIL_ABORT:
                    self.get_logger().error(
                        f'{motion_fail_streak} consecutive sort cycles failed — '
                        'the MoveIt stack is not executing motions (duplicate '
                        'move_group? paused sim? controllers down?). Aborting '
                        'instead of blacklisting every object.'
                    )
                    break

        # Phase 4: home (mission end only)
        self.get_logger().info('[Phase 4] Returning home...')
        if not self.move_to_joints(home):
            self.get_logger().warning(
                'Could not return home — MoveIt stack unhealthy?'
            )

        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'   Mission Complete: {red_ok} red | {blue_ok} blue | '
            f'{red_ok + blue_ok} sorted'
        )
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = SortingController()
    try:
        node.run()
    except RuntimeError as e:
        node.get_logger().error(f'Mission aborted: {e}')
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
