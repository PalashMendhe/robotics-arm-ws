#!/usr/bin/env python3
"""
shape_detector.py  —  B1 Classical Shape Detection Node
--------------------------------------------------------
Subscribes to the WRIST D415 colour + depth streams (/d415_wrist/*, bridged
in vision_sim.launch.py incl. camera_info — E1), segments all objects on
Table 2 by HSV COLOUR (red → red_cube, blue → blue_cylinder; red hue wraps
so two bands are used), forms one closed region per object via
open→close→connectedComponents, gates by adaptive area, border-touch and
depth-validity, back-projects to 3-D world coordinates via TF2, and publishes
vision_msgs/Detection2DArray.

Circularity + minAreaRect angle are LOGGED as diagnostics only (angle steers
the grasp-yaw correction); they never decide the class.

Topics
------
  SUB  /d415_wrist/image          sensor_msgs/Image  (bgr8 colour)
  SUB  /d415_wrist/depth_image    sensor_msgs/Image  (32FC1 depth, metres)
  SUB  /d415_wrist/camera_info    sensor_msgs/CameraInfo  (intrinsics)
  PUB  /vision/detections   vision_msgs/Detection2DArray
  PUB  /vision/debug_image  sensor_msgs/Image   (annotated colour frame)
"""

import math
import os

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

import message_filters


class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_detector_subscriber')

        # ── Load tuning parameters from camera_params.yaml ────────────────
        params_path = os.path.join(
            get_package_share_directory('arm_moveit_config'),
            'config', 'camera_params.yaml'
        )
        with open(params_path, 'r') as f:
            cfg = yaml.safe_load(f)

        det = cfg['detection']
        self.canny_low            = det['canny_low']
        self.canny_high           = det['canny_high']
        self.circularity_threshold = det['circularity_threshold']
        self.depth_scale          = det['depth_scale']   # 1.0 for Gazebo float32

        # ── Colour segmentation (class decision, §3.2 item 1) ─────────────
        col = det['colour']
        self.red_lo1 = np.array(col['red_low'],   dtype=np.uint8)
        self.red_hi1 = np.array(col['red_high'],  dtype=np.uint8)
        self.red_lo2 = np.array(col['red_low2'],  dtype=np.uint8)
        self.red_hi2 = np.array(col['red_high2'], dtype=np.uint8)
        self.blue_lo = np.array(col['blue_low'],  dtype=np.uint8)
        self.blue_hi = np.array(col['blue_high'], dtype=np.uint8)

        # ── Region formation + adaptive area gate (§3.2 items 2-3) ────────
        ko = int(det['morph_open_k'])
        kc = int(det['morph_close_k'])
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
        self.object_size_m   = float(det['object_size_m'])
        self.nominal_range_m = float(det['nominal_range_m'])
        self.area_scale_min  = float(det['area_scale_min'])
        self.area_scale_max  = float(det['area_scale_max'])
        self.min_area_px     = float(det['min_area_px'])

        # ── Silhouette trust + depth sampling (§3.2 items 4-5) ───────────
        self.border_margin_px     = int(det['border_margin_px'])
        self.depth_valid_frac_min = float(det['depth_valid_frac_min'])

        self.depth_min = cfg['depth']['min_m']
        self.depth_max = cfg['depth']['max_m']

        # ── Primary camera: eye-in-hand (wrist) ───────────────────────────
        # Topics + optical frame from the wrist_camera block.  Look-then-move:
        # the arm is parked at a scan pose while frames are processed, so the
        # latest TF is consistent with the images.
        wrist = cfg['wrist_camera']
        prefix = wrist.get('topic_prefix', '/d415_wrist')
        self.optical_frame = wrist['optical_frame_id']
        topic_image  = f'{prefix}/image'
        topic_depth  = f'{prefix}/depth_image'
        topic_info   = f'{prefix}/camera_info'
        # Depth/world lookup frame: true Gazebo world coordinates
        self.world_frame = cfg['gazebo_world_frame']

        # Camera intrinsics — will be populated on first CameraInfo message
        self.fx = self.fy = self.cx = self.cy = None
        self._camera_info_ready = False

        # ── TF2 buffer + listener ──────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── cv_bridge + publisher ──────────────────────────────────────────
        self.bridge = CvBridge()

        self.pub_detections = self.create_publisher(
            Detection2DArray, '/vision/detections', 10
        )
        self.pub_debug = self.create_publisher(
            Image, '/vision/debug_image', 10
        )

        # ── Camera info subscriber (one-shot to get intrinsics) ────────────
        self.sub_info = self.create_subscription(
            CameraInfo, topic_info, self._camera_info_cb, 1
        )

        # ── Synchronised colour + depth subscribers ────────────────────────
        sub_color = message_filters.Subscriber(self, Image, topic_image)
        sub_depth = message_filters.Subscriber(self, Image, topic_depth)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [sub_color, sub_depth], queue_size=10, slop=0.05
        )
        self._sync.registerCallback(self.image_callback)

        self.get_logger().info('ShapeDetector ready — waiting for camera feed...')

    # ── Camera intrinsics (populated once) ────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo):
        if not self._camera_info_ready:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self._camera_info_ready = True
            self.get_logger().info(
                f'CameraInfo received: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}'
            )

    # ── Diagnostic edge map (debug overlay only) ───────────────────────────
    def preprocess_image(self, cv_image: np.ndarray) -> np.ndarray:
        """Canny edge map — drawn in the debug image, never classified on."""
        gray    = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges

    # ── Colour segmentation → one mask per class (§3.2 item 1) ────────────
    def segment_by_colour(self, cv_image: np.ndarray) -> dict:
        """
        Returns {'red_cube': mask, 'blue_cylinder': mask} as uint8 0/255 masks.

        Colour decides the class; circularity is a diagnostic only.  Red hue
        wraps around 0/180, hence the two red bands.
        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, self.red_lo1, self.red_hi1),
            cv2.inRange(hsv, self.red_lo2, self.red_hi2),
        )
        blue = cv2.inRange(hsv, self.blue_lo, self.blue_hi)
        return {'red_cube': red, 'blue_cylinder': blue}

    # ── Adaptive area gate (§3.2 item 3) ──────────────────────────────────
    def expected_px(self, depth_m: float) -> float:
        """Object silhouette size in pixels at the given lens→object range."""
        if depth_m <= 0 or not self.fx:
            return 0.0
        return self.object_size_m * self.fx / depth_m

    def area_gate(self, expected_px_: float, loose: bool = False) -> tuple:
        """
        (area_lo, area_hi) in px².  `loose` widens the gate 4x/4x (16x in
        area) — used for the depth-free pre-filter so a stale
        `nominal_range_m` can never silently drop real objects; the strict
        gate is applied again once the measured depth is known.
        """
        lo_scale = self.area_scale_min / (4.0 if loose else 1.0)
        hi_scale = self.area_scale_max * (4.0 if loose else 1.0)
        return (max(self.min_area_px, lo_scale * expected_px_ ** 2),
                hi_scale * expected_px_ ** 2)

    # ── Regions from one colour mask (§3.2 items 2-4) ─────────────────────
    def components(self, mask: np.ndarray, class_id: str,
                   expected_px_: float) -> list:
        """
        Morph-open/close a colour mask, then return one entry per connected
        component that passes the (loose) adaptive area gate and lies fully
        inside the frame — a clipped silhouette is never trustworthy.

        Returns a list of dicts:
            {area, cx_px, cy_px, w, h, angle, circularity, mask}
        """
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel_close)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        h_img, w_img = mask.shape[:2]
        b = self.border_margin_px
        area_lo, area_hi = self.area_gate(expected_px_, loose=True)

        out = []
        for i in range(1, n):                     # 0 = background
            x    = int(stats[i, cv2.CC_STAT_LEFT])
            y    = int(stats[i, cv2.CC_STAT_TOP])
            w    = int(stats[i, cv2.CC_STAT_WIDTH])
            h    = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if not (area_lo <= area <= area_hi):
                self.get_logger().debug(
                    f'{class_id}: component rejected by area pre-gate '
                    f'({area} px2 outside [{area_lo:.0f}, {area_hi:.0f}])'
                )
                continue

            if x <= b or y <= b or x + w >= w_img - b or y + h >= h_img - b:
                self.get_logger().warning(
                    f'{class_id}: component touches the frame border '
                    f'(bbox=({x},{y},{w},{h}) in {w_img}x{h_img}) — rejected'
                )
                continue

            comp = (labels == i).astype(np.uint8)
            mm = cv2.moments(comp, True)          # binaryImage=True
            if mm['m00'] <= 0:
                continue

            # Circularity (diagnostic) + object yaw for grasp alignment:
            # Detection2D has no angle field, so it rides in bbox.center.theta.
            contour = cv2.findContours(
                comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0][0]
            _, _, angle = cv2.minAreaRect(contour)
            perim = cv2.arcLength(contour, True)
            circ = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0

            out.append({
                'area': area,
                'cx_px': mm['m10'] / mm['m00'],
                'cy_px': mm['m01'] / mm['m00'],
                'w': w, 'h': h, 'angle': angle,
                'circularity': circ, 'mask': comp,
            })
        return out

    # ── Depth lookup: full component mask median with validity fraction ───
    def get_valid_depth(self, depth_image: np.ndarray,
                        cx_px: int, cy_px: int,
                        mask: np.ndarray | None = None):
        """
        Returns depth in metres, or None if out-of-range / too few valid pixels.

        When `mask` (the full component binary image) is provided the median
        is computed over ALL in-mask pixels — more robust than a 3×3 centroid
        patch for oblique silhouettes where the top-face centre may be off the
        object or in shadow.  depth_valid_frac_min gate ensures that ≥ 30 %
        of the component has valid depth, flagging partial-view artefacts.

        Falls back to a 3×3 centroid patch when no mask is given (legacy /
        external call-sites).
        """
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                return None
            depths = depth_image[ys, xs].astype(np.float32)
            valid  = depths[(depths > 0) & np.isfinite(depths)]
            frac   = valid.size / ys.size
            if frac < self.depth_valid_frac_min:
                self.get_logger().debug(
                    f'depth_valid_frac={frac:.2f} < {self.depth_valid_frac_min} — rejected'
                )
                return None
        else:
            h, w = depth_image.shape[:2]
            x0, x1 = max(0, cx_px - 1), min(w, cx_px + 2)
            y0, y1 = max(0, cy_px - 1), min(h, cy_px + 2)
            patch = depth_image[y0:y1, x0:x1].astype(np.float32)
            valid  = patch[(patch > 0) & np.isfinite(patch)]

        if valid.size == 0:
            return None
        depth_m = float(np.median(valid)) * self.depth_scale
        if not (self.depth_min <= depth_m <= self.depth_max):
            return None
        return depth_m

    # ── Back-project pixel → 3-D camera frame ─────────────────────────────
    def back_project_to_3d(self, cx_px: int, cy_px: int, depth_m: float) -> tuple:
        X_c = (cx_px - self.cx) * depth_m / self.fx
        Y_c = (cy_px - self.cy) * depth_m / self.fy
        Z_c = depth_m
        return (X_c, Y_c, Z_c)

    # ── Transform camera frame → world frame via TF2 ──────────────────────
    def transform_to_world(self, X_c: float, Y_c: float, Z_c: float):
        """
        Uses tf2 to transform a point from the wrist camera's optical frame
        to `self.world_frame` (default 'world_gazebo' = true Gazebo world
        coordinates, matching PICK_WORLD_Z / sort-zone semantics).

        NOTE: the URDF root link is *named* 'world' but sits at the arm base,
        0.95 m above the Gazebo origin, so looking up 'world' directly yields
        z values 0.95 m too low.  camera_tf_broadcaster publishes the
        correcting static TF world_gazebo -> world from camera_params.yaml.

        Time() = latest transform: valid because the arm is parked at a scan
        pose while frames are processed (look-then-move).
        Returns (X_w, Y_w, Z_w) or None if the transform is unavailable.
        """
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.world_frame, self.optical_frame, Time()
            )
        except Exception as e:
            self.get_logger().warning(f'TF lookup failed: {e}')
            return None

        t = tf.transform.translation
        r = tf.transform.rotation

        # Quaternion → rotation matrix
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
            [  2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        p_c = np.array([X_c, Y_c, Z_c])
        tr  = np.array([t.x, t.y, t.z])
        p_w = R @ p_c + tr
        return (float(p_w[0]), float(p_w[1]), float(p_w[2]))

    # ── Build Detection2DArray message ─────────────────────────────────────
    def detection2darray(self, detections: list, header) -> Detection2DArray:
        """
        detections: list of (cx_px, cy_px, w, h, angle, class_id, circularity, X_w, Y_w, Z_w)

        The object's minAreaRect `angle` (degrees) is stored in
        bbox.center.theta — the sorting controller reads it back for yaw
        alignment (§3.2-6, §5.2): θ5 += nearest_90°(ψ_obj) − jaw_yaw.
        """
        array_msg = Detection2DArray()
        array_msg.header = header

        for (cx_px, cy_px, bw, bh, angle, class_id, circularity, X_w, Y_w, Z_w) in detections:
            det = Detection2D()
            det.header = header

            # 2-D bounding box (image space); theta carries the object yaw
            det.bbox.center.position.x = float(cx_px)
            det.bbox.center.position.y = float(cy_px)
            det.bbox.center.theta      = float(angle)   # minAreaRect angle (deg)
            det.bbox.size_x = float(bw)
            det.bbox.size_y = float(bh)

            # Hypothesis: class + confidence + 3-D world pose
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_id
            hyp.hypothesis.score    = float(min(circularity, 1.0))
            hyp.pose.pose.position.x = X_w
            hyp.pose.pose.position.y = Y_w
            hyp.pose.pose.position.z = Z_w

            det.results.append(hyp)
            array_msg.detections.append(det)

        return array_msg

    # ── Main callback (colour + depth synchronised) ────────────────────────
    def image_callback(self, color_msg: Image, depth_msg: Image):
        """
        Colour-segmentation pipeline (§3.2):
          1. Segment by HSV colour → one mask per class.
          2. Morph-open/close + connectedComponents → one detection per blob.
          3. Area gate (adaptive, loose pre-filter).
          4. Reject border-touching components.
          5. Full-mask depth median with validity-fraction guard.
          6. Back-project → world frame via TF2.
          7. Publish Detection2DArray + annotated debug image.
        """
        if not self._camera_info_ready:
            return  # wait until intrinsics are known

        cv_image    = self.bridge.imgmsg_to_cv2(color_msg,  desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg,  desired_encoding='passthrough')

        # ── Diagnostic edge overlay (debug only, never used for classification) ──
        debug_img = cv_image.copy()
        edges = self.preprocess_image(cv_image)
        debug_img[edges > 0] = [100, 100, 100]  # grey edge overlay

        # ── Colour segmentation → per-class component lists ────────────────
        masks = self.segment_by_colour(cv_image)

        # Nominal expected pixel size for the loose pre-filter gate
        nom_exp = self.expected_px(self.nominal_range_m)

        detections = []

        for class_id, colour_mask in masks.items():
            # BGR debug tint for this class
            draw_colour = (0, 0, 200) if class_id == 'red_cube' else (200, 60, 0)

            comps = self.components(colour_mask, class_id, nom_exp)
            for comp in comps:
                cx_px = int(round(comp['cx_px']))
                cy_px = int(round(comp['cy_px']))
                comp_mask = comp['mask']

                # ── Depth: full-mask median with validity-fraction guard ──
                depth_m = self.get_valid_depth(
                    depth_image, cx_px, cy_px, mask=comp_mask
                )
                if depth_m is None:
                    self.get_logger().debug(
                        f'{class_id}: no valid depth at ({cx_px},{cy_px})'
                    )
                    continue

                # ── Strict adaptive area gate (now that depth is known) ──
                exp_px = self.expected_px(depth_m)
                area_lo, area_hi = self.area_gate(exp_px, loose=False)
                if not (area_lo <= comp['area'] <= area_hi):
                    self.get_logger().debug(
                        f'{class_id}: area {comp["area"]} outside strict gate '
                        f'[{area_lo:.0f}, {area_hi:.0f}] at depth {depth_m:.3f} m'
                    )
                    continue

                # ── Back-project → world frame ───────────────────────────
                world = self.transform_to_world(
                    *self.back_project_to_3d(cx_px, cy_px, depth_m)
                )
                if world is None:
                    continue
                X_w, Y_w, Z_w = world

                angle = comp['angle']
                circ  = comp['circularity']
                bw, bh = comp['w'], comp['h']

                detections.append((cx_px, cy_px, bw, bh, angle,
                                   class_id, circ, X_w, Y_w, Z_w))

                # ── Debug drawing ─────────────────────────────────────────
                # Outline the component
                contour_pts, _ = cv2.findContours(
                    comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(debug_img, contour_pts, -1, draw_colour, 2)
                cv2.circle(debug_img, (cx_px, cy_px), 4, (0, 255, 0), -1)
                label = f'{class_id}  c={circ:.2f}  a={angle:.1f}°'
                cv2.putText(debug_img, label, (cx_px - 50, cy_px - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, draw_colour, 1)
                coord_label = f'({X_w:.2f},{Y_w:.2f},{Z_w:.2f})'
                cv2.putText(debug_img, coord_label, (cx_px - 50, cy_px + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 0), 1)

                self.get_logger().info(
                    f'[{class_id}] world=({X_w:.3f}, {Y_w:.3f}, {Z_w:.3f})  '
                    f'circ={circ:.3f}  angle={angle:.1f}°  depth={depth_m:.3f}m'
                )

        # ── Publish results ────────────────────────────────────────────────
        det_array = self.detection2darray(detections, color_msg.header)
        self.pub_detections.publish(det_array)
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        )


def main(args=None):
    rclpy.init(args=args)
    shape_detector_node = ShapeDetector()
    rclpy.spin(shape_detector_node)
    shape_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
