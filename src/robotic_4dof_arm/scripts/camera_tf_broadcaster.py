#!/usr/bin/env python3
"""
camera_tf_broadcaster.py
------------------------
Publishes the static TF chain:

    world → d415_camera_link        (mechanical mount pose, from YAML)
          → d415_color_optical_frame (standard camera→optical rotation)

The optical-frame step is REQUIRED: depth/colour pixels are back-projected in
the optical convention (Z forward, X right, Y down), while the Gazebo sensor
frame is X forward / Y left / Z up.  Publishing only the mount pose — as this
node used to do — applied the wrong rotation, so every back-projected world
(x, y, z) was systematically offset and the arm picked at the same wrong spot.

Reads the camera pose from camera_params.yaml so the transform can be adjusted
without modifying this script.
"""

import os
import math
import yaml

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert RPY (radians) to quaternion (x, y, z, w).  q = qz(yaw)*qy(pitch)*qx(roll)."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


# Standard ROS body→optical-frame rotation for an X-forward camera body:
#   optical Z = body X (forward), optical X = -body Y (right), optical Y = -body Z (down)
LINK_TO_OPTICAL_RPY = (-math.pi / 2.0, 0.0, -math.pi / 2.0)


class CameraTfBroadcaster(Node):
    def __init__(self):
        super().__init__('camera_tf_broadcaster')

        params_path = os.path.join(
            get_package_share_directory('arm_moveit_config'),
            'config', 'camera_params.yaml'
        )
        with open(params_path, 'r') as f:
            cfg = yaml.safe_load(f)

        broadcaster = StaticTransformBroadcaster(self)

        oqx, oqy, oqz, oqw = euler_to_quaternion(*LINK_TO_OPTICAL_RPY)

        # ── ROOT: Gazebo world -> URDF world (arm mount) ──────────────────
        # The URDF root link is named 'world' but lives at the arm spawn
        # position, 0.95 m above the Gazebo world origin.  Publishing the
        # offset lets every consumer work in true Gazebo coordinates by
        # looking up 'world_gazebo' (shape_detector does exactly that).
        gz_frame = cfg['gazebo_world_frame']
        ox, oy, oz = cfg['world_offset']
        tf_root = TransformStamped()
        tf_root.header.stamp = self.get_clock().now().to_msg()
        tf_root.header.frame_id = gz_frame
        tf_root.child_frame_id = 'world'
        tf_root.transform.translation.x = ox
        tf_root.transform.translation.y = oy
        tf_root.transform.translation.z = oz
        tf_root.transform.rotation.w = 1.0
        broadcaster.sendTransform(tf_root)
        self.get_logger().info(
            f'Published static TF: {gz_frame} -> world at '
            f'xyz=({ox}, {oy}, {oz}) (arm mount offset)'
        )

        # ── PRIMARY: wrist (eye-in-hand) camera ────────────────────────────
        # The mount TF (gripper_base_link -> d415_wrist_link) is published
        # dynamically by robot_state_publisher from the URDF fixed joint —
        # this node only adds the static link -> optical-frame rotation.
        wrist = cfg['wrist_camera']
        oqx, oqy, oqz, oqw = euler_to_quaternion(*LINK_TO_OPTICAL_RPY)
        tf_optical = TransformStamped()
        tf_optical.header.stamp = self.get_clock().now().to_msg()
        tf_optical.header.frame_id = wrist['frame_id']
        tf_optical.child_frame_id = wrist['optical_frame_id']
        tf_optical.transform.rotation.x = oqx
        tf_optical.transform.rotation.y = oqy
        tf_optical.transform.rotation.z = oqz
        tf_optical.transform.rotation.w = oqw
        broadcaster.sendTransform(tf_optical)
        self.get_logger().info(
            f'Published static TF: {wrist["frame_id"]} -> '
            f'{wrist["optical_frame_id"]} (camera->optical rotation; '
            f'mount TF comes from robot_state_publisher)'
        )

        # ── PARKED: static pole camera (census fallback) ───────────────────
        # Its mount is world-fixed, so the full static chain is correct.
        cfg = cfg['camera']
        xyz = cfg['xyz']
        rpy = cfg['rpy']
        parent = cfg['parent_frame']
        link = cfg['frame_id']
        child = cfg['optical_frame_id']

        # ── TF 1: world → camera_link (mechanical mount pose) ─────────────
        qx, qy, qz, qw = euler_to_quaternion(*rpy)
        tf_mount = TransformStamped()
        tf_mount.header.stamp = self.get_clock().now().to_msg()
        tf_mount.header.frame_id = parent
        tf_mount.child_frame_id = link
        tf_mount.transform.translation.x = xyz[0]
        tf_mount.transform.translation.y = xyz[1]
        tf_mount.transform.translation.z = xyz[2]
        tf_mount.transform.rotation.x = qx
        tf_mount.transform.rotation.y = qy
        tf_mount.transform.rotation.z = qz
        tf_mount.transform.rotation.w = qw
        broadcaster.sendTransform(tf_mount)

        # ── TF 2: camera_link → optical frame (standard convention) ───────
        tf_optical2 = TransformStamped()
        tf_optical2.header.stamp = self.get_clock().now().to_msg()
        tf_optical2.header.frame_id = link
        tf_optical2.child_frame_id = child
        tf_optical2.transform.rotation.x = oqx
        tf_optical2.transform.rotation.y = oqy
        tf_optical2.transform.rotation.z = oqz
        tf_optical2.transform.rotation.w = oqw
        broadcaster.sendTransform(tf_optical2)

        self.get_logger().info(
            f'Published static TF chain: {parent} → {link} '
            f'at xyz={xyz}, rpy={rpy} (parked census camera)'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraTfBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

