import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_share = get_package_share_directory('robotic_4dof_arm')

    urdf_path = os.path.join(pkg_share, 'urdf', 'arm.urdf.xacro')
    rviz_config_path = os.path.join(pkg_share, 'config', 'display.rviz')

    # Expand xacro directly in Python — bypasses shell quoting issues entirely
    robot_description = xacro.process_file(urdf_path).toxml()

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),
        # Joint State Publisher GUI
        # ── Joint State Publisher GUI ──
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            parameters=[{
                'zeros': {
                    'shoulder_lift_joint': -1.5708,
                    'elbow_joint': 1.5708,
                    'wrist_1_joint': 1.5708,
                    'wrist_2_joint': 1.5708
                }
            }]
        ),
        # World Marker Publisher (renders warehouse table, obstacle, target box, and markers in RViz)
        Node(
            package='robotic_4dof_arm',
            executable='world_publisher.py',
            output='screen'
        ),
        # RViz2 with pre-configured display settings
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),
    ])
