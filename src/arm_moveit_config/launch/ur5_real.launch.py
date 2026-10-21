#!/usr/bin/env python3
"""
ur5_real.launch.py
------------------
Launch file for running the arm on REAL UR5e hardware.
Replaces Gazebo with the Universal Robots ROS 2 driver.

TODO items are marked with # TODO: — fill these in when hardware is available.

Usage:
    ros2 launch arm_moveit_config ur5_real.launch.py robot_ip:=<ROBOT_IP>
"""

import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_moveit = get_package_share_directory('arm_moveit_config')
    pkg_robot = get_package_share_directory('robotic_4dof_arm')

    # Config file paths
    urdf_path = os.path.join(pkg_robot, 'urdf', 'arm.urdf.xacro')
    srdf_path = os.path.join(pkg_moveit, 'config', 'arm.srdf')
    kinematics_yaml = os.path.join(pkg_moveit, 'config', 'kinematics_real.yaml')
    ompl_yaml = os.path.join(pkg_moveit, 'config', 'ompl_planning.yaml')
    controllers_yaml = os.path.join(pkg_moveit, 'config', 'moveit_controllers_real.yaml')
    joint_limits_yaml = os.path.join(pkg_moveit, 'config', 'joint_limits.yaml')
    rviz_config_path = os.path.join(pkg_moveit, 'config', 'moveit.rviz')

    # Load descriptions
    robot_description = xacro.process_file(urdf_path).toxml()
    robot_description_semantic = open(srdf_path).read()

    with open(kinematics_yaml, 'r') as f:
        kinematics = yaml.safe_load(f)
    with open(ompl_yaml, 'r') as f:
        ompl_config = yaml.safe_load(f)
    with open(controllers_yaml, 'r') as f:
        controllers = yaml.safe_load(f)
    with open(joint_limits_yaml, 'r') as f:
        joint_limits = yaml.safe_load(f)

    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_config,
    }

    # Launch arguments
    robot_ip = LaunchConfiguration('robot_ip')  # noqa: F841 — used in hardware driver block
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([

        # ── Launch Arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            'robot_ip',
            default_value='192.168.1.100',   # TODO: set your UR5e's actual IP
            description='IP address of the real UR5e robot'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to launch RViz2'
        ),

        # ── UR Robot Driver ───────────────────────────────────────────────────
        # TODO: Uncomment when ur_robot_driver is installed
        #       (sudo apt install ros-<distro>-ur).  Phase 0 baselines the
        #       model on ur_type 'ur5e' — keep this in sync with the sim.
        #       Also run ur_calibration first and save to ur5e_calibration.yaml
        #
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(
        #             get_package_share_directory('ur_robot_driver'),
        #             'launch', 'ur_control.launch.py'
        #         )
        #     ),
        #     launch_arguments={
        #         'ur_type':                'ur5e',
        #         'robot_ip':               robot_ip,
        #         'kinematics_params_file': calibration_yaml,
        #         'use_fake_hardware':      'false',
        #         'launch_rviz':            'false',
        #     }.items()
        # ),

        # ── Robot State Publisher (real hardware — no Gazebo) ─────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time':      False,   # REAL TIME on hardware
            }],
            output='screen'
        ),

        # ── MoveIt MoveGroup ──────────────────────────────────────────────────
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            parameters=[
                {'robot_description':            robot_description},
                {'robot_description_semantic':   robot_description_semantic},
                {'robot_description_kinematics': kinematics},
                {'robot_description_planning':   joint_limits},
                planning_pipelines,
                controllers,
                {'use_sim_time': False},      # KEY: no sim time on hardware
            ],
            output='screen'
        ),

        # ── Optional RViz2 ────────────────────────────────────────────────────
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_path],
            parameters=[
                {'robot_description':            robot_description},
                {'robot_description_semantic':   robot_description_semantic},
                {'robot_description_kinematics': kinematics},
                planning_pipelines,
                {'use_sim_time': False},
            ],
            condition=IfCondition(use_rviz),
            output='screen'
        ),
    ])
