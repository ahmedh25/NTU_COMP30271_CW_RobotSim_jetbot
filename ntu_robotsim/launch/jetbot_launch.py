"""
JetBot Real-Hardware Launch File  (RealSense D435 version)
-----------------------------------------------------------
Replaces the CSI camera + HC-SR04 ultrasonic setup with the
Intel RealSense D435 depth camera.

Startup sequence:
  t=0s   RealSense camera driver  (publishes /camera/color/image_raw
                                   and /camera/depth/image_rect_raw)
  t=3s   YOLO detection           (subscribes /camera/color/image_raw
                                   → publishes /detections)
  t=3s   Motor driver             (subscribes /cmd_vel → drives motors)
  t=6s   Wall follower            (subscribes /camera/depth/image_rect_raw
                                   + /detections → /cmd_vel)

Usage:
  ros2 launch ntu_robotsim jetbot_launch.py

Optional overrides:
  ros2 launch ntu_robotsim jetbot_launch.py \
      yolo_model:=/path/to/best_nano.pt \
      enable_landmark_logger:=true
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    ExecuteProcess,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Paths ─────────────────────────────────────────────────────────────────────
LAUNCH_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(
    os.path.dirname(LAUNCH_DIR), 'models', 'custom_models'
)
DEFAULT_MODEL = os.path.join(MODELS_DIR, 'best_nano.pt')


def generate_launch_description() -> LaunchDescription:

    # ── Declare arguments ─────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'yolo_model', default_value=DEFAULT_MODEL,
            description='Path to YOLO .pt weights file'),
        DeclareLaunchArgument(
            'yolo_threshold', default_value='0.5',
            description='YOLO confidence threshold'),
        DeclareLaunchArgument(
            'yolo_device', default_value='cuda:0',
            description='Torch device for YOLO (cuda:0 or cpu)'),
        DeclareLaunchArgument(
            'linear_speed', default_value='0.20',
            description='Base forward speed (mapped to motor 0-1)'),
        DeclareLaunchArgument(
            'wall_follow_dist', default_value='0.50',
            description='Target distance to right wall (metres)'),
        DeclareLaunchArgument(
            'front_stop_dist', default_value='0.40',
            description='Front distance at which robot turns (metres)'),
        DeclareLaunchArgument(
            'enable_landmark_logger', default_value='false',
            description='Set true to also start the landmark CSV logger'),
        DeclareLaunchArgument(
            'camera_width', default_value='640',
            description='RealSense colour/depth stream width'),
        DeclareLaunchArgument(
            'camera_height', default_value='480',
            description='RealSense colour/depth stream height'),
        DeclareLaunchArgument(
            'camera_fps', default_value='30',
            description='RealSense stream frame rate'),
    ]

    # ── 1. RealSense D435 camera driver ───────────────────────────────────────
    #    Publishes:
    #      /camera/color/image_raw          (RGB, for YOLO)
    #      /camera/depth/image_rect_raw     (16UC1 mm, for wall following)
    #      /camera/color/camera_info
    #      /camera/depth/camera_info
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        namespace='camera',
        output='screen',
        parameters=[{
            'color_width':  LaunchConfiguration('camera_width'),
            'color_height': LaunchConfiguration('camera_height'),
            'color_fps':    LaunchConfiguration('camera_fps'),
            'depth_width':  LaunchConfiguration('camera_width'),
            'depth_height': LaunchConfiguration('camera_height'),
            'depth_fps':    LaunchConfiguration('camera_fps'),
            'enable_color': True,
            'enable_depth': True,
            'align_depth':  True,   # align depth to colour frame
        }],
    )

    # ── 2. YOLO detection node ────────────────────────────────────────────────
    #    Input:  /camera/color/image_raw
    #    Output: /detections (yolo_msgs/DetectionArray)
    yolo_node = TimerAction(
    period=3.0,
    actions=[
        ExecuteProcess(
            cmd=[
                'python3',
                os.path.join(LAUNCH_DIR, 'jetbot_yolo_node.py'),
            ],
            output='screen',
            )
        ],
    )

    # ── 3. Motor driver ───────────────────────────────────────────────────────
    #    Subscribes: /cmd_vel → drives JetBot motors via jetbot library
    motor_driver = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_motor_driver.py'),
                ],
                output='screen',
            )
        ],
    )

    # ── 4. Wall follower (D435 version) ───────────────────────────────────────
    #    Subscribes: /camera/depth/image_rect_raw + /detections
    #    Publishes:  /cmd_vel
    wall_follower = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_wall_follower_d435.py'),
                    '--ros-args',
                    '-p', ['linear_speed:=',    LaunchConfiguration('linear_speed')],
                    '-p', ['wall_follow_dist:=', LaunchConfiguration('wall_follow_dist')],
                    '-p', ['front_stop_dist:=',  LaunchConfiguration('front_stop_dist')],
                ],
                output='screen',
            )
        ],
    )

    # ── 5. Landmark CSV logger (optional) ─────────────────────────────────────
    landmark_logger = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_landmark_csv_logger.py'),
                ],
                output='screen',
            )
        ],
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    ld = LaunchDescription(args + [
        realsense_node,
        yolo_node,
        motor_driver,
        wall_follower,
    ])

    ld.add_action(
        GroupAction(
            condition=IfCondition(LaunchConfiguration('enable_landmark_logger')),
            actions=[landmark_logger],
        )
    )

    return ld