"""
JetBot Real-Hardware Launch File  (ROS2 Eloquent / Intel RealSense D435)
-------------------------------------------------------------------------
Replaces yolo_unified.launch.py for the physical Waveshare JetBot.
No Gazebo, no Nav2, no OctoMap — just the nodes needed on real hardware.

KEY DIFFERENCES from the original file (now fixed):
  - Camera   : Uses realsense2_camera (Intel D435) NOT v4l2_camera (CSI cam).
               The module JetBot has a RealSense D435, not a CSI/USB camera.
  - Sensing  : Uses jetbot_realsense_distance.py (reads RealSense depth)
               NOT jetbot_ultrasonic_driver.py (HC-SR04 sensors not fitted).
  - YOLO     : Uses jetbot_yolo_detector.py (ultralytics direct)
               NOT yolo_ros package (not available for Eloquent).
  - Topics   : Camera publishes /camera/color/image_raw (RealSense namespace).

Startup sequence:
  t=0s   RealSense camera driver  (/camera/color/image_raw, /camera/depth/…)
  t=3s   YOLO detector            (subscribes /camera/color/image_raw → /detections)
  t=3s   Motor driver             (subscribes /cmd_vel → drives motors)
  t=4s   RealSense distance node  (subscribes depth → /distance/front, /distance/right)
  t=6s   Wall follower            (subscribes /distance/* + /detections → /cmd_vel)
  t=6s   Landmark logger          (optional, subscribes /detections + /odom)

Usage:
  ros2 launch ntu_robotsim jetbot_launch.py

Optional overrides:
  ros2 launch ntu_robotsim jetbot_launch.py \\
      yolo_model:=/path/to/best_nano.pt \\
      yolo_threshold:=0.45 \\
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

# ── Paths ──────────────────────────────────────────────────────────────────────
LAUNCH_DIR    = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR    = os.path.join(os.path.dirname(LAUNCH_DIR), 'models', 'custom_models')
DEFAULT_MODEL = os.path.join(MODELS_DIR, 'best_nano.pt')


def generate_launch_description():

    # ── Declare arguments ───────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'yolo_model', default_value=DEFAULT_MODEL,
            description='Path to YOLO .pt weights file'),
        DeclareLaunchArgument(
            'yolo_threshold', default_value='0.5',
            description='YOLO confidence threshold (0.0–1.0)'),
        DeclareLaunchArgument(
            'yolo_device', default_value='cuda:0',
            description='Torch device: cuda:0 or cpu'),
        DeclareLaunchArgument(
            'linear_speed', default_value='0.20',
            description='Base forward speed (0–1 motor scale)'),
        DeclareLaunchArgument(
            'wall_follow_dist', default_value='0.50',
            description='Target distance to right wall (metres)'),
        DeclareLaunchArgument(
            'front_stop_dist', default_value='0.40',
            description='Front distance at which the robot turns (metres)'),
        DeclareLaunchArgument(
            'camera_width', default_value='640',
            description='RealSense colour stream width (pixels)'),
        DeclareLaunchArgument(
            'camera_height', default_value='480',
            description='RealSense colour stream height (pixels)'),
        DeclareLaunchArgument(
            'camera_fps', default_value='15',
            description='RealSense colour/depth frame rate (fps) — keep low on Nano'),
        DeclareLaunchArgument(
            'enable_landmark_logger', default_value='false',
            description='Set true to also start the landmark CSV logger'),
    ]

    # ── 1. Intel RealSense D435 camera driver ──────────────────────────────
    #
    #  Publishes (among others):
    #    /camera/color/image_raw          (sensor_msgs/Image)
    #    /camera/depth/image_rect_raw     (sensor_msgs/Image, 16UC1 mm values)
    #    /camera/depth/color/points       (sensor_msgs/PointCloud2)
    #
    #  Package: realsense2_camera  (ros2-legacy branch)
    #  Install: see JETBOT_SETUP.md
    #
    camera_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense_camera',
        output='screen',
        parameters=[{
            'enable_color':     True,
            'enable_depth':     True,
            'enable_infra1':    False,
            'enable_infra2':    False,
            'color_width':      LaunchConfiguration('camera_width'),
            'color_height':     LaunchConfiguration('camera_height'),
            'color_fps':        LaunchConfiguration('camera_fps'),
            'depth_width':      LaunchConfiguration('camera_width'),
            'depth_height':     LaunchConfiguration('camera_height'),
            'depth_fps':        LaunchConfiguration('camera_fps'),
            'align_depth':      False,
        }],
    )

    # ── 2. YOLO detector (t=3 s) ───────────────────────────────────────────
    #
    #  Uses ultralytics directly — no dependency on yolo_ros.
    #  Input:  /camera/color/image_raw
    #  Output: /detections  (yolo_msgs/DetectionArray)
    #
    yolo_node = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_yolo_detector.py'),
                    '--ros-args',
                    '-p', ['model_path:=',  LaunchConfiguration('yolo_model')],
                    '-p', ['threshold:=',   LaunchConfiguration('yolo_threshold')],
                    '-p', ['device:=',      LaunchConfiguration('yolo_device')],
                    '-p', 'image_topic:=/camera/color/image_raw',
                ],
                output='screen',
            )
        ],
    )

    # ── 3. Motor driver (t=3 s) ────────────────────────────────────────────
    #
    #  Subscribes: /cmd_vel  →  drives JetBot motors via jetbot Python library.
    #  The jetbot library must be installed (see JETBOT_SETUP.md).
    #
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

    # ── 4. RealSense depth → scalar distances (t=4 s) ─────────────────────
    #
    #  Converts /camera/depth/image_rect_raw into:
    #    /distance/front  (std_msgs/Float32)  metres
    #    /distance/right  (std_msgs/Float32)  metres
    #
    #  This replaces jetbot_ultrasonic_driver.py.
    #  HC-SR04 ultrasonic sensors are NOT fitted on the module JetBot.
    #
    realsense_distance = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_realsense_distance.py'),
                ],
                output='screen',
            )
        ],
    )

    # ── 5. Wall follower (t=6 s) ───────────────────────────────────────────
    #
    #  Subscribes: /distance/front, /distance/right, /detections
    #  Publishes:  /cmd_vel
    #  No changes required to jetbot_wall_follower.py — it still reads the
    #  same Float32 distance topics, now supplied by realsense_distance.
    #
    wall_follower = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_wall_follower.py'),
                    '--ros-args',
                    '-p', ['linear_speed:=',    LaunchConfiguration('linear_speed')],
                    '-p', ['wall_follow_dist:=', LaunchConfiguration('wall_follow_dist')],
                    '-p', ['front_stop_dist:=',  LaunchConfiguration('front_stop_dist')],
                ],
                output='screen',
            )
        ],
    )

    # ── 6. Landmark CSV logger (optional, t=6 s) ──────────────────────────
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

    # ── Assemble ───────────────────────────────────────────────────────────
    ld = LaunchDescription(args + [
        camera_node,
        yolo_node,
        motor_driver,
        realsense_distance,
        wall_follower,
        GroupAction(
            condition=IfCondition(LaunchConfiguration('enable_landmark_logger')),
            actions=[landmark_logger],
        ),
    ])

    return ld
