#!/usr/bin/env python3
"""
JetBot RealSense Distance Node
--------------------------------
Converts Intel RealSense D435 depth images into scalar distances published
as std_msgs/Float32.  This REPLACES jetbot_ultrasonic_driver.py because the
module JetBot has a RealSense D435 camera, NOT HC-SR04 ultrasonic sensors.

Published topics:
  /distance/front  (std_msgs/Float32)  -- nearest obstacle ahead (metres)
  /distance/right  (std_msgs/Float32)  -- nearest obstacle to the right (metres)

The depth image is 16-bit unsigned (encoding: 16UC1), where each pixel value
is the distance in millimetres (0 = invalid/no reading).

Region-of-interest layout (fractions of image width W, height H):
  Front region : columns [W*0.375 .. W*0.625],  rows [H*0.25 .. H*0.75]
  Right region : columns [W*0.75  .. W*1.0  ],  rows [H*0.25 .. H*0.75]

Distance is taken as the 10th percentile of valid pixels in each region,
which effectively gives the nearest obstacle while ignoring isolated noise.

Run standalone:
  python3 jetbot_realsense_distance.py

Or via launch (see jetbot_launch.py):
  ros2 run ntu_robotsim jetbot_realsense_distance
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class RealSenseDistanceNode(Node):

    def __init__(self):
        super().__init__('jetbot_realsense_distance')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('depth_topic',         '/camera/depth/image_rect_raw')
        self.declare_parameter('max_range_m',         3.0)
        self.declare_parameter('min_range_m',         0.05)
        # Front ROI: central column strip (fraction of width)
        self.declare_parameter('front_col_lo',        0.375)
        self.declare_parameter('front_col_hi',        0.625)
        # Right ROI: rightmost column strip (fraction of width)
        self.declare_parameter('right_col_lo',        0.75)
        self.declare_parameter('right_col_hi',        1.0)
        # Row band shared by both ROIs (fraction of height) — ignores floor/sky
        self.declare_parameter('row_lo',              0.25)
        self.declare_parameter('row_hi',              0.75)

        depth_topic      = self.get_parameter('depth_topic').value
        self.max_mm      = int(self.get_parameter('max_range_m').value * 1000)
        self.min_mm      = int(self.get_parameter('min_range_m').value * 1000)
        self.front_c_lo  = self.get_parameter('front_col_lo').value
        self.front_c_hi  = self.get_parameter('front_col_hi').value
        self.right_c_lo  = self.get_parameter('right_col_lo').value
        self.right_c_hi  = self.get_parameter('right_col_hi').value
        self.row_lo      = self.get_parameter('row_lo').value
        self.row_hi      = self.get_parameter('row_hi').value

        # ── QoS — RealSense publishes with BEST_EFFORT ───────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscription ────────────────────────────────────────────────────
        self.create_subscription(Image, depth_topic, self._depth_cb, sensor_qos)

        # ── Publishers ──────────────────────────────────────────────────────
        self.front_pub = self.create_publisher(Float32, '/distance/front', 10)
        self.right_pub = self.create_publisher(Float32, '/distance/right', 10)

        if not HAS_NUMPY:
            self.get_logger().error(
                'numpy is not installed!  '
                'Run:  pip3 install numpy'
            )

        self.get_logger().info(
            'RealSense distance node started.  '
            'Depth topic: %s' % depth_topic
        )
        self.get_logger().info(
            'Front ROI cols [%.2f .. %.2f],  '
            'Right ROI cols [%.2f .. %.2f],  '
            'Row band [%.2f .. %.2f]' % (
                self.front_c_lo, self.front_c_hi,
                self.right_c_lo, self.right_c_hi,
                self.row_lo, self.row_hi,
            )
        )

    # ── Depth callback ────────────────────────────────────────────────────

    def _depth_cb(self, msg):
        if not HAS_NUMPY:
            return

        if msg.encoding != '16UC1':
            self.get_logger().warn(
                'Unexpected depth encoding: %s  '
                '(expected 16UC1)' % msg.encoding,
                throttle_duration_sec=5.0,
            )
            return

        h = msg.height
        w = msg.width

        # Decode raw bytes into a 2-D uint16 array (values in mm)
        depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)

        # Row band indices
        r0 = int(h * self.row_lo)
        r1 = int(h * self.row_hi)

        # Front column indices
        cf0 = int(w * self.front_c_lo)
        cf1 = int(w * self.front_c_hi)

        # Right column indices
        cr0 = int(w * self.right_c_lo)
        cr1 = int(w * self.right_c_hi)

        front_m = self._region_near_m(depth, r0, r1, cf0, cf1)
        right_m = self._region_near_m(depth, r0, r1, cr0, cr1)

        msg_f      = Float32()
        msg_f.data = front_m
        self.front_pub.publish(msg_f)

        msg_r      = Float32()
        msg_r.data = right_m
        self.right_pub.publish(msg_r)

        self.get_logger().debug(
            'front=%.3f m   right=%.3f m' % (front_m, right_m)
        )

    # ── Helper ────────────────────────────────────────────────────────────

    def _region_near_m(self, depth, r0, r1, c0, c1):
        """
        Return the 10th-percentile distance in metres from the given ROI.
        Invalid pixels (0) and out-of-range values are excluded.
        Falls back to max_range if no valid pixels found.
        """
        region = depth[r0:r1, c0:c1]
        valid  = region[(region > self.min_mm) & (region < self.max_mm)]
        if valid.size == 0:
            return float(self.max_mm) / 1000.0
        # 10th percentile ≈ nearest obstacle while ignoring isolated noise
        return float(np.percentile(valid, 10)) / 1000.0


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
