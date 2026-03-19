#!/usr/bin/env python3
"""
JetBot Right-Hand Wall Follower  (RealSense D435 version)
----------------------------------------------------------
Drop-in replacement for jetbot_wall_follower.py that uses the
Intel RealSense D435 depth camera instead of HC-SR04 ultrasonic sensors.

Depth image processing:
  /camera/depth/image_rect_raw  (sensor_msgs/Image, 16UC1, mm units)

  Front distance  = median of a central ROI strip
  Right distance  = median of a right-side ROI strip

  ROI layout (640×480 depth frame):
    Front : cols 220–420, rows 180–300  (centre band, mid-height)
    Right : cols 540–620, rows 150–330  (right edge, mid-height)

  Depth values of 0 are invalid (no return) — treated as infinity.

All control logic, YOLO reactions, object counting, and CSV logging are
identical to the original jetbot_wall_follower.py.

Run standalone:
  python3 jetbot_wall_follower_d435.py

Or with custom parameters:
  python3 jetbot_wall_follower_d435.py --ros-args \
      -p linear_speed:=0.20  -p wall_follow_dist:=0.5
"""

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

try:
    from yolo_msgs.msg import DetectionArray
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

COUNTS_FILE = os.path.expanduser('~/ros2_ws/object_counts.txt')

# How long to wait for a depth frame before declaring sensor offline (s)
SENSOR_TIMEOUT_S = 5.0

# ── ROI definitions (pixels) ─────────────────────────────────────────────────
# Tune these if your camera resolution differs from 640×480
FRONT_COL_MIN, FRONT_COL_MAX = 220, 420
FRONT_ROW_MIN, FRONT_ROW_MAX = 180, 300

RIGHT_COL_MIN, RIGHT_COL_MAX = 540, 620
RIGHT_ROW_MIN, RIGHT_ROW_MAX = 150, 330


def _depth_roi_median(img_array: np.ndarray,
                      row_min: int, row_max: int,
                      col_min: int, col_max: int) -> float:
    """Return median depth (metres) of a ROI, ignoring invalid (0) pixels."""
    roi = img_array[row_min:row_max, col_min:col_max]
    valid = roi[roi > 0]
    if valid.size == 0:
        return float('inf')
    return float(np.median(valid)) / 1000.0   # mm → m


class JetBotWallFollower(Node):

    def __init__(self):
        super().__init__('jetbot_wall_follower')

        # ── Tuneable parameters ──────────────────────────────────────────────
        self.declare_parameter('linear_speed',    0.20)
        self.declare_parameter('angular_speed',   0.5)
        self.declare_parameter('wall_follow_dist', 0.5)
        self.declare_parameter('front_stop_dist', 0.40)
        self.declare_parameter('stop_duration',   3.0)

        self.linear_speed     = self.get_parameter('linear_speed').value
        self.angular_speed    = self.get_parameter('angular_speed').value
        self.wall_follow_dist = self.get_parameter('wall_follow_dist').value
        self.front_stop_dist  = self.get_parameter('front_stop_dist').value
        self.stop_duration    = self.get_parameter('stop_duration').value

        # ── Depth image subscription ─────────────────────────────────────────
        self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self._depth_cb,
            10,
        )

        # ── YOLO subscription ────────────────────────────────────────────────
        if HAS_YOLO:
            self.create_subscription(
                DetectionArray, '/detections',
                self._detection_cb, 10)
            self.get_logger().info('YOLO detection subscription active')
        else:
            self.get_logger().warn(
                'yolo_msgs not found — sign detection disabled')

        # ── Publisher ────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── State ────────────────────────────────────────────────────────────
        self.front_dist = float('inf')
        self.right_dist = float('inf')
        self._last_depth_t = None

        self.stopped        = False
        self.stop_start_ns  = None
        self.speed_factor   = 1.0
        self.sign_cooldown  = {}

        # ── Object counting ──────────────────────────────────────────────────
        self.object_counts   = {}
        self.object_cooldown = {}
        self.COUNTABLE       = {'orange', 'tree', 'trees', 'car'}

        # ── Timers ───────────────────────────────────────────────────────────
        self.create_timer(0.1,  self._control_loop)
        self.create_timer(10.0, self._save_counts)

        self.get_logger().info(
            f'JetBot D435 wall follower started  '
            f'speed={self.linear_speed}  '
            f'wall_d={self.wall_follow_dist}  '
            f'front_d={self.front_stop_dist}'
        )
        self.get_logger().info(
            'Waiting for /camera/depth/image_rect_raw ...'
        )

    # ── Depth image callback ─────────────────────────────────────────────────

    def _depth_cb(self, msg: Image) -> None:
        # Convert ROS Image to numpy array (encoding: 16UC1, values in mm)
        dtype = np.uint16
        img = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.width
        )

        self.front_dist = _depth_roi_median(
            img,
            FRONT_ROW_MIN, FRONT_ROW_MAX,
            FRONT_COL_MIN, FRONT_COL_MAX,
        )
        self.right_dist = _depth_roi_median(
            img,
            RIGHT_ROW_MIN, RIGHT_ROW_MAX,
            RIGHT_COL_MIN, RIGHT_COL_MAX,
        )
        self._last_depth_t = time.monotonic()

    # ── YOLO detection callback ──────────────────────────────────────────────

    def _detection_cb(self, msg) -> None:
        now_ns = self.get_clock().now().nanoseconds

        for d in msg.detections:
            if d.score < 0.5:
                continue
            name = d.class_name
            area = d.bbox.size.x * d.bbox.size.y

            # ── Count objects ────────────────────────────────────────────────
            if name.lower() in self.COUNTABLE and area > 2000:
                last_count = self.object_cooldown.get(name, 0)
                if (now_ns - last_count) > 15_000_000_000:
                    self.object_counts[name] = (
                        self.object_counts.get(name, 0) + 1
                    )
                    self.object_cooldown[name] = now_ns
                    self.get_logger().info(
                        f'COUNTED {name}  '
                        f'(total: {self.object_counts[name]})  '
                        f'conf={d.score:.2f}  area={area:.0f}'
                    )
                    self._save_counts()
                continue

            # ── Sign reactions ───────────────────────────────────────────────
            if area < 4000:
                continue

            last = self.sign_cooldown.get(name, 0)
            if (now_ns - last) < 30_000_000_000:
                continue

            if 'stop' in name.lower():
                self.get_logger().info(
                    f'STOP sign  conf={d.score:.2f}  area={area:.0f} — '
                    f'stopping {self.stop_duration}s'
                )
                self.stopped       = True
                self.stop_start_ns = now_ns
                self.sign_cooldown[name] = now_ns

            elif 'slow' in name.lower():
                self.get_logger().info(
                    f'SLOW sign  conf={d.score:.2f}  area={area:.0f} — '
                    f'reducing speed'
                )
                self.speed_factor = 0.5
                self.sign_cooldown[name] = now_ns

            elif 'fast' in name.lower():
                self.get_logger().info(
                    f'FAST sign  conf={d.score:.2f}  area={area:.0f} — '
                    f'increasing speed'
                )
                self.speed_factor = 1.5
                self.sign_cooldown[name] = now_ns

    # ── Save counts ──────────────────────────────────────────────────────────

    def _save_counts(self) -> None:
        if not self.object_counts:
            return
        try:
            with open(COUNTS_FILE, 'w') as f:
                f.write('=== YOLO Object Detection Counts ===\n\n')
                for k in sorted(self.object_counts):
                    f.write(f'{k}: {self.object_counts[k]}\n')
                total = sum(self.object_counts.values())
                f.write(f'\nTotal objects detected: {total}\n')
        except Exception:
            pass
        summary = '  '.join(
            f'{k}: {v}' for k, v in sorted(self.object_counts.items())
        )
        self.get_logger().info(
            f'=== OBJECT COUNTS: {summary} ===',
            throttle_duration_sec=10.0,
        )

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        twist = Twist()
        now_m = time.monotonic()

        # ── Check sensor health ──────────────────────────────────────────────
        depth_ok = (
            self._last_depth_t is not None
            and (now_m - self._last_depth_t) < SENSOR_TIMEOUT_S
        )

        if not depth_ok:
            self.get_logger().warn(
                'No /camera/depth/image_rect_raw data — '
                'is the RealSense camera driver running?  Robot stopped.',
                throttle_duration_sec=3.0,
            )
            self.cmd_pub.publish(twist)   # zero velocity — safe stop
            return

        # ── Handle stop state ────────────────────────────────────────────────
        if self.stopped:
            if self.stop_start_ns is not None:
                elapsed = (
                    self.get_clock().now().nanoseconds - self.stop_start_ns
                ) / 1e9
                if elapsed > self.stop_duration:
                    self.stopped       = False
                    self.stop_start_ns = None
                    self.speed_factor  = 1.0
                    self.get_logger().info('Resuming after stop sign')
            self.cmd_pub.publish(twist)
            return

        speed = self.linear_speed * self.speed_factor
        turn  = self.angular_speed
        wd    = self.wall_follow_dist

        # ── Right-hand wall following ────────────────────────────────────────
        if self.front_dist < self.front_stop_dist:
            # Blocked ahead — turn left in place
            twist.linear.x  = 0.0
            twist.angular.z = turn

        elif self.front_dist < self.front_stop_dist * 2.0:
            # Approaching front wall — slow and start turning left
            twist.linear.x  = speed * 0.3
            twist.angular.z = turn * 0.5

        elif self.right_dist > wd * 2.0:
            # No right wall — turn right to find it
            twist.linear.x  = speed * 0.5
            twist.angular.z = -turn * 0.7

        elif self.right_dist < wd * 0.4:
            # Too close to right wall — veer left
            twist.linear.x  = speed * 0.5
            twist.angular.z = turn * 0.5

        else:
            # Cruising — proportional correction
            error = self.right_dist - wd
            twist.linear.x  = speed
            twist.angular.z = -error * 0.8

        self.cmd_pub.publish(twist)

        self.get_logger().debug(
            f'front={self.front_dist:.2f}m  right={self.right_dist:.2f}m  '
            f'lin={twist.linear.x:.3f}  ang={twist.angular.z:.3f}',
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JetBotWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_counts()
        if node.object_counts:
            print('\n=== FINAL OBJECT COUNTS ===')
            for k in sorted(node.object_counts):
                print(f'  {k}: {node.object_counts[k]}')
            print(f'Saved to {COUNTS_FILE}')
        node.cmd_pub.publish(Twist())   # stop motors
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()