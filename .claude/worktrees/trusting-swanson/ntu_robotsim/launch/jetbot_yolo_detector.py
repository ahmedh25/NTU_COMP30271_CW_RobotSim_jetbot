#!/usr/bin/env python3
"""
JetBot YOLO Detector Node  (ROS2 Eloquent compatible)
------------------------------------------------------
Replaces the 'yolo_ros' package which only supports ROS2 Humble.
Uses the 'ultralytics' library directly to run YOLO inference and
publishes results as yolo_msgs/DetectionArray — the same message type
consumed by jetbot_wall_follower.py and jetbot_landmark_csv_logger.py.

Subscribes : /camera/color/image_raw   (sensor_msgs/Image)
             -- The Intel RealSense D435 colour stream topic.
             -- Inspect with:  ros2 topic list   after launching realsense2_camera.

Publishes  : /detections               (yolo_msgs/DetectionArray)

Python 3.6 compatible (no walrus operator, no | union types,
no lowercase generics).

Run standalone:
  python3 jetbot_yolo_detector.py

Or with custom parameters:
  python3 jetbot_yolo_detector.py --ros-args \\
      -p model_path:=/home/jetbot/ros2_ws/src/.../best_nano.pt \\
      -p threshold:=0.45 \\
      -p device:=cuda:0 \\
      -p image_topic:=/camera/color/image_raw
"""

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    from yolo_msgs.msg import DetectionArray, Detection, BoundingBox2D
    from geometry_msgs.msg import Point2D
    HAS_YOLO_MSGS = True
except ImportError:
    HAS_YOLO_MSGS = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Default model path — relative to this file's location (launch/)
_LAUNCH_DIR   = os.path.dirname(os.path.realpath(__file__))
_MODELS_DIR   = os.path.join(os.path.dirname(_LAUNCH_DIR), 'models', 'custom_models')
_DEFAULT_MODEL = os.path.join(_MODELS_DIR, 'best_nano.pt')


class JetBotYoloDetector(Node):

    def __init__(self):
        super().__init__('jetbot_yolo_detector')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('model_path',   _DEFAULT_MODEL)
        self.declare_parameter('threshold',    0.5)
        self.declare_parameter('device',       'cuda:0')
        self.declare_parameter('image_topic',  '/camera/color/image_raw')

        model_path  = self.get_parameter('model_path').value
        self.thresh = self.get_parameter('threshold').value
        device      = self.get_parameter('device').value
        image_topic = self.get_parameter('image_topic').value

        # ── Dependency checks ────────────────────────────────────────────────
        if not HAS_NUMPY:
            self.get_logger().error(
                'numpy not installed.  Run:  pip3 install numpy'
            )
        if not HAS_ULTRALYTICS:
            self.get_logger().error(
                'ultralytics not installed.  Run:  pip3 install ultralytics'
            )
        if not HAS_YOLO_MSGS:
            self.get_logger().error(
                'yolo_msgs not found.  '
                'Build yolo_msgs in ~/ros2_ws/src/yolo_msgs then '
                'source ~/ros2_ws/install/setup.bash'
            )

        # ── Load YOLO model ──────────────────────────────────────────────────
        self.model = None
        if HAS_ULTRALYTICS:
            try:
                self.model = YOLO(model_path)
                self.model.to(device)
                self.get_logger().info('YOLO model loaded: %s' % model_path)
                self.get_logger().info('Inference device : %s' % device)
            except Exception as exc:
                self.get_logger().error(
                    'Failed to load YOLO model: %s' % str(exc)
                )

        # ── QoS — RealSense colour topics use BEST_EFFORT ───────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscription ────────────────────────────────────────────────────
        self.create_subscription(Image, image_topic, self._image_cb, sensor_qos)

        # ── Publisher ────────────────────────────────────────────────────────
        if HAS_YOLO_MSGS:
            self.det_pub = self.create_publisher(DetectionArray, '/detections', 10)
        else:
            self.det_pub = None

        self.get_logger().info(
            'YOLO detector started.  Subscribing to: %s' % image_topic
        )

    # ── Image callback ────────────────────────────────────────────────────

    def _image_cb(self, msg):
        if not (HAS_ULTRALYTICS and HAS_NUMPY and HAS_YOLO_MSGS):
            return
        if self.model is None or self.det_pub is None:
            return

        # ── Convert ROS Image → numpy ────────────────────────────────────────
        try:
            channels = msg.step // msg.width
            img_np   = np.frombuffer(
                bytes(msg.data), dtype=np.uint8
            ).reshape(msg.height, msg.width, channels)
        except Exception as exc:
            self.get_logger().warn(
                'Image conversion failed: %s' % str(exc)
            )
            return

        # ultralytics expects RGB
        if msg.encoding == 'bgr8':
            img_np = img_np[:, :, ::-1].copy()
        elif msg.encoding in ('rgb8', 'rgba8'):
            img_np = img_np[:, :, :3]
        else:
            self.get_logger().warn(
                'Unsupported image encoding: %s' % msg.encoding,
                throttle_duration_sec=5.0,
            )
            return

        # ── Run YOLO inference ───────────────────────────────────────────────
        try:
            results = self.model(img_np, conf=self.thresh, verbose=False)
        except Exception as exc:
            self.get_logger().warn('YOLO inference error: %s' % str(exc))
            return

        # ── Build DetectionArray message ─────────────────────────────────────
        det_array        = DetectionArray()
        det_array.header = msg.header

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            names = result.names

            for i in range(len(boxes)):
                box    = boxes[i]
                cls_id = int(box.cls[0].item())
                score  = float(box.conf[0].item())
                xyxy   = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2]

                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), \
                                  float(xyxy[2]), float(xyxy[3])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

                center      = Point2D()
                center.x    = cx
                center.y    = cy

                size        = Point2D()
                size.x      = bw
                size.y      = bh

                bbox        = BoundingBox2D()
                bbox.center = center
                bbox.size   = size

                det            = Detection()
                det.class_name = names[cls_id]
                det.score      = score
                det.bbox       = bbox

                det_array.detections.append(det)

        self.det_pub.publish(det_array)

        if det_array.detections:
            self.get_logger().debug(
                'Detected %d objects' % len(det_array.detections)
            )


def main(args=None):
    rclpy.init(args=args)
    node = JetBotYoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
