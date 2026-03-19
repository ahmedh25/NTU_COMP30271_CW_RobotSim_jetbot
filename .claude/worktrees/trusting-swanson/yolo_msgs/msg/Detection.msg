# Single YOLO detection result.
# Fields accessed by jetbot_wall_follower.py and jetbot_landmark_csv_logger.py:
#   d.class_name
#   d.score
#   d.bbox.size.x / d.bbox.size.y

string class_name
float32 score
BoundingBox2D bbox
