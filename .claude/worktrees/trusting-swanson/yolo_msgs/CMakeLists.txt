cmake_minimum_required(VERSION 3.5)
project(yolo_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox2D.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_package()
