find_package(OpenCV REQUIRED)

add_library(
  vikit_sophus
  STATIC
  # vikit
  vikit/src/atan_camera.cpp
  vikit/src/omni_camera.cpp
  vikit/src/math_utils.cpp
  vikit/src/vision.cpp
  vikit/src/performance_monitor.cpp
  vikit/src/robust_cost.cpp
  vikit/src/user_input_thread.cpp
  vikit/src/pinhole_camera.cpp
  vikit/src/homography.cpp
  vikit/src/img_align.cpp
  # sophus
  Sophus/sophus/scso3.cpp
  Sophus/sophus/se2.cpp
  Sophus/sophus/se3.cpp
  Sophus/sophus/sim3.cpp
  Sophus/sophus/so2.cpp
  Sophus/sophus/so3.cpp
)
target_link_libraries(
  vikit_sophus
  PUBLIC
  ${OpenCV_LIBS}
  eigen
)
target_include_directories(
  vikit_sophus
  PUBLIC
  vikit/
  vikit/include/
  sophus/
  ${OpenCV_INCLUDE_DIRS}
)
