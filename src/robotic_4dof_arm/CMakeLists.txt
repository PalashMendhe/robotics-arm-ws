cmake_minimum_required(VERSION 3.20)
project(robotic_4dof_arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

install(DIRECTORY urdf launch worlds config
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/pick_and_place.py
  scripts/world_publisher.py
  scripts/planning_scene_manager.py
  scripts/camera_tf_broadcaster.py
  scripts/shape_detector.py
  scripts/sorting_controller.py
  scripts/defect_inspector.py
  scripts/mcp_server.py
  DESTINATION lib/${PROJECT_NAME}
)
# scene_config.py is imported at runtime by pick_and_place / sorting_controller /
# planning_scene_manager (it resolves robot_params.yaml); install it next to them.
install(FILES
  scripts/scene_config.py
  DESTINATION lib/${PROJECT_NAME}
)
# Testing
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_params test/test_params.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_scene_manager test/test_scene_manager.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_vision_sorting test/test_vision_sorting.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_shape_detector test/test_shape_detector.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_cloud_cool test/test_cloud_cool.py
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
endif()

ament_package()
