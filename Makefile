# ── Robotic Arm Workspace Makefile ──────────────────────────────────────────
SHELL  := /bin/bash
ROS    := /opt/ros/lyrical/setup.bash
WS     := $(shell pwd)/install/setup.bash

.PHONY: build sim run test clean docker-build docker-sim docker-run vision-sim vision-sim-headless vision-sort vision-clean

## Local (sourced terminal) targets
build:
	source $(ROS) && colcon build --symlink-install

test:
	source $(ROS) && source $(WS) && \
	colcon test --packages-select robotic_4dof_arm && \
	colcon test-result --verbose

sim:
	source $(ROS) && source $(WS) && \
	ros2 launch arm_moveit_config moveit.launch.py

run:
	source $(ROS) && source $(WS) && \
	ros2 run robotic_4dof_arm pick_and_place.py

## Vision sorting targets
vision-sim:
	source $(ROS) && source $(WS) && \
	ros2 launch robotic_4dof_arm vision_sim.launch.py

## Same stack without the Gazebo GUI (server only) — CI / headless boxes.
vision-sim-headless:
	source $(ROS) && source $(WS) && \
	ros2 launch robotic_4dof_arm vision_sim.launch.py headless:=true

vision-sort:
	source $(ROS) && source $(WS) && \
	ros2 run robotic_4dof_arm sorting_controller.py

## Kill stale Gazebo / MoveIt / controller processes without wiping the build.
## Fixes the "duplicate action server on /move_action" failure mode: an old
## stack still answering goals with FAILURE(99999). Patterns use the [x]
## trick so pkill cannot match its own command line.
vision-clean:
	-pkill -9 -f '[m]ove_group' >/dev/null 2>&1 || true
	-pkill -9 -f '[s]orting_controller' >/dev/null 2>&1 || true
	-pkill -9 -f '[p]ick_and_place' >/dev/null 2>&1 || true
	-pkill -9 -f '[r]viz2' >/dev/null 2>&1 || true
	-pkill -9 -f '[g]z sim' >/dev/null 2>&1 || true
	-pkill -9 -f '[p]arameter_bridge' >/dev/null 2>&1 || true
	-pkill -9 -f '[r]obot_state_publisher' >/dev/null 2>&1 || true
	-pkill -9 -f '[c]amera_tf_broadcaster' >/dev/null 2>&1 || true
	-pkill -9 -f '[c]ontroller_manager.*spawner' >/dev/null 2>&1 || true
	@echo "Stale sim processes killed. Restart: make vision-sim  (then)  make vision-sort"

clean:
	rm -rf build/ install/ log/
	-killall -9 move_group ruby ros2 rviz2 >/dev/null 2>&1 || true
	-pkill -9 -f "gz sim" >/dev/null 2>&1 || true
	-pkill -9 -f "ros2" >/dev/null 2>&1 || true

## Docker targets
docker-build:
	docker build -t robotic_arm:lyrical .

docker-sim:
	xhost +local:docker
	docker compose up sim

docker-run:
	docker compose up run

docker-stop:
	docker compose down