# syntax=docker/dockerfile:1
FROM osrf/ros:lyrical-desktop

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-lyrical-moveit \
    ros-lyrical-ros2-control \
    ros-lyrical-ros2-controllers \
    ros-lyrical-gz-ros2-control \
    ros-lyrical-ros-gz-sim \
    ros-lyrical-ros-gz-bridge \
    ros-lyrical-ur-description \
    && rm -rf /var/lib/apt/lists/*

# Create workspace and copy source
WORKDIR /ros_ws
COPY src/ src/

# Install ROS deps and build
RUN . /opt/ros/lyrical/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install

# Source the workspace on every shell invocation
RUN echo "source /opt/ros/lyrical/setup.bash" >> /root/.bashrc && \
    echo "source /ros_ws/install/setup.bash" >> /root/.bashrc

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/lyrical/setup.bash && source /ros_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
