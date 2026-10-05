source /opt/ros/humble/setup.bash
source /home/abhi/dev/ros2_ws/install/setup.bash

ros2 launch mpose_consumer consumer.launch.py
ros2 bag record \
  --storage mcap \
  --output "rosbags/mpose_run_$(date +%s)" \
  --topics \
    /mpose/error \