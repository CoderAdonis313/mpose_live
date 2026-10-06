source /opt/ros/humble/setup.bash
source /home/abhi/dev/ros2_ws/install/setup.bash

ros2 launch mpose_consumer consumer.launch.py

################################################### RERUN THIS ON SEPARATE TERMINAL ###################################################
BAG_NAME="mpose_run_$(date +%s)"
echo "INFO: Saving to rosbag $BAG_NAME"
ros2 bag record \
  --storage mcap \
  --output "rosbags/$BAG_NAME" \
  --all-topics
  # --topics \
  #   /mpose/error 


############################################################ Play the ros bag #########################################################
# ros2 bag play "rosbags/$BAG_NAME"