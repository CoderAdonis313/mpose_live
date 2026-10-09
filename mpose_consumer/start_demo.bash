source /opt/ros/humble/setup.bash
source /home/abhi/dev/ros2_ws/install/setup.bash

ros2 launch mpose_consumer consumer.launch.py

################################################### RERUN THIS ON SEPARATE TERMINAL ###################################################
BAG_NAME="mpose_run_$(date +%s)"
echo "INFO: Saving to rosbag $BAG_NAME"
ros2 bag record \
  --storage sqlite3 \
  --output "rosbags/$BAG_NAME" \
  --all
  # --topics \
  #   /mpose/error 

############################################################ Summarize the ros bag ####################################################
echo "INFO: Summarizing the rosbag: $BAG_NAME"
python3 analyze_pose_bag.py "rosbags/$BAG_NAME/*.db3"\
 --beta 0.1 \
 --output summary.json


############################################################ Play the ros bag #########################################################
# ros2 bag play "rosbags/$BAG_NAME"
