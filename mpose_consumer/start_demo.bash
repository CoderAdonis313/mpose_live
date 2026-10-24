source /opt/ros/jazzy/setup.bash
source /home/abhi/Dev/ros2_codes/research_ws/install/setup.bash

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
python3 mpose_consumer/analyze_pose_bag.py "rosbags/$BAG_NAME" \
  --beta 0.1 \
  --output summary_$BAG_NAME.json

############################################################ Play the ros bag #########################################################
ros2 bag play "rosbags/$BAG_NAME"

############################################################ Multi bot demo ###########################################################
# Run these separetely
ros2 run mpose_consumer controller_node --ros-args \
  -r __node:=controller_bot1 \
  -p control_mode:=mpose \
  -p estimated_pose_topic:=/relative_pose/estimated/bot1_marker/bot1_marker \
  -p ground_truth_pose_topic:=/relative_pose/ground_truth/bot1_marker/bot1_marker \
  -p drive_topic:=/TB1/cmd_vel \
  -p arena_frame:=arena1_marker \
  -p 'goals:=[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]'