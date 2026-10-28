BOT_TOPIC="${1:-TB1}"
ros2 topic pub --rate 10 /$BOT_TOPIC/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"