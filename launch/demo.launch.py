from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="mpose_live",
            executable="mpose_bridge_node",
            output="screen",
            parameters=[{"use_sim_time": False}],
        ),
        Node(
            package="mpose_live",
            executable="gt_error_node",
            output="screen",
            parameters=[{"use_sim_time": False}],
        ),
    ])