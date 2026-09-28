from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge_node = Node(
        package="mpose_live",
        executable="mpose_bridge_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
        emulate_tty=True
    )

    gt_node = Node(
        package="mpose_live",
        executable="gt_error_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
        emulate_tty=True
    )

    controller_node = Node(
        package="mpose_live",
        executable="controller_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
        emulate_tty=True
    )
    
    return LaunchDescription([
        bridge_node,
        controller_node,
        # gt_node
    ])