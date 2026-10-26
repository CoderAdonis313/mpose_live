from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    bridge_node = Node(
        package="mpose_producer",
        executable="mpose_bridge_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "udp_port": 5005,
                "ip": "192.168.0.119",
                "pose_topic_prefix": "/mpose/poses",
                "packet_topic": "/mpose/packets",
                "max_packets_per_tick": 100,
            }
        ],
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            bridge_node,
        ]
    )
