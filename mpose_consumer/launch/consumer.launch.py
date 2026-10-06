from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mpose_pose_topic = '/mpose/pose'
    mpose_error_topic = '/mpose/error'
    mpose_status_topic = '/mpose/status'
    robot_vicon_topic = '/vicon/MARKER_YFORWARD/MARKER_YFORWARD'
    cam_vicon_topic = '/vicon/ZED_CAM/ZED_CAM'
    gt_pose_topic = '/gt/pose'
    mpose_terror_topic = '/mpose/terror'
    cmd_vel_topic = '/cmd_vel'


    gt_node = Node(
        package="mpose_consumer",
        executable="gt_error_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,

            "mpose_topic_prefix":
                "/mpose/poses",

            "vicon_topic_prefix":
                "/vicon",

            "bot_marker_pattern":
                r"^bot.*_marker$",

            "arena_marker_pattern":
                r"^arena.*_marker$",

            # Recommended when you have exactly one arena.
            "arena_marker_name":
                "arena1_marker",

            "vicon_history_s":
                10.0,

            "max_vicon_gap_s":
                0.05,

            "discovery_period_s":
                1.0,
        }],
        emulate_tty=True,
    )

    controller_node = Node(
        package="mpose_consumer",
        executable="controller_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "mpose_pose_topic": mpose_pose_topic,
            "mpose_status_topic": mpose_status_topic,
            "cmd_vel_topic": cmd_vel_topic
        }],
        emulate_tty=True
    )
    
    return LaunchDescription([
        # controller_node,
        gt_node
    ])