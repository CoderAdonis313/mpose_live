from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mpose_pose_topic = "/mpose/pose"
    mpose_status_topic = "/mpose/status"
    vicon_prefix = "/vicon"
    mpose_prefix = "/mpose/poses"
    vicon_msg_type = "TransformStamped"

    robot_vicon_topic = "/vicon/MARKER_YFORWARD/MARKER_YFORWARD"
    mpose_error_topic = "/mpose/error"
    cam_vicon_topic = "/vicon/ZED_CAM/ZED_CAM"
    gt_pose_topic = "/gt/pose"
    mpose_terror_topic = "/mpose/terror"

    gt_node = Node(
        package="mpose_consumer",
        executable="gt_error_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "mpose_topic_prefix": mpose_prefix,
                "vicon_topic_prefix": vicon_prefix,
                "vicon_msg_type": vicon_msg_type,
                "bot_marker_pattern": r"^bot.*_marker$",
                "arena_marker_pattern": r"^arena.*_marker$",
                # Recommended when you have exactly one arena.
                "arena_marker_name": "arena1_marker",
                "vicon_history_s": 10.0,
                "max_vicon_gap_s": 0.05,
                "discovery_period_s": 1.0,
            }
        ],
        emulate_tty=True,
    )


    relative_pose_node = Node(
        package="mpose_consumer",
        executable="rel_pose_node",
        output="screen",
        parameters=[
            {
                "arena_marker_name": "arena1_marker",
                "bot_marker_name": "bot1_marker",
                "vicon_topic_prefix": vicon_prefix,
                "mpose_topic_prefix": mpose_prefix,
                "vicon_msg_type": vicon_msg_type,
                "estimated_output_topic": "/relative_pose/estimated",
                "ground_truth_output_topic": "/relative_pose/ground_truth",
                "sync_tolerance_s": 0.01,
            }
        ],
    )


    controller_node = Node(
        package="mpose_consumer",
        executable="controller_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": False, 
                "relative_pose_topic": '/relative_pose/ground_truth',
                "pose_rate": 15,
                "drive_topic": '/cmd_vel'
            }
        ],
        emulate_tty=True,
    )


    return LaunchDescription(
        [
            controller_node,
            gt_node,
            relative_pose_node,
        ]
    )
