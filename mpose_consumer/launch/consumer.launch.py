from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mpose_pose_topic = "/mpose/pose"
    mpose_status_topic = "/mpose/status"
    vicon_prefix = "/vicon"
    mpose_prefix = "/mpose/poses"
    vicon_msg_type = "PoseStamped"
    bot_topic = "/TB1/cmd_vel"
    bot_pattern = r"^bot.*_marker$"
    arena_pattern = f"^arena.*_marker$"
    bot_gt_pattern = r"^bot.*_marker$"
    arena_gt_pattern = f"^arena.*_marker$"
    controlled_bot = "bot1_marker"
    controlled_arena = "arena1_marker"

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
                "bot_marker_pattern": bot_pattern,
                "arena_marker_pattern": arena_pattern,
                "vicon_history_s": 10.0,
                "max_vicon_gap_s": 0.05,
                "discovery_period_s": 1.0,
            }
        ],
        emulate_tty=True,
    )

    estimated_relative_pose = Node(
        package="mpose_consumer",
        executable="rel_pose_node",
        name="estimated_relative_pose",
        parameters=[{
            "input_topic_prefix": mpose_prefix,
            "input_message_type": "PoseStamped",
            "output_topic_prefix": "/relative_pose/estimated",
            "bot_marker_pattern": bot_pattern,
            "arena_marker_pattern": arena_pattern,
            "remap": False
        }],
    )

    ground_truth_relative_pose = Node(
        package="mpose_consumer",
        executable="rel_pose_node",
        name="ground_truth_relative_pose",
        parameters=[{
            "input_topic_prefix": vicon_prefix,
            "input_message_type": vicon_msg_type,
            "output_topic_prefix": "/relative_pose/ground_truth",
            "bot_marker_pattern": bot_gt_pattern,
            "arena_marker_pattern": arena_gt_pattern,
            "remap": False
        }],
    )

    controller_node = Node(
        package="mpose_consumer",
        executable="controller_node",
        name="controller_bot1",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "control_mode": "mpose",
                "estimated_pose_topic": (f"/relative_pose/estimated/" f"{controlled_bot}/{controlled_bot}"),
                "ground_truth_pose_topic": (f"/relative_pose/ground_truth/" f"{controlled_bot}/{controlled_bot}"),
                "arena_frame": controlled_arena,
                "bot_frame": controlled_bot,
                "pose_rate": 30.0,
                "pose_timeout_s": 1.0,
                "drive_topic": bot_topic,
                "dist_tolerance": 0.05,
                "angle_tolerance": 10.0,
                "max_lin_speed": 0.1,
                "max_ang_speed": 0.1,
            }
        ],
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            gt_node,
            ground_truth_relative_pose,
            estimated_relative_pose,
            # controller_node,
        ]
    )
