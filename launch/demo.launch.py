from launch import LaunchDescription
from launch_ros.actions import Node
import time


def generate_launch_description():
    capture_topic = "/capture_step"
    robot_vicon_topic = "/vicon/MARKER_YFORWARD/MARKER_YFORWARD"
    cam_vicon_topic = "/vicon/ZED_CAM/ZED_CAM"
    resolution = "HD1080"
    experiment_name = f"experiment_{time.strftime('%H_%M_%S')}"
    duration = 30.0
    view = 'left'

    vid_node = Node(
        package='mpose_demo',
        executable='rt_node',
        name='rt_node',
        output='screen',
        parameters=[{
            "resolution": resolution,
            "experiment_name": experiment_name,
            "capture_topic": capture_topic,
            "duration": duration,
            "view": view
        }],
        emulate_tty=True,
    )

    gt_node = Node(
        package='mpose_demo',
        executable='gt_pose_node',
        name='gt_pose_node',
        output='screen',
        parameters=[{
            "experiment_name": experiment_name,
            "capture_topic": capture_topic,
            "robot_vicon_topic": robot_vicon_topic,
            "cam_vicon_topic": cam_vicon_topic,
        }],
        emulate_tty=True,
    )

    return LaunchDescription([
        vid_node,
        gt_node
    ])