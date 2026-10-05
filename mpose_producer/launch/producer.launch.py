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


    bridge_node = Node(
        package="mpose_producer",
        executable="mpose_bridge_node",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "mpose_pose_topic": mpose_pose_topic,
            "mpose_status_topic": mpose_status_topic
        }],
        emulate_tty=True
    )
    
    return LaunchDescription([
        bridge_node,
    ])