#!/usr/bin/env python3
"""Publish the bot pose relative to the arena from MegaPose and Vicon."""

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation


def pose_to_matrix(message):
    position = message.pose.position
    orientation = message.pose.orientation

    quaternion = np.array(
        [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ],
        dtype=float,
    )

    if np.linalg.norm(quaternion) < 1e-9:
        raise ValueError("Invalid zero-length quaternion")

    quaternion /= np.linalg.norm(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = [
        position.x,
        position.y,
        position.z,
    ]

    if not np.isfinite(matrix).all():
        raise ValueError("Pose contains non-finite values")
    return matrix


def matrix_to_pose(matrix, stamp, frame_id):
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = frame_id

    message.pose.position.x = float(matrix[0, 3])
    message.pose.position.y = float(matrix[1, 3])
    message.pose.position.z = float(matrix[2, 3])

    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat() #type: ignore

    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])
    return message


class RelativePoseNode(Node):
    def __init__(self):
        super().__init__("relative_pose_node")

        self.arena_name = str(self.declare_parameter("arena_marker_name", "").value)
        self.bot_name = str(self.declare_parameter("bot_marker_name", "").value)
        mpose_prefix = str(self.declare_parameter("mpose_topic_prefix", "/mpose/poses",).value).rstrip("/")
        vicon_prefix = str(self.declare_parameter("vicon_topic_prefix", "").value)


        estimated_topic = str(self.declare_parameter("estimated_output_topic", "",).value)
        ground_truth_topic = str(self.declare_parameter("ground_truth_output_topic", "",).value)
        sync_tolerance = float(self.declare_parameter("sync_tolerance_s", 0.01,).value) #type: ignore

        mpose_arena_topic = f"{mpose_prefix}/{self.arena_name}"
        mpose_bot_topic = f"{mpose_prefix}/{self.bot_name}"

        arena_vicon_name = self.arena_name.upper()
        bot_vicon_name = self.bot_name.upper()

        vicon_arena_topic = (f"{vicon_prefix}/" f"{arena_vicon_name}/" f"{arena_vicon_name}")
        vicon_bot_topic = f"{vicon_prefix}/" f"{bot_vicon_name}/" f"{bot_vicon_name}"

        self.estimated_publisher = self.create_publisher(
            PoseStamped,
            estimated_topic,
            10,
        )

        self.ground_truth_publisher = self.create_publisher(
            PoseStamped,
            ground_truth_topic,
            10,
        )

        # MegaPose topics are published reliably.
        self.mpose_arena_sub = Subscriber(
            self,
            PoseStamped,
            mpose_arena_topic,
            qos_profile=10,
        )

        self.mpose_bot_sub = Subscriber(
            self,
            PoseStamped,
            mpose_bot_topic,
            qos_profile=10,
        )

        self.mpose_sync = ApproximateTimeSynchronizer(
            [
                self.mpose_arena_sub,
                self.mpose_bot_sub,
            ],
            queue_size=30,
            slop=sync_tolerance,
        )
        self.mpose_sync.registerCallback(self.on_mpose_pair)

        # BEST_EFFORT works with both live Vicon and bag playback.
        vicon_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.vicon_arena_sub = Subscriber(
            self,
            PoseStamped,
            vicon_arena_topic,
            qos_profile=vicon_qos,
        )

        self.vicon_bot_sub = Subscriber(
            self,
            PoseStamped,
            vicon_bot_topic,
            qos_profile=vicon_qos,
        )

        self.vicon_sync = ApproximateTimeSynchronizer(
            [
                self.vicon_arena_sub,
                self.vicon_bot_sub,
            ],
            queue_size=100,
            slop=sync_tolerance,
        )
        self.vicon_sync.registerCallback(self.on_vicon_pair)

        self.get_logger().info(f"Estimated input: {mpose_arena_topic} + " f"{mpose_bot_topic}")
        self.get_logger().info(f"Ground-truth input: {vicon_arena_topic} + " f"{vicon_bot_topic}")
        self.get_logger().info(f"Estimated output: {estimated_topic}")
        self.get_logger().info(f"Ground-truth output: {ground_truth_topic}")

    def calculate_relative_pose(self, arena_message, bot_message,):
        if arena_message.header.frame_id != bot_message.header.frame_id:
            raise ValueError("Arena and bot have different parent frames")

        parent_T_arena = pose_to_matrix(arena_message)
        parent_T_bot = pose_to_matrix(bot_message)

        # Pose of the bot expressed in the arena frame.
        return np.linalg.inv(parent_T_arena) @ parent_T_bot

    def publish_relative(self, arena_message, bot_message, publisher, source_name):
        try:
            arena_T_bot = self.calculate_relative_pose(arena_message, bot_message,)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.get_logger().warning(f"Rejected {source_name} pose pair: {error}")
            return

        output = matrix_to_pose(arena_T_bot, bot_message.header.stamp, self.arena_name,)
        publisher.publish(output)

    def on_mpose_pair(self, arena_message, bot_message):
        self.publish_relative(
            arena_message,
            bot_message,
            self.estimated_publisher,
            "MegaPose",
        )

    def on_vicon_pair(self, arena_message, bot_message,):
        self.publish_relative(
            arena_message,
            bot_message,
            self.ground_truth_publisher,
            "Vicon",
        )


def main(args=None):
    rclpy.init(args=args)
    node = RelativePoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
