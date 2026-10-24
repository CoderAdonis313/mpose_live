#!/usr/bin/env python3
"""Compare multiple bot poses relative to one arena using MegaPose and Vicon."""

import math
import re
from bisect import bisect_left

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp
from mpose_interfaces.msg import PoseError


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pose_to_matrix(position, orientation):
    translation = np.array(
        [position.x, position.y, position.z],
        dtype=float,
    )
    quaternion = np.array(
        [orientation.x, orientation.y, orientation.z, orientation.w],
        dtype=float,
    )

    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
        raise ValueError("Pose contains invalid values")

    quaternion /= np.linalg.norm(quaternion)

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def pose_stamped_to_matrix(message):
    return pose_to_matrix(
        message.pose.position,
        message.pose.orientation,
    )


def transform_stamped_to_matrix(message):
    return pose_to_matrix(
        message.transform.translation,
        message.transform.rotation,
    )


def relative_pose(parent_pose, child_pose):
    """Return the child pose expressed in the parent frame."""
    return np.linalg.inv(parent_pose) @ child_pose


def rotation_error_degrees(estimated, ground_truth):
    error_rotation = ground_truth[:3, :3].T @ estimated[:3, :3]
    return float(np.degrees(Rotation.from_matrix(error_rotation).magnitude()))


class PoseHistory:
    """Timestamped Vicon history with translation and rotation interpolation."""

    def __init__(self, duration_ns):
        self.duration_ns = duration_ns
        self.timestamps = []
        self.matrices = []


    def add(self, timestamp_ns, matrix):
        if self.timestamps and timestamp_ns <= self.timestamps[-1]:
            return

        self.timestamps.append(timestamp_ns)
        self.matrices.append(matrix)

        oldest_allowed = timestamp_ns - self.duration_ns
        cut = bisect_left(self.timestamps, oldest_allowed)

        if cut > 0:
            del self.timestamps[:cut]
            del self.matrices[:cut]


    def at(self, timestamp_ns, maximum_gap_ns):
        index = bisect_left(self.timestamps, timestamp_ns)

        if index < len(self.timestamps) and self.timestamps[index] == timestamp_ns:
            return self.matrices[index]

        if index == len(self.timestamps):
            raise LookupError("Waiting for a newer Vicon sample")

        if index == 0:
            raise ValueError("Timestamp is older than Vicon history")

        time_before = self.timestamps[index - 1]
        time_after = self.timestamps[index]

        if time_after - time_before > maximum_gap_ns:
            raise ValueError("Vicon interpolation gap is too large")

        matrix_before = self.matrices[index - 1]
        matrix_after = self.matrices[index]

        alpha = (timestamp_ns - time_before) / (time_after - time_before)

        interpolated = np.eye(4)
        interpolated[:3, 3] = (1.0 - alpha) * matrix_before[:3, 3] + alpha * matrix_after[:3, 3]

        rotations = Rotation.from_matrix(
            np.stack(
                [
                    matrix_before[:3, :3],
                    matrix_after[:3, :3],
                ]
            )
        )
        interpolated[:3, :3] = Slerp(
            [0.0, 1.0],
            rotations,
        )(
            [alpha]
        ).as_matrix()[0]
        return interpolated


class GTErrorNode(Node):
    def __init__(self):
        super().__init__("gt_error_node")

        if self.get_parameter("use_sim_time").value:
            raise ValueError("This live comparison requires use_sim_time=false")

        self.mpose_topic_prefix = str(self.declare_parameter("mpose_topic_prefix", "/mpose/poses").value).rstrip("/")
        self.vicon_topic_prefix = str(self.declare_parameter("vicon_topic_prefix", "/vicon").value).rstrip("/")

        bot_pattern = str(self.declare_parameter("bot_marker_pattern", r"^bot.*_marker$").value)
        arena_pattern = str(self.declare_parameter("arena_marker_pattern", r"^arena.*_marker$").value)

        self.vicon_msg_type = str(self.declare_parameter("vicon_msg_type", "TransformStamped").value)

        # Output prefixes. Each bot label is appended to these.
        self.pose_error_topic = str(self.declare_parameter("pose_error_topic", "/pose_error").value).rstrip("/")

        history_seconds = float(self.declare_parameter("vicon_history_s", 10.0).value) # type: ignore
        maximum_gap_seconds = float(self.declare_parameter("max_vicon_gap_s", 0.05).value)  # type: ignore
        discovery_period = float(self.declare_parameter("discovery_period_s", 1.0).value)  # type: ignore
        self.beta = float(self.declare_parameter("beta", 1.0).value)  # type: ignore

        if not self.beta > 0:
            raise ValueError("Beta value must be positive")

        if not math.isfinite(history_seconds) or history_seconds <= 0:
            raise ValueError("vicon_history_s must be positive")

        if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
            raise ValueError("max_vicon_gap_s must be positive")

        self.bot_pattern = re.compile(bot_pattern, re.IGNORECASE)
        self.arena_pattern = re.compile(arena_pattern, re.IGNORECASE)

        self.history_duration_ns = int(history_seconds * 1_000_000_000)
        self.maximum_vicon_gap_ns = int(maximum_gap_seconds * 1_000_000_000)

        # Marker label -> subscription.
        self.mpose_subscriptions = {}
        self.vicon_subscriptions = {}

        # Marker label -> (timestamp, camera frame, pose matrix).
        self.latest_mpose = {}

        # Marker label -> PoseHistory.
        self.vicon_histories = {}

        # Marker label -> Vicon parent frame.
        self.vicon_parent_frames = {}

        # (arena, label, Bot label) -> last evaluated or permanently rejected timestamp.
        self.last_evaluated = {}

        # Bot label -> (translation, rotation, loss) publishers.
        self.error_publishers = {}

        self.vicon_qos = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_timer(
            discovery_period,
            self.discover_topics,
        )

        # Discover existing markers immediately. The timer discovers
        # additional markers that appear after startup.
        self.discover_topics()

        self.get_logger().info(f"Looking for MegaPose topics under " f"{self.mpose_topic_prefix}/")
        self.get_logger().info(f"Looking for Vicon topics under " f"{self.vicon_topic_prefix}/")
        self.get_logger().info(f"Bot pattern: {bot_pattern}; " f"arena pattern: {arena_pattern}")

    def is_marker_name(self, label):
        return bool(self.bot_pattern.fullmatch(label) or self.arena_pattern.fullmatch(label))

    def pose_loss(self, estimated, ground_truth):
        est_quats = Rotation.from_matrix(estimated[:3, :3]).as_quat() # type: ignore
        gt_quats = Rotation.from_matrix(ground_truth[:3, :3]).as_quat() # type: ignore

        # q and -q represent the same rotation.
        diff_quats = np.sum((est_quats - gt_quats) ** 2)
        sum_quats = np.sum((est_quats + gt_quats) ** 2)

        rotation_loss = float(np.sqrt(min(diff_quats, sum_quats)))
        translation_loss = float(np.sqrt(np.sum((estimated[:3, 3] - ground_truth[:3, 3]) ** 2)))
        loss = float(translation_loss + self.beta * rotation_loss)
        return loss, translation_loss, rotation_loss

    def error_publisher(self, bot_label):
        """Create and cache one PoseError publisher per bot."""
        if bot_label not in self.error_publishers:
            topic = f"{self.pose_error_topic}/{bot_label}"

            self.error_publishers[bot_label] = self.create_publisher(
                PoseError,
                topic,
                10,
            )

            self.get_logger().info(f"Pose errors for {bot_label!r} -> {topic}")
        return self.error_publishers[bot_label]

    def publish_pose_error(
        self,
        timestamp_ns,
        arena_label,
        bot_label,
        translation_error_m,
        rotation_error_deg,
        pose_loss,
        quat_loss
    ):
        """Publish one complete, timestamped error measurement."""
        message = PoseError()

        seconds, nanoseconds = divmod(
            timestamp_ns,
            1_000_000_000,
        )

        message.header.stamp.sec = int(seconds)
        message.header.stamp.nanosec = int(nanoseconds)
        message.header.frame_id = arena_label

        message.bot_label = bot_label
        message.translation_error_m = translation_error_m
        message.rotation_error_deg = rotation_error_deg
        message.pose_loss = pose_loss
        message.rotation_error_quat = quat_loss
        self.error_publisher(bot_label).publish(message)

    def pose_error_calculation(
        self,
        timestamp_ns,
        arena_label,
        bot_label,
        ground_truth,
        estimated,
    ):
        translation_delta = (
            estimated[:3, 3] - ground_truth[:3, 3]
        )
        translation_error_m = float(
            np.linalg.norm(translation_delta)
        )

        rotation_error_deg = rotation_error_degrees(
            estimated,
            ground_truth,
        )

        loss, _, quat_loss = self.pose_loss(estimated, ground_truth)

        self.publish_pose_error(
            timestamp_ns,
            arena_label,
            bot_label,
            translation_error_m,
            rotation_error_deg,
            loss,
            quat_loss
        )

    @staticmethod
    def topic_marker_name(topic):
        """Match MegaPose and Vicon using the final topic component."""
        return topic.rstrip("/").split("/")[-1].lower()

    @staticmethod
    def under_prefix(topic, prefix):
        return topic == prefix.lower() or topic.startswith(prefix + "/")

    def discover_topics(self):
        for topic, message_types in self.get_topic_names_and_types():
            label = self.topic_marker_name(topic)

            if not self.is_marker_name(label):
                continue

            if self.under_prefix(topic, self.mpose_topic_prefix):
                self.add_mpose_subscription(label, topic)

            if self.under_prefix(topic, self.vicon_topic_prefix):
                if self.vicon_msg_type == "PoseStamped":
                    self.add_vicon_pose_subscription(label, topic)
                elif self.vicon_msg_type == "TransformStamped":
                    self.add_vicon_transform_subscription(label, topic)

    def add_mpose_subscription(self, label, topic):
        if label in self.mpose_subscriptions:
            return

        subscription = self.create_subscription(
            PoseStamped,
            topic,
            lambda message, marker=label: self.on_mpose(marker, message),
            50,
        )
        self.mpose_subscriptions[label] = subscription
        self.get_logger().info(f"MegaPose marker {label!r}: {topic}")

    def add_vicon_pose_subscription(self, label, topic):
        if label in self.vicon_subscriptions:
            return

        subscription = self.create_subscription(
            PoseStamped,
            topic,
            lambda message, marker=label: self.on_vicon_pose(marker, message),
            self.vicon_qos,
        )
        self.vicon_subscriptions[label] = subscription
        self.vicon_histories[label] = PoseHistory(self.history_duration_ns)

        self.get_logger().info(f"Vicon marker {label!r}: {topic} [PoseStamped]")

    def add_vicon_transform_subscription(self, label, topic):
        if label in self.vicon_subscriptions:
            return

        subscription = self.create_subscription(
            TransformStamped,
            topic,
            lambda message, marker=label: self.on_vicon_transform(marker, message),
            self.vicon_qos,
        )
        self.vicon_subscriptions[label] = subscription
        self.vicon_histories[label] = PoseHistory(self.history_duration_ns)

        self.get_logger().info(f"Vicon marker {label!r}: {topic} [TransformStamped]")

    def on_mpose(self, label, message):
        timestamp_ns = stamp_ns(message.header.stamp)

        if timestamp_ns <= 0 or not message.header.frame_id:
            return

        try:
            matrix = pose_stamped_to_matrix(message)
        except ValueError:
            return

        self.latest_mpose[label] = (
            timestamp_ns,
            message.header.frame_id,
            matrix,
        )
        self.compare_available()

    def add_vicon_sample(
        self,
        label,
        timestamp_ns,
        parent_frame,
        matrix,
    ):
        if timestamp_ns <= 0 or not parent_frame:
            return

        existing_frame = self.vicon_parent_frames.get(label)

        if existing_frame is not None and existing_frame != parent_frame:
            self.get_logger().warning(f"Vicon parent frame changed for {label!r}: " f"{existing_frame!r} -> {parent_frame!r}")
            return

        self.vicon_parent_frames[label] = parent_frame
        self.vicon_histories[label].add(timestamp_ns, matrix)

        # New Vicon data may allow a previously pending comparison.
        self.compare_available()

    def on_vicon_transform(self, label, message):
        try:
            matrix = transform_stamped_to_matrix(message)
        except ValueError:
            return

        self.add_vicon_sample(
            label,
            stamp_ns(message.header.stamp),
            message.header.frame_id,
            matrix,
        )

    def on_vicon_pose(self, label, message):
        try:
            matrix = pose_stamped_to_matrix(message)
        except ValueError:
            return

        self.add_vicon_sample(
            label,
            stamp_ns(message.header.stamp),
            message.header.frame_id,
            matrix,
        )


    def compare_available(self):
        arena_labels = sorted(label for label in self.latest_mpose if self.arena_pattern.fullmatch(label) and label in self.vicon_histories)
        bot_labels = sorted(label for label in self.latest_mpose if self.bot_pattern.fullmatch(label) and label in self.vicon_histories)

        for arena_label in arena_labels:
            (
                arena_timestamp,
                camera_frame,
                camera_T_arena,
            ) = self.latest_mpose[arena_label]

            arena_history = self.vicon_histories[arena_label]
            arena_parent = self.vicon_parent_frames.get(arena_label)

            if arena_parent is None:
                continue

            for bot_label in bot_labels:
                (
                    bot_timestamp,
                    bot_camera_frame,
                    camera_T_bot,
                ) = self.latest_mpose[bot_label]

                # Both estimates must originate from the same camera image.
                if bot_timestamp != arena_timestamp:
                    continue

                if bot_camera_frame != camera_frame:
                    continue

                pair_key = (arena_label, bot_label)

                if bot_timestamp <= self.last_evaluated.get(pair_key, -1):
                    continue

                bot_parent = self.vicon_parent_frames.get(bot_label)

                if bot_parent is None or bot_parent != arena_parent:
                    continue

                bot_history = self.vicon_histories[bot_label]

                try:
                    world_T_arena = arena_history.at(
                        bot_timestamp,
                        self.maximum_vicon_gap_ns,
                    )
                    world_T_bot = bot_history.at(
                        bot_timestamp,
                        self.maximum_vicon_gap_ns,
                    )

                except LookupError:
                    # A newer Vicon sample may still arrive.
                    continue

                except ValueError as error:
                    self.get_logger().warning(f"Cannot match {bot_label!r} relative to " f"{arena_label!r} at {bot_timestamp}: {error}")
                    self.last_evaluated[pair_key] = bot_timestamp
                    continue

                estimated_arena_T_bot = relative_pose(
                    camera_T_arena,
                    camera_T_bot,
                )
                ground_truth_arena_T_bot = relative_pose(
                    world_T_arena,
                    world_T_bot,
                )

                self.pose_error_calculation(
                    bot_timestamp,
                    arena_label,
                    bot_label,
                    ground_truth_arena_T_bot,
                    estimated_arena_T_bot,
                )

                self.last_evaluated[pair_key] = bot_timestamp


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = GTErrorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
