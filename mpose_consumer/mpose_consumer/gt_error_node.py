#!/usr/bin/env python3
"""Compare arena-relative MegaPose and Vicon marker poses."""

import math
import re
from bisect import bisect_left
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pose_to_matrix(position, orientation):
    translation = np.array(
        [
            position.x,
            position.y,
            position.z,
        ],
        dtype=float,
    )

    quaternion = np.array(
        [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ],
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
    """Timestamped Vicon history with interpolation."""
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
            raise ValueError("Timestamp is older than " "Vicon history")

        time_before = self.timestamps[index - 1]
        time_after = self.timestamps[index]

        if time_after - time_before > maximum_gap_ns:
            raise ValueError("Vicon interpolation gap " "is too large")

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
            [
                0.0,
                1.0,
            ],
            rotations,
        )(
            [alpha]
        ).as_matrix()[0]
        return interpolated


class GTErrorNode(Node):
    def __init__(self):
        super().__init__("gt_error_node")

        if self.get_parameter("use_sim_time").value:
            raise ValueError("This live comparison requires " "use_sim_time=false")

        self.mpose_topic_prefix = str(self.declare_parameter("mpose_topic_prefix", "/mpose/poses").value).rstrip("/")
        self.vicon_topic_prefix = str(self.declare_parameter("vicon_topic_prefix", "/vicon").value).rstrip("/")
        bot_pattern = str(self.declare_parameter("bot_marker_pattern", r"^bot.*_marker$").value)
        arena_pattern = str(self.declare_parameter("arena_marker_pattern", r"^arena.*_marker$").value)
        self.vicon_msg_type = str(self.declare_parameter('vicon_msg_type', "TransformStamped").value)

        # Optional exact arena name. Leave empty when the
        # arena pattern matches only one marker.
        self.configured_arena = str(self.declare_parameter("arena_marker_name", "").value).lower()
        history_seconds = float(self.declare_parameter("vicon_history_s", 10.0).value)  # type: ignore
        maximum_gap_seconds = float(self.declare_parameter("max_vicon_gap_s", 0.05,).value)  # type: ignore
        discovery_period = float(self.declare_parameter("discovery_period_s", 1.0,).value)  # type: ignore

        if not math.isfinite(history_seconds) or history_seconds <= 0:
            raise ValueError("vicon_history_s must be positive")

        if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
            raise ValueError("max_vicon_gap_s must be positive")

        self.bot_pattern = re.compile(bot_pattern, re.IGNORECASE,)
        self.arena_pattern = re.compile(arena_pattern, re.IGNORECASE,)
        self.history_duration_ns = int(history_seconds * 1_000_000_000)
        self.maximum_vicon_gap_ns = int(maximum_gap_seconds * 1_000_000_000)

        # label -> subscription
        self.mpose_subscriptions = {}
        self.vicon_subscriptions = {}

        # label -> (timestamp, camera frame, matrix)
        self.latest_mpose = {}

        # label -> PoseHistory
        self.vicon_histories = {}

        # label -> Vicon parent frame
        self.vicon_parent_frames = {}

        # bot label -> last successfully evaluated timestamp
        self.last_evaluated = {}

        self.discovery_warning_shown = False

        self.vicon_qos = QoSProfile(
            depth=1000,
            reliability=(ReliabilityPolicy.BEST_EFFORT),
        )

        self.create_timer(
            discovery_period,
            self.discover_topics,
        )

        # Try immediately instead of waiting for the first timer.
        self.discover_topics()

        self.get_logger().info("Looking for MegaPose topics under " f"{self.mpose_topic_prefix}/")
        self.get_logger().info("Looking for Vicon topics under " f"{self.vicon_topic_prefix}/")
        self.get_logger().info(f"Bot pattern: {bot_pattern}; " f"arena pattern: {arena_pattern}")


    def is_marker_name(self, label):
        return bool(self.bot_pattern.fullmatch(label) or self.arena_pattern.fullmatch(label))


    @staticmethod
    def topic_marker_name(topic):
        """Use the final topic component as the marker name."""
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
                if self.vicon_msg_type == 'PoseStamped':
                    self.add_vicon_pose_subscription(label, topic)
                if self.vicon_msg_type == 'TransformStamped':
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
        self.get_logger().info(f"MegaPose marker {label!r}: " f"{topic}")


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
        self.get_logger().info(f"Vicon marker {label!r}: " f"{topic} [PoseStamped]")


    def add_vicon_transform_subscription(self, label, topic):
        if label in self.vicon_subscriptions:
            return

        subscription = self.create_subscription(
            TransformStamped,
            topic,
            lambda message, marker=label: self.on_vicon_transform(marker, message,),
            self.vicon_qos,
        )

        self.vicon_subscriptions[label] = subscription
        self.vicon_histories[label] = PoseHistory(self.history_duration_ns)
        self.get_logger().info(f"Vicon marker {label!r}: " f"{topic} [TransformStamped]")


    def on_mpose(self, label, message,):
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
            self.get_logger().warning(f"Vicon parent frame changed " f"for {label!r}: " f"{existing_frame!r} -> " f"{parent_frame!r}")
            return

        self.vicon_parent_frames[label] = parent_frame
        self.vicon_histories[label].add(
            timestamp_ns,
            matrix,
        )
        # A newer Vicon sample may now allow interpolation at
        # a previously received MegaPose image timestamp.
        self.compare_available()


    def on_vicon_transform(
        self,
        label,
        message,
    ):
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


    def on_vicon_pose(
        self,
        label,
        message,
    ):
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


    def arena_label(self):
        if self.configured_arena:
            return self.configured_arena

        candidates = {label for label in (set(self.latest_mpose) | set(self.vicon_histories)) if self.arena_pattern.fullmatch(label)}

        if len(candidates) == 1:
            return next(iter(candidates))

        if len(candidates) > 1 and not self.discovery_warning_shown:
            self.get_logger().warning("Multiple arena markers matched. " "Set arena_marker_name explicitly.")

            self.discovery_warning_shown = True
        return None


    def compare_available(self):
        arena_label = self.arena_label()

        if arena_label is None:
            return

        arena_estimate = self.latest_mpose.get(arena_label)
        arena_history = self.vicon_histories.get(arena_label)

        if arena_estimate is None or arena_history is None:
            return

        (
            arena_timestamp,
            camera_frame,
            camera_T_arena,
        ) = arena_estimate

        for bot_label, bot_estimate in self.latest_mpose.items():
            if not self.bot_pattern.fullmatch(bot_label):
                continue

            bot_history = self.vicon_histories.get(bot_label)

            if bot_history is None:
                continue

            (
                bot_timestamp,
                bot_camera_frame,
                camera_T_bot,
            ) = bot_estimate

            # Both MegaPose matrices must originate from the
            # same source image.
            if bot_timestamp != arena_timestamp:
                continue

            if bot_camera_frame != camera_frame:
                continue

            if bot_timestamp <= self.last_evaluated.get(
                bot_label,
                -1,
            ):
                continue

            arena_parent = self.vicon_parent_frames.get(arena_label)

            bot_parent = self.vicon_parent_frames.get(bot_label)

            if arena_parent is None or bot_parent is None or arena_parent != bot_parent:
                continue

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
                # Wait for newer Vicon samples.
                continue

            except ValueError as error:
                self.get_logger().warning(f"Cannot match {bot_label!r} " f"at {bot_timestamp}: {error}")

                self.last_evaluated[bot_label] = bot_timestamp

                continue

            estimated_arena_T_bot = relative_pose(
                camera_T_arena,
                camera_T_bot,
            )

            ground_truth_arena_T_bot = relative_pose(
                world_T_arena,
                world_T_bot,
            )

            translation_delta = estimated_arena_T_bot[:3, 3] - ground_truth_arena_T_bot[:3, 3]

            translation_error_m = float(np.linalg.norm(translation_delta))

            rotation_error_deg = rotation_error_degrees(
                estimated_arena_T_bot,
                ground_truth_arena_T_bot,
            )

            self.last_evaluated[bot_label] = bot_timestamp

            self.get_logger().info(
                f"{bot_label} wrt "
                f"{arena_label}: "
                f"translation error="
                f"{translation_error_m * 100.0:.2f} cm, "
                f"rotation error="
                f"{rotation_error_deg:.2f} deg"
            )


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
