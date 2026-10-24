#!/usr/bin/env python3
"""Publish every visible bot pose relative to every visible arena marker."""

import math
import re
from collections import defaultdict
from functools import partial

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation

MESSAGE_TYPES = {
    "PoseStamped": PoseStamped,
    "TransformStamped": TransformStamped,
}

BOT_MAP = {
    "bot1_marker": "SS_TB3_1",
    "bot2_marker": "SS_TB3_2",
    "bot3_marker": "SS_TB3_3",
    "bot4_marker": "SS_TB3_4"
}


def pose_to_matrix(message):
    """Convert a supported pose message into a 4x4 transform."""
    if isinstance(message, PoseStamped):
        position = message.pose.position
        orientation = message.pose.orientation
    elif isinstance(message, TransformStamped):
        position = message.transform.translation
        orientation = message.transform.rotation
    else:
        raise ValueError(f"Unsupported message type: {type(message).__name__}")

    translation = np.array(
        [position.x, position.y, position.z],
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

    if not np.isfinite(translation).all():
        raise ValueError("Position contains NaN or infinity")

    if not np.isfinite(quaternion).all():
        raise ValueError("Quaternion contains NaN or infinity")

    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm < 1e-9:
        raise ValueError("Quaternion has zero length")

    quaternion /= quaternion_norm

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = translation
    return transform


def matrix_to_transform(matrix, stamp, arena_label, bot_label):
    """Convert a 4x4 relative transform into PoseStamped."""
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat() # type: ignore

    message = TransformStamped()
    message.header.stamp = stamp
    message.header.frame_id = arena_label
    message.child_frame_id = bot_label

    message.transform.translation.x = float(matrix[0, 3])
    message.transform.translation.y = float(matrix[1, 3])
    message.transform.translation.z = float(matrix[2, 3])

    message.transform.rotation.x = float(quaternion[0])
    message.transform.rotation.y = float(quaternion[1])
    message.transform.rotation.z = float(quaternion[2])
    message.transform.rotation.w = float(quaternion[3])

    return message


class RelativePoseNode(Node):
    """Calculate bot poses relative to every discovered arena marker."""

    def __init__(self):
        super().__init__("relative_pose_node")

        self.input_prefix = str(self.declare_parameter("input_topic_prefix", "/mpose/poses",).value).rstrip("/")
        self.output_prefix = str(self.declare_parameter("output_topic_prefix","/relative_pose",).value).rstrip("/")

        input_type_name = str(self.declare_parameter("input_message_type","PoseStamped",).value)

        if input_type_name not in MESSAGE_TYPES:
            raise ValueError("input_message_type must be " "'PoseStamped' or 'TransformStamped'")

        self.input_message_type = MESSAGE_TYPES[input_type_name]
        self.input_ros_type = f"geometry_msgs/msg/{input_type_name}"

        self.bot_pattern = self.compile_pattern("bot_marker_pattern", r"^bot.*_marker$",)
        self.arena_pattern = self.compile_pattern("arena_marker_pattern",r"^arena.*_marker$",)

        self.sync_tolerance = float(self.declare_parameter("sync_tolerance_s",0.02,).value) # type: ignore
        discovery_period = float(self.declare_parameter("discovery_period_s",1.0,).value) # type: ignore

        self.remap = self.declare_parameter('remap', False).value

        for name, value in (
            ("sync_tolerance_s", self.sync_tolerance),
            ("discovery_period_s", discovery_period),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

        if not self.input_prefix:
            raise ValueError("input_topic_prefix must not be empty")

        if not self.output_prefix:
            raise ValueError("output_topic_prefix must not be empty")

        # A best-effort subscriber is compatible with best-effort and
        # reliable publishers.
        self.input_qos = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # topic -> message_filters.Subscriber
        self.pose_subscribers = {}

        # (arena topic, bot topic) -> synchronizer
        self.pose_synchronizers = {}

        # (arena label, bot label) -> publisher
        self.pose_publishers = {}    #type: ignore

        # Pairs currently present in the ROS graph.
        self.active_pairs = set()

        # Prevent repeated discovery warnings.
        self.warned = set()

        self.get_logger().info(f"Reading {input_type_name} topics below " f"{self.input_prefix}/")
        self.get_logger().info(f"Publishing below " f"{self.output_prefix}/<bot>/<bot>")

        self.discovery_timer = self.create_timer(
            discovery_period,
            self.discover_markers,
        )
        self.discover_markers()


    def compile_pattern(self, parameter_name, default):
        value = str(
            self.declare_parameter(
                parameter_name,
                default,
            ).value
        )

        try:
            return re.compile(value, re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"Invalid {parameter_name}: {error}") from error


    @staticmethod
    def marker_label(topic):
        """Return the final component of a topic name."""
        return topic.rstrip("/").split("/")[-1].lower()


    def warn_once(self, key, message):
        if key in self.warned:
            return

        self.warned.add(key)
        self.get_logger().warning(message)


    def discover_markers(self):
        """Discover markers and create every arena/bot pairing."""
        arenas = defaultdict(list)
        bots = defaultdict(list)

        for topic, topic_types in self.get_topic_names_and_types():
            if not topic.startswith(self.input_prefix + "/"):
                continue

            label = self.marker_label(topic)

            is_arena = bool(self.arena_pattern.fullmatch(label))
            is_bot = bool(self.bot_pattern.fullmatch(label))

            if not is_arena and not is_bot:
                continue

            if is_arena and is_bot:
                self.warn_once(
                    ("overlapping_patterns", label),
                    f"Skipping {topic}: label matches both patterns",
                )
                continue

            if self.input_ros_type not in topic_types:
                self.warn_once(
                    ("wrong_type", topic, tuple(topic_types)),
                    f"Skipping {topic}: expected " f"{self.input_ros_type}, found {topic_types}",
                )
                continue

            if is_arena:
                arenas[label].append(topic)
            else:
                bots[label].append(topic)

        arenas = self.unique_topics(arenas)
        bots = self.unique_topics(bots)

        if not arenas:
            self.warn_once(
                ("no_arena", self.input_prefix),
                f"No arena topics found below {self.input_prefix}/",
            )

        if not bots:
            self.warn_once(
                ("no_bots", self.input_prefix),
                f"No bot topics found below {self.input_prefix}/",
            )

        # Every visible bot is paired with every visible arena.
        self.active_pairs = {(arena_topic, bot_topic) for arena_topic in arenas.values() for bot_topic in bots.values()}

        for arena_label, arena_topic in arenas.items():
            for bot_label, bot_topic in bots.items():
                self.ensure_pair(
                    arena_label,
                    bot_label,
                    arena_topic,
                    bot_topic,
                )


    def unique_topics(self, topics_by_label):
        """Reject duplicate topic labels to prevent ambiguity."""
        unique = {}

        for label, topics in topics_by_label.items():
            if len(topics) == 1:
                unique[label] = topics[0]
                continue

            self.warn_once(
                ("duplicate_label", label, tuple(sorted(topics))),
                f"Skipping duplicate marker label " f"{label!r}: {sorted(topics)}",
            )

        return unique


    def get_subscriber(self, topic):
        """Create a subscriber once and reuse it between pairs."""
        if topic not in self.pose_subscribers:
            self.pose_subscribers[topic] = Subscriber(
                self,
                self.input_message_type,
                topic,
                qos_profile=self.input_qos,
            )

        return self.pose_subscribers[topic]


    def ensure_pair(
        self,
        arena_label,
        bot_label,
        arena_topic,
        bot_topic,
    ):
        """Create the synchronizer and publisher for one pair."""
        pair_key = (arena_topic, bot_topic)

        if pair_key in self.pose_synchronizers:
            return

        publisher_key = (arena_label, bot_label)
        output_bot_label = bot_label

        if publisher_key not in self.pose_publishers: # type: ignore
            output_topic = f"{self.output_prefix}/{bot_label}/{bot_label}"

            if self.remap:
                output_bot_label = BOT_MAP.get(bot_label)

                if output_bot_label is None:
                    self.warn_once(
                        ("missing_bot_mapping", bot_label),
                        f"No output mapping for {bot_label!r}",
                    )
                    return
                output_topic = f"{self.output_prefix}/{output_bot_label}/{output_bot_label}"

            self.pose_publishers[publisher_key] = self.create_publisher( # type: ignore
                TransformStamped,
                output_topic,
                10,
            )

            self.get_logger().info(f"{bot_label} relative to {arena_label} " f"-> {output_topic}")

        synchronizer = ApproximateTimeSynchronizer(
            [
                self.get_subscriber(arena_topic),
                self.get_subscriber(bot_topic),
            ],
            queue_size=30,
            slop=self.sync_tolerance,
        )

        synchronizer.registerCallback(
            partial(
                self.on_pair,
                pair_key,
                arena_label,
                bot_label,
                output_bot_label
            )
        )
        self.pose_synchronizers[pair_key] = synchronizer


    def on_pair(
        self,
        pair_key,
        arena_label,
        bot_label,
        output_bot_label,
        arena_message,
        bot_message,
    ):
        """Publish only when a new synchronized pair arrives."""
        if pair_key not in self.active_pairs:
            return

        try:
            self.validate_message(arena_message)
            self.validate_message(bot_message)

            if arena_message.header.frame_id != bot_message.header.frame_id:
                raise ValueError("Arena and bot have different parent frames")

            parent_t_arena = pose_to_matrix(arena_message)
            parent_t_bot = pose_to_matrix(bot_message)

            # Bot pose expressed in the arena frame.
            arena_t_bot = np.linalg.inv(parent_t_arena) @ parent_t_bot

            output = matrix_to_transform(
                arena_t_bot,
                bot_message.header.stamp,
                arena_label,
                output_bot_label
            )

            self.pose_publishers[(arena_label, bot_label)].publish(output) # type: ignore

        except (ValueError, np.linalg.LinAlgError) as error:
            self.get_logger().warning(f"Rejected {bot_label} relative to " f"{arena_label}: {error}")


    @staticmethod
    def validate_message(message):
        stamp = message.header.stamp
        timestamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

        if timestamp_ns <= 0:
            raise ValueError("Message has no valid timestamp")

        if not message.header.frame_id:
            raise ValueError("Message has no parent frame")


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RelativePoseNode()
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
