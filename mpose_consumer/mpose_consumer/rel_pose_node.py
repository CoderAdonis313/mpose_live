#!/usr/bin/env python3
"""Publish each bot marker's pose relative to one discovered arena marker."""

import math
import re
from functools import partial
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation


def pose_to_matrix(message):
    if isinstance(message, PoseStamped):
        position = message.pose.position
        orientation = message.pose.orientation
    else:
        position = message.transform.translation
        orientation = message.transform.rotation

    translation = np.array([position.x, position.y, position.z], dtype=float)
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


def matrix_to_pose(matrix, stamp, frame_id):
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = frame_id

    message.pose.position.x = float(matrix[0, 3])
    message.pose.position.y = float(matrix[1, 3])
    message.pose.position.z = float(matrix[2, 3])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])
    return message


class RelativePoseNode(Node):
    def __init__(self):
        super().__init__("relative_pose_node")

        self.bot_pattern = re.compile(
            str(self.declare_parameter("bot_marker_pattern", r"^bot.*_marker$").value),
            re.IGNORECASE,
        )
        self.arena_pattern = re.compile(
            str(self.declare_parameter("arena_marker_pattern", r"^arena.*_marker$").value),
            re.IGNORECASE,
        )

        self.input_prefixes = {
            "estimated": str(self.declare_parameter("mpose_topic_prefix", "/mpose/poses").value).rstrip("/"),
            "ground_truth": str(self.declare_parameter("vicon_topic_prefix", "/vicon").value).rstrip("/"),
        }

        # These parameters are now prefixes; the bot label is appended.
        self.output_prefixes = {
            "estimated": str(self.declare_parameter("estimated_output_topic", "/relative_pose/estimated").value).rstrip("/"),
            "ground_truth": str(self.declare_parameter("ground_truth_output_topic", "/relative_pose/ground_truth").value).rstrip("/"),
        }

        vicon_type = str(self.declare_parameter("vicon_msg_type", "TransformStamped").value)
        if vicon_type not in ("PoseStamped", "TransformStamped"):
            raise ValueError("vicon_msg_type must be PoseStamped or TransformStamped")

        self.message_types = {
            "estimated": PoseStamped,
            "ground_truth": (PoseStamped if vicon_type == "PoseStamped" else TransformStamped),
        }
        self.ros_types = {
            "estimated": "geometry_msgs/msg/PoseStamped",
            "ground_truth": f"geometry_msgs/msg/{vicon_type}",
        }

        self.sync_tolerance = float(self.declare_parameter("sync_tolerance_s", 0.01).value)
        discovery_period = float(self.declare_parameter("discovery_period_s", 1.0).value)

        for name, value in (
            ("sync_tolerance_s", self.sync_tolerance),
            ("discovery_period_s", discovery_period),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.qos = {
            "estimated": QoSProfile(depth=30),
            "ground_truth": QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        }
        self.queue_sizes = {"estimated": 30, "ground_truth": 100}

        self.arena_name = None
        self.last_arena_candidates = None
        self.active_pairs = set()

        # Shared subscriptions: (source, topic) -> Subscriber.
        self.subscribers = {}

        # Independent synchronizers: (source, arena topic, bot topic).
        self.synchronizers = {}

        # (source, bot label) -> PoseStamped publisher.
        self.pose_publishers = {}

        self.discovery_timer = self.create_timer(discovery_period, self.discover_topics)
        self.discover_topics()


    @staticmethod
    def marker_name(topic):
        return topic.rstrip("/").split("/")[-1].lower()


    @staticmethod
    def under_prefix(topic, prefix):
        return topic.startswith(prefix + "/")


    def discover_topics(self):
        topics = {source: {} for source in self.input_prefixes}
        ambiguous = set()

        for topic, types in self.get_topic_names_and_types():
            label = self.marker_name(topic)
            is_arena = bool(self.arena_pattern.fullmatch(label))
            is_bot = bool(self.bot_pattern.fullmatch(label))

            if not (is_arena or is_bot):
                continue

            if is_arena and is_bot:
                self.get_logger().warning(f"Skipping {label!r}: matches both marker patterns")
                continue

            for source, prefix in self.input_prefixes.items():
                if not self.under_prefix(topic, prefix) or self.ros_types[source] not in types:
                    continue

                # Two distinct topics must not claim the same marker.
                previous = topics[source].get(label)
                if previous is not None and previous != topic:
                    ambiguous.add((source, label))

                topics[source][label] = topic

        for source, label in ambiguous:
            del topics[source][label]
            self.get_logger().warning(f"Skipping ambiguous {source} marker {label!r}")

        # Normalize names so ARENA1_MARKER and arena1_marker agree.
        arena_candidates = {label for markers in topics.values() for label in markers if self.arena_pattern.fullmatch(label)}

        self.active_pairs = set()

        if len(arena_candidates) != 1:
            self.arena_name = None

            if arena_candidates != self.last_arena_candidates:
                self.get_logger().warning("Expected one arena marker; found " f"{sorted(arena_candidates)}. Waiting for one arena.")

            self.last_arena_candidates = arena_candidates
            return

        self.last_arena_candidates = arena_candidates
        self.arena_name = next(iter(arena_candidates))

        for source, markers in topics.items():
            arena_topic = markers.get(self.arena_name)
            if arena_topic is None:
                continue

            for bot_label, bot_topic in markers.items():
                if not self.bot_pattern.fullmatch(bot_label):
                    continue

                pair_key = (source, arena_topic, bot_topic)
                self.active_pairs.add(pair_key)

                self.ensure_pair(
                    source,
                    self.arena_name,
                    bot_label,
                    arena_topic,
                    bot_topic,
                )


    def get_subscriber(self, source, topic):
        key = (source, topic)

        if key not in self.subscribers:
            self.subscribers[key] = Subscriber(
                self,
                self.message_types[source],
                topic,
                qos_profile=self.qos[source],
            )

        return self.subscribers[key]


    def ensure_pair(self, source, arena_label, bot_label, arena_topic, bot_topic):
        pair_key = (source, arena_topic, bot_topic)

        if pair_key in self.synchronizers:
            return

        publisher_key = (source, bot_label)

        if publisher_key not in self.pose_publishers:
            output_topic = f"{self.output_prefixes[source]}/{bot_label}"
            self.pose_publishers[publisher_key] = self.create_publisher(
                PoseStamped,
                output_topic,
                10,
            )
            self.get_logger().info(f"{source}: {bot_label} relative to {arena_label} " f"-> {output_topic}")

        # Each bot has its own queue. One absent bot cannot hold up others.
        # The shared arena subscription feeds every bot synchronizer.
        synchronizer = ApproximateTimeSynchronizer(
            [
                self.get_subscriber(source, arena_topic),
                self.get_subscriber(source, bot_topic),
            ],
            queue_size=self.queue_sizes[source],
            slop=self.sync_tolerance,
        )

        synchronizer.registerCallback(
            partial(
                self.on_pair,
                source,
                arena_label,
                bot_label,
                pair_key,
            )
        )
        self.synchronizers[pair_key] = synchronizer


    def on_pair(
        self,
        source,
        arena_label,
        bot_label,
        pair_key,
        arena_message,
        bot_message,
    ):
        if arena_label != self.arena_name or pair_key not in self.active_pairs:
            return

        try:
            for message in (arena_message, bot_message):
                stamp = message.header.stamp
                if stamp.sec * 1_000_000_000 + stamp.nanosec <= 0 or not message.header.frame_id:
                    raise ValueError("Missing timestamp or parent frame")

            if arena_message.header.frame_id != bot_message.header.frame_id:
                raise ValueError("Arena and bot have different parent frames")

            parent_T_arena = pose_to_matrix(arena_message)
            parent_T_bot = pose_to_matrix(bot_message)

            # Bot pose expressed in the arena marker frame.
            arena_T_bot = np.linalg.inv(parent_T_arena) @ parent_T_bot

            output = matrix_to_pose(
                arena_T_bot,
                bot_message.header.stamp,
                arena_label,
            )
            self.pose_publishers[(source, bot_label)].publish(output)

        except (ValueError, np.linalg.LinAlgError) as error:
            self.get_logger().warning(f"Rejected {source} pair for {bot_label}: {error}")


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
