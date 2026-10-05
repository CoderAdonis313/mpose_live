#!/usr/bin/env python3
"""Relay multi-marker MegaPose UDP packets into ROS 2."""

import json
import re
import socket

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import String


def set_stamp(header, timestamp_ns, frame_id):
    seconds, nanoseconds = divmod(
        int(timestamp_ns),
        1_000_000_000,
    )

    header.stamp.sec = seconds
    header.stamp.nanosec = nanoseconds
    header.frame_id = frame_id


def matrix_to_pose_message(matrix_value, timestamp_ns, frame_id):
    matrix = np.asarray(
        matrix_value,
        dtype=float,
    )

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 matrix")

    rotation = matrix[:3, :3]

    # These checks prevent scipy from silently correcting an
    # invalid rotation before it reaches the consumer.
    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-5,
    ):
        raise ValueError("Invalid homogeneous transform")

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-3,
    ):
        raise ValueError("Rotation is not orthonormal")

    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-3,
    ):
        raise ValueError("Rotation determinant is not one")

    quaternion = Rotation.from_matrix(rotation).as_quat() #type: ignore
    message = PoseStamped()

    set_stamp(
        message.header,
        timestamp_ns,
        frame_id,
    )

    (
        message.pose.position.x,
        message.pose.position.y,
        message.pose.position.z,
    ) = map(
        float,
        matrix[:3, 3],
    )

    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = map(
        float,
        quaternion,
    )
    return message


class MPoseBridgeNode(Node):
    def __init__(self):
        super().__init__("mpose_bridge_node")

        udp_port = int(self.declare_parameter("udp_port", 5005).value)  #type: ignore
        self.pose_topic_prefix = str(self.declare_parameter("pose_topic_prefix", "/mpose/poses").value).rstrip("/")
        packet_topic = str(self.declare_parameter("packet_topic", "/mpose/packets").value)
        self.max_packets_per_tick = int(self.declare_parameter("max_packets_per_tick", 100).value)  #type: ignore

        if type(udp_port) is not int or not 1 <= udp_port <= 65_535:
            raise ValueError("udp_port must be between " "1 and 65535")

        if type(self.max_packets_per_tick) is not int or self.max_packets_per_tick <= 0:
            raise ValueError("max_packets_per_tick must " "be positive")

        if not self.pose_topic_prefix:
            raise ValueError("pose_topic_prefix cannot be empty")

        # The original packet is sent to the consumer so it
        # can perform all comprehensive validation.
        self.packet_publisher = self.create_publisher(
            String,
            packet_topic,
            100,
        )

        # label -> PoseStamped publisher
        self.marker_publishers = {}

        # label -> source identity
        #
        # MegaPose sends the same completed result repeatedly
        # as a heartbeat. This prevents publishing the same
        # source-frame pose repeatedly.
        self.last_published_source = {}

        self.rejected_pose_messages = 0
        self.received_packets = 0

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,)

        try:
            self.socket.bind(("127.0.0.1", udp_port))
            self.socket.setblocking(False)

        except Exception:
            self.socket.close()
            raise

        self.receive_timer = self.create_timer(
            0.01,
            self.receive_packets,
        )

        self.get_logger().info(f"Listening on " f"127.0.0.1:{udp_port}")
        self.get_logger().info(f"Raw packets: {packet_topic}")
        self.get_logger().info(f"Marker topics: " f"{self.pose_topic_prefix}/<label>")


    def publisher_for_marker(self, label):
        publisher = self.marker_publishers.get(label)

        if publisher is not None:
            return publisher

        topic = f"{self.pose_topic_prefix}/{label}"
        publisher = self.create_publisher(
            PoseStamped,
            topic,
            10,
        )

        self.marker_publishers[label] = publisher
        self.get_logger().info(f"Created marker publisher: " f"{label!r} -> {topic}")
        return publisher


    def publish_raw_packet(self, packet_text):
        message = String()
        message.data = packet_text
        self.packet_publisher.publish(message)


    def process_for_pose_topics(self, packet):
        """Extract valid marker poses using minimal checks."""

        if not isinstance(packet, dict):
            return

        if packet.get("schema_version") != 1:
            return

        poses = packet.get("poses")

        if not isinstance(poses, dict):
            return

        frame_id = packet.get("frame_id")
        image_time_ns = packet.get("image_time_ns")

        if (
            not isinstance(frame_id, str)
            or not frame_id
            or type(image_time_ns) is not int
            or image_time_ns <= 0
        ):
            return

        session_id = packet.get("session_id")

        frame_index = packet.get("frame_index")

        # These fields identify one inference result. They are
        # only used to avoid repeatedly publishing a heartbeat.
        source_key = (
            session_id,
            frame_index,
            image_time_ns,
        )

        packet_valid = packet.get("valid") is True

        for label, pose_data in poses.items():
            # A marker label becomes part of a ROS topic name.
            if not isinstance(label, str):
                self.rejected_pose_messages += 1
                continue

            # Create the publisher for every safe marker label
            # received, even if this particular pose is stale.
            publisher = self.publisher_for_marker(label)

            if (
                not packet_valid
                or not isinstance(
                    pose_data,
                    dict,
                )
                or pose_data.get("valid") is not True
            ):
                continue

            if self.last_published_source.get(label) == source_key:
                continue

            try:
                message = matrix_to_pose_message(
                    pose_data.get("T_camera_object"),
                    image_time_ns,
                    frame_id,
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                self.rejected_pose_messages += 1

                if self.rejected_pose_messages % 100 == 1:
                    self.get_logger().warning(
                        f"Rejected pose for " f"{label!r}: {error}"
                    )

                continue

            publisher.publish(message)
            self.last_published_source[label] = source_key


    def receive_packets(self):
        """Drain queued UDP packets without blocking ROS."""

        for _ in range(self.max_packets_per_tick):
            try:
                raw_data, _ = self.socket.recvfrom(65_535)

            except BlockingIOError:
                break

            except OSError as error:
                self.get_logger().error(f"UDP receive failed: {error}")
                return

            try:
                packet_text = raw_data.decode("utf-8")

            except UnicodeDecodeError:
                self.rejected_pose_messages += 1
                continue

            # Forward the original packet unchanged. The
            # consumer performs comprehensive validation.
            self.publish_raw_packet(packet_text)

            self.received_packets += 1

            # Parsing here is only for creating per-marker
            # PoseStamped topics.
            try:
                packet = json.loads(packet_text)
            except ValueError:
                self.rejected_pose_messages += 1
                continue

            self.process_for_pose_topics(packet)


    def close(self):
        self.receive_timer.cancel()
        self.socket.close()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = MPoseBridgeNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
