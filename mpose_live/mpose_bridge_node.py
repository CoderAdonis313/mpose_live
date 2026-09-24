#!/usr/bin/env python3
"""Receive MegaPose UDP packets and publish ROS poses and stream status."""

import json
import math
import socket
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation


STATUSES = {
    "STARTING",
    "TRACKING",
    "NO_TARGET",
    "NO_POSE",
    "RESET",
    "ERROR",
    "STOPPED",
}


def set_stamp(header, ns, frame):
    header.stamp.sec, header.stamp.nanosec = divmod(int(ns), 1_000_000_000)
    header.frame_id = frame


def checked_transform(value):
    matrix = np.asarray(value, dtype=float)

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 transform")

    rot = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-3)
    ):
        raise ValueError("Invalid rigid transform")

    return matrix


def pose_message(matrix, ns, frame):
    msg = PoseStamped()
    set_stamp(msg.header, ns, frame)

    p = matrix[:3, 3]
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()

    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
    (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ) = map(float, q)

    return msg


def diagnostic(pub, name, message, level, ns, **values):
    msg = DiagnosticArray()
    set_stamp(msg.header, ns, "")

    status = DiagnosticStatus()
    status.name = name
    status.message = message
    status.level = level
    status.values = [
        KeyValue(key=str(k), value=str(v)) for k, v in values.items()
    ]

    msg.status = [status]
    pub.publish(msg)


def positive(node, name, default):
    value = float(node.declare_parameter(name, default).value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class PacketGate:
    """Validate packets before updating ordering state."""

    def __init__(self, frame, object_id):
        self.frame = frame
        self.object_id = object_id
        self.session = None
        self.seq = -1
        self.result_ns = 0
        self.image_ns = 0

    def accept(self, data, now_ns, packet_age_ns):
        p = json.loads(data)

        if not isinstance(p, dict) or type(p.get("schema_version")) is not int:
            raise ValueError("Missing packet schema")

        if p["schema_version"] != 1:
            raise ValueError("Unsupported packet schema")

        if (
            p.get("frame_id") != self.frame
            or p.get("object_id") != self.object_id
            or p.get("timestamp_source") != "host_read"
        ):
            raise ValueError("Unexpected frame, object, or timestamp source")

        session = p.get("session_id")
        seq = p.get("packet_seq")
        result_ns = p.get("result_time_ns")
        state = p.get("status")

        if not isinstance(session, str) or not session:
            raise ValueError("Missing session_id")

        if type(seq) is not int or seq < 0 or type(result_ns) is not int:
            raise ValueError("Invalid sequence or result timestamp")

        if not isinstance(state, str) or state not in STATUSES:
            raise ValueError("Unknown sender status")

        if type(p.get("valid")) is not bool or p["valid"] != (state == "TRACKING"):
            raise ValueError("Inconsistent validity flag")

        if (
            result_ns <= 0
            or result_ns > now_ns + 50_000_000
            or now_ns - result_ns > packet_age_ns
        ):
            raise ValueError("Old packet or incompatible clock")

        same_session = session == self.session
        if (
            (same_session and seq <= self.seq)
            or result_ns < self.result_ns
            or (not same_session and result_ns <= self.result_ns)
        ):
            raise ValueError("Duplicate or out-of-order packet")

        matrix = None

        if state in {"TRACKING", "NO_TARGET", "NO_POSE"}:
            image_ns = p.get("image_time_ns")
            index = p.get("frame_index")

            if (
                type(image_ns) is not int
                or image_ns <= 0
                or image_ns > result_ns
                or type(index) is not int
                or index < 0
            ):
                raise ValueError("Invalid source frame timestamp or index")

            if same_session and image_ns <= self.image_ns:
                raise ValueError("Non-increasing image timestamp")

        elif (
            p.get("image_time_ns") is not None
            or p.get("frame_index") is not None
        ):
            raise ValueError("Non-frame status must not contain an image timestamp")

        if p["valid"]:
            matrix = checked_transform(p.get("T_camera_object"))
            if matrix[2, 3] <= 0:
                raise ValueError("Object is behind camera")
        elif p.get("T_camera_object") is not None:
            raise ValueError("Invalid result contains a pose")

        self.session = session
        self.seq = seq
        self.result_ns = result_ns

        if not same_session:
            self.image_ns = 0

        if state in {"TRACKING", "NO_TARGET", "NO_POSE"}:
            self.image_ns = p["image_time_ns"]

        return p, matrix


class UDPBridgeNode(Node):
    def __init__(self):
        super().__init__("mpose_bridge_node")

        if self.get_parameter("use_sim_time").value:
            raise ValueError("Host timestamp packets require use_sim_time=false")

        frame = self.declare_parameter(
            "camera_frame", "zed_left_camera_optical_frame"
        ).value
        object_id = self.declare_parameter("object_id", "fiducial").value
        port = self.declare_parameter("udp_port", 5005).value

        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid UDP port")

        self.timeout = positive(self, "stream_timeout_s", 2.0)
        self.max_age_ns = int(positive(self, "max_pose_age_s", 5.0) * 1e9)
        self.gate = PacketGate(frame, object_id)

        self.pose_pub = self.create_publisher(
            PoseStamped, "/megapose/object_pose", 100
        )
        self.status_pub = self.create_publisher(
            DiagnosticArray, "/megapose/status", 10
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("127.0.0.1", port))
            self.sock.setblocking(False)
        except Exception:
            self.sock.close()
            raise

        self.last_receive = None
        self.last_packet = None
        self.state = "WAITING"
        self.rejected = 0

        self.create_timer(0.01, self.receive)
        self.create_timer(0.2, self.watchdog)

        self.get_logger().info(f"Listening on 127.0.0.1:{port}")

    def receive(self):
        # Bound work so other callbacks can run.
        for _ in range(100):
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break

            try:
                packet, matrix = self.gate.accept(
                    data, time.time_ns(), int(self.timeout * 1e9)
                )
            except (ValueError, TypeError, OverflowError, RecursionError) as error:
                self.rejected += 1
                if self.rejected % 100 == 1:
                    self.get_logger().warning(f"Rejected UDP packet: {error}")
                continue

            self.last_receive = time.monotonic()
            self.last_packet = packet
            self.state = packet["status"]

            if matrix is not None:
                if time.time_ns() - packet["image_time_ns"] > self.max_age_ns:
                    self.state = "STALE_POSE"
                else:
                    self.pose_pub.publish(
                        pose_message(
                            matrix,
                            packet["image_time_ns"],
                            self.gate.frame,
                        )
                    )

            self.publish_status()

    def watchdog(self):
        if self.last_receive is not None:
            if time.monotonic() - self.last_receive > self.timeout:
                self.state = "STREAM_TIMEOUT"
            elif (
                self.state == "TRACKING"
                and time.time_ns() - self.last_packet["image_time_ns"]
                > self.max_age_ns
            ):
                self.state = "STALE_POSE"

        self.publish_status()

    def publish_status(self):
        packet = self.last_packet or {}
        image_ns = packet.get("image_time_ns")
        age = (time.time_ns() - image_ns) / 1e9 if image_ns else None

        diagnostic(
            self.status_pub,
            "megapose_stream",
            self.state,
            DiagnosticStatus.OK
            if self.state == "TRACKING"
            else DiagnosticStatus.WARN,
            time.time_ns(),
            valid=self.state == "TRACKING",
            session_id=self.gate.session,
            packet_seq=self.gate.seq,
            image_time_ns=image_ns,
            pose_age_s=age,
            rejected_packets=self.rejected,
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = UDPBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.sock.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()