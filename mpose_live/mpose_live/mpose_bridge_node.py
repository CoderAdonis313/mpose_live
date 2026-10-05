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
from mpose_live.utils import PacketGate


def set_stamp(header, ns, frame):
    header.stamp.sec, header.stamp.nanosec = divmod(int(ns), 1_000_000_000)
    header.frame_id = frame


def pose_message(matrix, ns, frame):
    msg = PoseStamped()
    set_stamp(msg.header, ns, frame)

    p = matrix[:3, 3]
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat() #type: ignore

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



class MPoseBridgeNode(Node):
    def __init__(self):
        super().__init__("mpose_bridge_node")

        if self.get_parameter("use_sim_time").value:
            raise ValueError("Host timestamp packets require use_sim_time=false")

        frame = self.declare_parameter("camera_frame", "zed_left_camera_optical_frame").value
        object_id = self.declare_parameter("object_id", "fiducial").value
        mpose_pose_topic = self.declare_parameter('mpose_pose_topic', "").value
        mpose_status_topic = self.declare_parameter('mpose_status_topic', "").value
        port = self.declare_parameter("udp_port", 5005).value

        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid UDP port")

        self.timeout = positive(self, "stream_timeout_s", 2.0)
        self.max_age_ns = int(positive(self, "max_pose_age_s", 5.0) * 1e9)
        self.gate = PacketGate(frame, object_id)

        self.pose_pub = self.create_publisher(msg_type=PoseStamped, topic=mpose_pose_topic, qos_profile=100)  #type: ignore
        self.status_pub = self.create_publisher(msg_type=DiagnosticArray, topic=mpose_status_topic, qos_profile=10)   #type: ignore

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
                and time.time_ns() - self.last_packet["image_time_ns"] #type: ignore
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
        node = MPoseBridgeNode()
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