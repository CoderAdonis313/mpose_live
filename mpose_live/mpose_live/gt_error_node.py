#!/usr/bin/env python3
"""Compare MegaPose with Vicon ground truth at the image timestamp."""

import math
import time
from bisect import bisect_left
from collections import deque

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp


# Existing metrics_postprocess.py assumptions:
# camera optical frame expressed in the tracked camera-body frame.
CAMERA_AXES = np.array([
    [-1.0,  0.0,  0.0, 0.0],
    [ 0.0,  0.0, -1.0, 0.0],
    [ 0.0, -1.0,  0.0, 0.0],
    [ 0.0,  0.0,  0.0, 1.0],
])


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


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


def from_pose(position, orientation):
    p = np.array([position.x, position.y, position.z])
    q = np.array([
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ])

    if (
        not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or np.linalg.norm(q) < 1e-12
    ):
        raise ValueError("Invalid position or quaternion")

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q).as_matrix()
    matrix[:3, 3] = p
    return matrix


def pose_message(matrix, ns, frame):
    msg = PoseStamped()
    set_stamp(msg.header, ns, frame)

    p = matrix[:3, 3]
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()  #type: ignore

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


class PoseHistory:
    """Keep timestamped transforms and interpolate within their coverage."""

    def __init__(self, duration_ns):
        self.duration_ns = duration_ns
        self.stamps = []
        self.matrices = []

    def add(self, ns, matrix):
        if self.stamps and ns <= self.stamps[-1]:
            return

        self.stamps.append(ns)
        self.matrices.append(matrix)

        cut = max(
            0,
            bisect_left(self.stamps, ns - self.duration_ns) - 1,
            len(self.stamps) - 10000,
        )

        if cut:
            del self.stamps[:cut]
            del self.matrices[:cut]

    def at(self, ns, max_gap_ns):
        i = bisect_left(self.stamps, ns)

        if i < len(self.stamps) and self.stamps[i] == ns:
            return self.matrices[i]

        if i == len(self.stamps):
            raise LookupError("Waiting for Vicon samples after the image")

        if i == 0:
            raise ValueError("Image is older than Vicon history")

        t0, t1 = self.stamps[i - 1:i + 1]
        if t1 - t0 > max_gap_ns:
            raise ValueError("Vicon interpolation gap is too large")

        a, b = self.matrices[i - 1:i + 1]
        alpha = (ns - t0) / (t1 - t0)

        matrix = np.eye(4)
        matrix[:3, 3] = (1 - alpha) * a[:3, 3] + alpha * b[:3, 3]
        matrix[:3, :3] = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])),
        )(alpha).as_matrix()

        return matrix


class GTErrorNode(Node):
    def __init__(self):
        super().__init__("gt_error_node")

        if self.get_parameter("use_sim_time").value:
            raise ValueError("Live host timestamps require use_sim_time=false")

        self.frame = self.declare_parameter("camera_frame", "zed_left_camera_optical_frame").value
        self.ready = self.declare_parameter("evaluation_enabled", True).value
        self.preliminary = self.declare_parameter("preliminary", True).value
        bot_topic = self.declare_parameter('robot_vicon_topic', "").value
        cam_topic = self.declare_parameter('cam_vicon_topic', "").value
        mpose_pose_topic = self.declare_parameter('mpose_pose_topic', "").value
        gt_pose_topic = self.declare_parameter('gt_pose_topic', "").value
        mpose_error_topic = self.declare_parameter('mpose_error_topic', "").value
        mpose_terror_topic = self.declare_parameter('mpose_terror_topic', "").value


        self.T_vc_c = checked_transform(
            np.array(
                self.declare_parameter(
                    "camera_body_to_optical",
                    CAMERA_AXES.ravel().tolist(),
                ).value
            ).reshape(4, 4)
        )
        self.T_vo_o = checked_transform(
            np.array(
                self.declare_parameter(
                    "marker_body_to_mesh",
                    np.eye(4).ravel().tolist(),
                ).value
            ).reshape(4, 4)
        )

        history_ns = int(positive(self, "history_s", 20.0) * 1e9)
        self.gap_ns = int(positive(self, "max_vicon_gap_s", 0.05) * 1e9)
        self.wait_s = positive(self, "sync_wait_s", 1.0)

        self.history = {
            name: PoseHistory(history_ns) for name in ("robot", "cam")
        }
        self.frames = {}
        self.frame_fault = False
        self.pending = deque()
        self.last_estimate = None
        self.last_log = 0.0
        self.matched = 0
        self.missed = 0

        self.gt_pub = self.create_publisher(msg_type=PoseStamped, topic=gt_pose_topic, qos_profile=100)  #type: ignore
        self.error_pub = self.create_publisher(msg_type=DiagnosticArray, topic=mpose_error_topic, qos_profile=10)   #type: ignore
        self.delta_pub = self.create_publisher(msg_type=Vector3Stamped, topic=mpose_terror_topic, qos_profile=10)    #type: ignore

        qos = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(
            msg_type=PoseStamped,
            topic=bot_topic,   #type: ignore
            callback=lambda msg: self.on_vicon("robot", msg),
            qos_profile=qos)
        
        self.create_subscription(
            msg_type=PoseStamped,
            topic=cam_topic,   #type: ignore
            callback=lambda msg: self.on_vicon("cam", msg),
            qos_profile=qos)
        
        self.create_subscription(msg_type=PoseStamped, topic=mpose_pose_topic, callback=self.on_estimate, qos_profile=100)    #type: ignore

        self.create_timer(0.02, self.process)
        self.create_timer(1.0, self.idle_status)

        if self.preliminary:
            self.get_logger().warning(
                "Using metrics_postprocess.py frame assumptions; "
                "errors are preliminary."
            )


    def report(self, state, ns=None, **values):
        diagnostic(
            self.error_pub,
            "megapose_vs_vicon",
            state,
            DiagnosticStatus.OK
            if state == "MATCHED"
            else DiagnosticStatus.WARN,
            time.time_ns() if ns is None else ns,
            matched=self.matched,
            missed=self.missed,
            preliminary=self.preliminary,
            **values,
        )


    def idle_status(self):
        if not self.ready:
            self.report("EVALUATION_DISABLED")
        elif self.frame_fault:
            self.report("VICON_FRAME_MISMATCH")
        elif (
            self.last_estimate is None
            or time.monotonic() - self.last_estimate > 2.0
        ):
            self.report("WAITING_FOR_ESTIMATE")


    def on_vicon(self, name, msg):
        ns = stamp_ns(msg.header.stamp)
        frame = msg.header.frame_id

        if ns <= 0 or not frame or self.frame_fault:
            return

        if name in self.frames and self.frames[name] != frame:
            self.frame_fault = True
            self.report("VICON_FRAME_MISMATCH")
            return

        try:
            matrix = from_pose(
                msg.pose.position,
                msg.pose.orientation,
            )
        except ValueError:
            return

        self.frames[name] = frame

        if (
            len(self.frames) == 2
            and self.frames["cam"] != self.frames["robot"]
        ):
            self.frame_fault = True
            self.report("VICON_FRAME_MISMATCH")
            return
        self.history[name].add(ns, matrix)


    def on_estimate(self, msg):
        self.last_estimate = time.monotonic()

        if not self.ready or self.frame_fault:
            return

        ns = stamp_ns(msg.header.stamp)
        if ns <= 0 or msg.header.frame_id != self.frame:
            self.report("INVALID_ESTIMATE_FRAME")
            return

        try:
            matrix = from_pose(msg.pose.position, msg.pose.orientation)
        except ValueError:
            self.report("INVALID_ESTIMATE")
            return

        if len(self.pending) >= 200:
            self.pending.popleft()
            self.missed += 1

        self.pending.append((ns, matrix, time.monotonic()))


    def process(self):
        if self.frame_fault:
            self.pending.clear()
            return

        while self.pending:
            ns, estimate, received = self.pending[0]

            try:
                camera = self.history["cam"].at(ns, self.gap_ns)
                marker = self.history["robot"].at(ns, self.gap_ns)
            except LookupError as error:
                if time.monotonic() - received < self.wait_s:
                    break

                self.pending.popleft()
                self.missed += 1
                self.report("VICON_TIMEOUT", ns, reason=error)
                continue
            except ValueError as error:
                self.pending.popleft()
                self.missed += 1
                self.report("UNMATCHED", ns, reason=error)
                continue

            self.pending.popleft()

            # Object/mesh frame expressed in the camera optical frame.
            # Same operation as your convert_to_cvframe(camera):
            T_world_camera_optical = camera @ self.T_vc_c

            # Express the tracked object's mesh pose in the camera optical frame:
            T_world_object = marker @ self.T_vo_o
            gt = np.linalg.inv(T_world_camera_optical) @ T_world_object

            delta = estimate[:3, 3] - gt[:3, 3]
            position_error = float(np.linalg.norm(delta))
            rotation_error = float(
                np.degrees(
                    Rotation.from_matrix(
                        gt[:3, :3].T @ estimate[:3, :3]
                    ).magnitude()
                )
            )

            self.gt_pub.publish(pose_message(gt, ns, self.frame))

            msg = Vector3Stamped()
            set_stamp(msg.header, ns, self.frame)
            msg.vector.x, msg.vector.y, msg.vector.z = map(float, delta)
            self.delta_pub.publish(msg)

            self.matched += 1
            self.report(
                "MATCHED",
                ns,
                position_error_m=position_error,
                rotation_error_deg=rotation_error,
                estimate_age_s=(time.time_ns() - ns) / 1e9,
            )

            if time.monotonic() - self.last_log >= 1.0:
                self.get_logger().info(
                    f"Position error: {position_error * 100:.2f} cm; "
                    f"rotation error: {rotation_error:.2f} deg"
                )
                self.last_log = time.monotonic()


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