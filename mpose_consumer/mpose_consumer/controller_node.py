#!/usr/bin/env python3
"""Drive a TurtleBot through camera-frame MegaPose waypoints."""

import math
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


# Five default (x, z) points in zed_left_camera_optical_frame, metres.
DEFAULT_WAYPOINTS = [
    -0.16, 2.16,
    -0.05, 2.05,
    0.10, 2.05,
    0.20, 2.16,
    0.00, 2.20,
]


def clamp(value, limit):
    return max(-limit, min(limit, value))


def rotate_vector(vector, quaternion):
    """Rotate a three-vector by an xyzw quaternion."""
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)

    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid pose quaternion")

    x, y, z, w = (component / norm for component in quaternion)
    vx, vy, vz = vector

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def planar_heading_error(forward_x, forward_z, target_x, target_z):
    """Signed angle from the current heading to the target direction."""
    forward_norm = math.hypot(forward_x, forward_z)
    target_norm = math.hypot(target_x, target_z)

    if forward_norm < 1e-9 or target_norm < 1e-9:
        raise ValueError("degenerate planar direction")

    forward_x /= forward_norm
    forward_z /= forward_norm
    target_x /= target_norm
    target_z /= target_norm

    cross = forward_z * target_x - forward_x * target_z
    dot = forward_x * target_x + forward_z * target_z

    return math.atan2(cross, dot)


class TurtleBotControllerNode(Node):
    def __init__(self):
        super().__init__("turtlebot_controller_node")

        waypoint_values = self.declare_parameter("waypoints", DEFAULT_WAYPOINTS).value  #type: ignore

        if len(waypoint_values) % 2:    #type: ignore
            raise ValueError("waypoints must contain x, z pairs")

        if not all(math.isfinite(value) for value in waypoint_values):  #type: ignore
            raise ValueError("waypoints must contain finite values")

        self.waypoints = list(
            zip(waypoint_values[0::2], waypoint_values[1::2])   #type: ignore
        )

        forward_axis = self.declare_parameter(
            "object_forward_axis",
            [0.0, 1.0, 0.0],
        ).value

        if len(forward_axis) != 3:  #type: ignore
            raise ValueError("object_forward_axis must have three values")

        self.forward_axis = tuple(float(value) for value in forward_axis)   #type: ignore

        self.linear_gain = float(self.declare_parameter("linear_gain", 0.8).value)  #type: ignore
        self.angular_gain = float(self.declare_parameter("angular_gain", 1.8).value)    #type: ignore
        self.max_linear_speed = float(self.declare_parameter("max_linear_speed", 0.12).value)   #type: ignore
        self.max_angular_speed = float(self.declare_parameter("max_angular_speed", 0.6).value)  #type: ignore
        self.waypoint_tolerance = float(self.declare_parameter("waypoint_tolerance", 0.06).value)   #type: ignore
        self.turn_in_place_angle = float(self.declare_parameter("turn_in_place_angle", 0.45).value) #type: ignore
        self.pose_timeout = float(self.declare_parameter("pose_timeout_s", 0.25).value) #type: ignore
        self.max_pose_age = float(self.declare_parameter("max_pose_age_s", 0.25).value) #type: ignore
        self.angular_velocity_sign = float(self.declare_parameter("angular_velocity_sign", -1.0).value) #type: ignore

        mpose_pose_topic = self.declare_parameter("mpose_pose_topic", "").value
        mpose_status_topic = self.declare_parameter("mpose_status_topic", "").value
        cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel").value

        self.cmd_vel_pub = self.create_publisher(msg_type=Twist, topic=cmd_vel_topic, qos_profile=10) #type: ignore
        self.create_subscription(msg_type=PoseStamped, topic=mpose_pose_topic, callback=self.on_pose, qos_profile=10) #type: ignore
        self.create_subscription(msg_type=DiagnosticArray, topic=mpose_status_topic, callback=self.on_status, qos_profile=10) #type: ignore

        self.pose = None
        self.pose_received_at = None
        self.pose_frame = None
        self.tracking = False
        self.waypoint_index = 0
        self.last_stop_reason = None

        self.create_timer(0.05, self.control)

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} camera x-z waypoints"
        )


    def on_pose(self, msg):
        values = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        if not msg.header.frame_id or not all(map(math.isfinite, values)):
            self.pose = None
            self.stop("invalid pose")
            return

        if self.pose_frame is None:
            self.pose_frame = msg.header.frame_id
        elif msg.header.frame_id != self.pose_frame:
            self.pose = None
            self.stop("pose frame changed")
            return

        try:
            rotate_vector(self.forward_axis, values[3:])
        except ValueError as error:
            self.pose = None
            self.stop(str(error))
            return

        self.pose = msg
        self.pose_received_at = time.monotonic()


    def on_status(self, msg):
        states = [
            status.message
            for status in msg.status
            if status.name == "megapose_stream"
        ]

        if not states:
            return

        self.tracking = states[-1] == "TRACKING"

        if not self.tracking:
            self.stop(f"tracking state is {states[-1]}")


    def pose_is_fresh(self):
        if self.pose is None or self.pose_received_at is None:
            return False, "waiting for pose"

        if time.monotonic() - self.pose_received_at > self.pose_timeout:
            return False, "pose stream timed out"

        stamp = self.pose.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        age = (time.time_ns() - stamp_ns) / 1e9

        if stamp_ns <= 0:
            return False, "pose has no timestamp"

        if age > self.max_pose_age:
            return False, f"pose is {age:.3f} seconds old"

        if age < -0.05:
            return False, "pose timestamp is in the future"

        return True, None


    def stop(self, reason=None):
        self.cmd_vel_pub.publish(Twist())

        if reason is not None and reason != self.last_stop_reason:
            self.get_logger().warning(f"Stopping: {reason}")

        self.last_stop_reason = reason


    def control(self):
        if self.waypoint_index >= len(self.waypoints):
            self.stop()
            return

        if not self.tracking:
            self.stop("MegaPose is not tracking")
            return

        fresh, reason = self.pose_is_fresh()

        if not fresh:
            self.stop(reason)
            return

        position = self.pose.pose.position  #type: ignore
        target_x, target_z = self.waypoints[self.waypoint_index]

        dx = target_x - position.x
        dz = target_z - position.z
        distance = math.hypot(dx, dz)

        if distance <= self.waypoint_tolerance:
            self.waypoint_index += 1
            self.stop()

            self.get_logger().info(
                "Reached waypoint "
                f"{self.waypoint_index}/{len(self.waypoints)}"
            )

            if self.waypoint_index == len(self.waypoints):
                self.get_logger().info("All waypoints reached")

            return

        orientation = self.pose.pose.orientation    #type: ignore

        try:
            forward = rotate_vector(
                self.forward_axis,
                (
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ),
            )

            angle_error = planar_heading_error(
                forward[0],
                forward[2],
                dx,
                dz,
            )
        except ValueError as error:
            self.stop(str(error))
            return

        command = Twist()

        command.angular.z = clamp(
            self.angular_velocity_sign
            * self.angular_gain
            * angle_error,
            self.max_angular_speed,
        )

        if abs(angle_error) < self.turn_in_place_angle:
            command.linear.x = min(
                self.max_linear_speed,
                self.linear_gain * distance,
            ) * max(0.0, math.cos(angle_error))

        self.cmd_vel_pub.publish(command)
        self.last_stop_reason = None


def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()