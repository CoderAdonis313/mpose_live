#!/usr/bin/env python3
"""Drive through waypoints expressed in the arena marker frame."""

from math import atan2, cos, hypot, isfinite, radians, sin
from time import monotonic

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from tf_transformations import euler_from_quaternion


class DriverPIDNode(Node):
    def __init__(self):
        super().__init__("driver_pid_node")

        est_pose_topic = str(self.declare_parameter("estimated_pose_topic", "").value)
        gt_pose_topic = str(self.declare_parameter("ground_truth_pose_topic", "").value)
        control_mode = str(self.declare_parameter("control_mode", "").value)
        pose_topic = gt_pose_topic if control_mode == 'vicon' else est_pose_topic

        assert control_mode in ['vicon', 'mpose']

        drive_topic = str(self.declare_parameter("drive_topic", "/cmd_vel").value)
        self.rate = float(self.declare_parameter("pose_rate", 15).value)    #type: ignore
        self.pose_timeout = float(self.declare_parameter("pose_timeout_s", 1.0).value)  #type: ignore
        self.arena_frame = str(self.declare_parameter("arena_frame", "arena1_marker").value)

        # Angle from marker +X to robot forward, about marker +Z.
        self.yaw_offset = float(self.declare_parameter("heading_offset_rad", 0.0).value)    #type: ignore

        if not all(isfinite(v) and v > 0
                   for v in (self.rate, self.pose_timeout)):
            raise ValueError("pose_rate and pose_timeout_s must be positive")
        if not isfinite(self.yaw_offset):
            raise ValueError("heading_offset_rad must be finite")

        # Metres in the arena frame; visit in this order.
        self.goals = [
            (0.5, 0.5),
            (0.5, -0.5),
            (-0.5, -0.5),
            (-0.5, 0.5),
            (0.5, 0.5)
        ]
        self.goal_idx = 0
        self.bot_loc = (0, 0)
        self.bot_yaw = 0.0
        self.last_pose_time = None

        self.dist_tolerance = float(self.declare_parameter('dist_tolerance', '0.1').value)  # type: ignore
        self.heading_tolerance = radians(float(self.declare_parameter('angle_tolerance', '10').value))  # type: ignore
        self.max_linear_speed = float(self.declare_parameter('max_lin_speed', '0.1').value)   # m/s # type: ignore
        self.max_angular_speed = float(float(self.declare_parameter('max_ang_speed', '0.1').value)) # rad/s # type: ignore
        self.linear_kp = 0.5
        self.angular_kp = 2.0
        self.shutdown_signal = False

        self.pub_ = self.create_publisher(Twist, drive_topic, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.listen_pose, 1)
        self.control_timer_ = self.create_timer(
            1.0 / self.rate, self.control_loop)


    def publish_stop(self):
        self.pub_.publish(Twist())


    def listen_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        norm = hypot(q.x, q.y, q.z, q.w)

        stamp = msg.header.stamp
        pose_time = stamp.sec + stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - pose_time

        if (msg.header.frame_id != self.arena_frame
                or not all(isfinite(v) for v in values)
                or norm < 1e-9
                or not -0.1 <= age <= self.pose_timeout):
            self.last_pose_time = None
            self.publish_stop()
            return

        _, _, yaw = euler_from_quaternion(
            [q.x / norm, q.y / norm, q.z / norm, q.w / norm])

        self.bot_loc = (p.x, p.y)
        self.bot_yaw = yaw + self.yaw_offset
        self.last_pose_time = monotonic()


    def control_loop(self):
        if self.shutdown_signal:
            self.publish_stop()
            return

        if (self.last_pose_time is None or monotonic() - self.last_pose_time > self.pose_timeout):
            self.publish_stop()
            return

        if self.goal_idx >= len(self.goals):
            self.publish_stop()
            self.shutdown_signal = True
            self.control_timer_.cancel()
            return

        goal = self.goals[self.goal_idx]
        dx = goal[0] - self.bot_loc[0]
        dy = goal[1] - self.bot_loc[1]
        distance = hypot(dx, dy)

        # Check arrival first: waypoints do not specify final orientation.
        if distance <= self.dist_tolerance:
            self.publish_stop()
            self.get_logger().info(
                f"Reached waypoint {self.goal_idx + 1}: {goal}")
            self.goal_idx += 1

            if self.goal_idx == len(self.goals):
                self.shutdown_signal = True
                self.control_timer_.cancel()
                self.get_logger().info("Reached all waypoints")
            return

        error = atan2(dy, dx) - self.bot_yaw
        error = atan2(sin(error), cos(error))  # Signed shortest turn.

        command = Twist()
        command.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.angular_kp * error))

        # Turn in place until aligned; keep steering while moving.
        if abs(error) <= self.heading_tolerance:
            command.linear.x = min(
                self.max_linear_speed, self.linear_kp * distance)

        self.pub_.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = DriverPIDNode()

    try:
        while rclpy.ok() and not node.shutdown_signal:
            rclpy.spin_once(node, timeout_sec=1.0 / node.rate)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()