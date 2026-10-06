#!/usr/bin/env python3
from rclpy.node import Node
import rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped, TransformStamped
from tf_transformations import euler_from_quaternion
from math import pi, atan2


class PID:
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.dt = 0.1
        self.sum = 0

    def compute(self, de):
        P = self.Kp * de
        self.sum = self.sum + de * self.dt
        I = self.Ki * self.sum
        D = self.Kd * (de / self.dt)

        return P + I + D


class DriverPIDNode(Node):
    def __init__(self):
        super().__init__(node_name="driver_pid_node")

        self.rel_pose_topic = str(self.declare_parameter('relative_pose_topic', '').value)
        self.drive_topic = str(self.declare_parameter('bot1_drive_topic', '').value)
        self.rate = int(self.declare_parameter('pose_rate', 1/15).value) # type: ignore

        self.create_subscription(
            msg_type=Odometry,
            topic=self.rel_pose_topic,
            qos_profile=10,
            callback=self.listen_imu,
        )

        self.pub_ = self.create_publisher(msg_type=TwistStamped, topic=self.drive_topic, qos_profile=10)

        self.control_timer_ = self.create_timer(1 / self.rate, self.control_loop)

        self.log_ = self.get_logger()
        self.log_.info("Created the node to drive bot")

        self.bot_yaw = 0
        self.bot_loc = (0, 0)
        self.shutdown_signal = False
        self.DIST_THRESHOLD = 0.05  # 5 cm tolerance
        self.ROT_THRESHOLD = 20 * (pi / 360)  # 10 degree tolerance
        self.LIN_SPEED = 0.1
        self.ANG_SPEED = 0.1

        # Run once method
        self.goal_idx = 0
        self.goals = self.read_goals()
        self.ang_pid = PID()
        self.dist_pid = PID()

    def read_goals(self):
        pkg_path = get_package_share_directory("goal_seek")
        goal_txt_path = f"{pkg_path}/config/goals.txt"
        goals = []

        with open(goal_txt_path, "r") as f:
            for line in f.readlines():
                x, y = line.split(" ")
                goals.append((float(x), float(y)))

        self.log_.info(f"Goals to move to: {goals}")
        return goals

    def listen_imu(self, msg: Odometry):
        quats = msg.pose.pose.orientation
        trans = msg.pose.pose.position

        self.bot_loc = (trans.x, trans.y)
        # xyzw quaternion order
        _, _, self.bot_yaw = euler_from_quaternion([quats.x, quats.y, quats.z, quats.w])

    def publish_stop(self):
        msg = TwistStamped()
        self.pub_.publish(msg)

    def calc_heading(self, goal):
        dy = goal[1] - self.bot_loc[1]
        dx = goal[0] - self.bot_loc[0]

        ang = atan2(dy, dx)
        return abs(ang - self.bot_yaw)

    def calc_dist(self, goal):
        dy = goal[1] - self.bot_loc[1]
        dx = goal[0] - self.bot_loc[0]

        return (dy**2 + dx**2) ** 0.5

    def control_loop(self):
        current_goal = self.goals[self.goal_idx]
        move_msg = TwistStamped()
        ang_error = self.calc_heading(current_goal)
        dist_error = self.calc_dist(current_goal)

        if ang_error > self.ROT_THRESHOLD:
            # tilt
            self.log_.info(f"Angle error: {ang_error}")
            ang_vel = self.ang_pid.compute(ang_error)
            move_msg.twist.angular.z = ang_vel
        elif dist_error > self.DIST_THRESHOLD:
            # move
            self.log_.info(f"Dist error: {dist_error}")
            lin_vel = self.dist_pid.compute(dist_error)
            move_msg.twist.linear.x = lin_vel
        else:
            # reached
            self.log_.info(f"Reached goal {current_goal}, moving to next")
            self.publish_stop()
            self.goal_idx += 1

        self.pub_.publish(move_msg)

        if self.goal_idx >= len(self.goals):  # type: ignore
            self.shutdown_signal = True
            self.publish_stop()
            self.log_.info("Reached all goals")
            self.control_timer_.cancel()
            return


def main(args=None):
    rclpy.init(args=args)
    node = DriverPIDNode()

    try:
        while rclpy.ok() and not node.shutdown_signal:
            rclpy.spin_once(node, timeout_sec=(1 / node.rate))

    except KeyboardInterrupt:
        print("Stopping Node")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
