import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import time
import numpy as np

def sigmoid_pi_range(x):
    return 1.0 / (1+np.exp(-(x * 6.0 / np.pi - np.pi)/2.5))

def linear_pi_range(x):
    return np.clip(x / np.pi, 0.001, 0.999)

class SimpleObstacleAvoider(Node):
    def __init__(self):
        super().__init__('simple_obstacle_avoider')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.max_angular_acc = 0.35
        self.max_angular_vel = 0.3
        # self.curr_angular_acc = 0.2
        self.curr_angular_vel = 0.0
        self.max_linear_acc = 0
        self.curr_linear_vel = 0.2

        self.goal_waypoints = [(2.0, 5.0), (-2.0, 5.0), (-2.0, -5.0), (2.0, -5.0)]
        self.goal_waypoints_idx = 0
        self.goal_x = self.goal_waypoints[0][0]  # Arbitrary goal for demo
        self.goal_y = self.goal_waypoints[0][1]
        self.lap_counter = 0
        self.max_laps = 1

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.yaw = 0.0

        self.latest_distance = float('inf')
        self.previous_distance = -1
        self.distance_threshold = 1.25
        self.prev_time = 0.0
        self.turning_left = False
        self.last_direction_change_time = 0.0
        self.start = True # Orient the robot in the direction of the goal at the start

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.dist_history = []
        self.past_obstacle = False
        self.slow_recovery = True
        self.slow_recovery_counter = 0
        self.slow_recovery_max = 60 # 20 counts of slow recovery before going to direct yaw error


    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        # For simplicity, only yaw (from quaternion)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        dist = msg.ranges[0]
        self.previous_distance = self.latest_distance
        self.latest_distance = dist

    def distance_logic(self, time_diff, yaw_error):
        if self.start:
            return
        # print(time_diff, yaw_error, self.curr_angular_vel, self.latest_distance < self.distance_threshold)
        # Logic needed
        ##  Distance check to see if the distance needed went past the threshold
        ##  Distance derivative check to see if the distance is not increasing as it turns right (turn left then)
        ###     If turning left, would probably need to do a derivative sign counter check as it'll be
        ###         positive, negative, then positive again assuming left is also not a wall
        ##  Yaw error feedback loop as it'll be at the incorrect orientation
        # print(sigmoid_pi_range(abs(yaw_error)))
        if self.latest_distance < self.distance_threshold:
            # Distance check
            self.curr_angular_vel = self.max_angular_vel
            self.slow_recovery_counter = 0
            self.slow_recovery = True
        else:
            # Yaw error
            if self.slow_recovery:
                temp = self.max_angular_acc * time_diff
                temp = math.copysign(temp, yaw_error)
                self.curr_angular_vel = self.curr_angular_vel + temp * sigmoid_pi_range(abs(yaw_error))
                self.curr_angular_vel = min(max(self.curr_angular_vel, -self.max_angular_vel), self.max_angular_vel)
                self.slow_recovery_counter += 1
                if self.slow_recovery_counter >= self.slow_recovery_max:
                    self.slow_recovery_counter = 0
                    self.slow_recovery = False
            else:
                temp = math.copysign(self.max_angular_vel, yaw_error)
                self.curr_angular_vel = temp if abs(yaw_error) > self.max_angular_vel else yaw_error

    def control_loop(self):
        cmd = Twist()
        dx = self.goal_x - self.curr_x
        dy = self.goal_y - self.curr_y
        goal_dist = math.hypot(dx, dy)

        if goal_dist < 0.2:
            self.goal_waypoints_idx += 1
            if self.goal_waypoints_idx >= len(self.goal_waypoints):
                self.goal_waypoints_idx = 0
                self.lap_counter += 1
            self.goal_x = self.goal_waypoints[self.goal_waypoints_idx][0]
            self.goal_y = self.goal_waypoints[self.goal_waypoints_idx][1]
            return
        if self.lap_counter >= self.max_laps:
            # Stop logic, want to do laps
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # Normalize
        
        now = time.time()
        time_diff = now - self.prev_time
        self.prev_time = now
        angular_z = 0.0
        linear_x = 0.0
        if self.start:
            angular_z = np.clip(yaw_error, -self.max_angular_vel, self.max_angular_vel)
            linear_x = 0.0
            self.curr_angular_vel = angular_z
        else:
            angular_z = self.curr_angular_vel
            linear_x = self.curr_linear_vel
        self.distance_logic(time_diff, yaw_error)
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        # print(linear_x, angular_z, self.start)
        self.cmd_pub.publish(cmd)

        if self.start and abs(yaw_error) < 0.1:
            self.start = False


def main(args=None):
    rclpy.init(args=args)
    node = SimpleObstacleAvoider()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
