import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist
from std_msgs.msg import Float32, Float32MultiArray
import math
import time
import numpy as np

import torch
import torch.nn as nn

def sigmoid_pi_range(x):
    return 1.0 / (1+np.exp(-(x * 6.0 / np.pi - np.pi)/2.5))

def linear_pi_range(x):
    return np.clip(x / np.pi, 0.001, 0.999)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Binary output
        )

    def forward(self, x):
        return self.model(x)

class SimpleObstacleAvoider(Node):
    def __init__(self):
        super().__init__('simple_obstacle_avoider')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # self.urad_sub = self.create_subscription(Float32, '/urad_distance_calc', self.urad_callback, 10)
        # self.urad_vel_sub = self.create_subscription(Float32MultiArray, '/urad_velocity', self.urad_vel_callback, 10)
        # self.pose_sub = self.create_subscription(TransformStamped, '/raph/nwu/pose', self.pose_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.max_angular_acc = 0.325
        self.max_angular_vel = 0.425
        # self.curr_angular_acc = 0.2
        self.curr_angular_vel = 0.0
        self.max_linear_acc = 0
        # self.max_linear_vel = 0.2
        self.max_linear_vel = 0.1 + np.random.random() * 0.2
        self.curr_linear_vel = self.max_linear_vel

        self.goal_waypoints = [(2.0, 5.0), (-2.0, 5.0), (-2.0, -5.0), (2.0, -5.0)] # Sim corners
        # self.goal_waypoints = [(1.2, -3.59), (1.21, 3.69), (-1.76, 3.63), (-1.78, -3.59)] # REEF corners
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
        self.odom_prev_time = 0.0
        self.measured_vel = 0
        self.distance_threshold = 1.25
        self.prev_time = 0.0
        self.turning_left = False
        self.last_direction_change_time = 0.0
        self.start = True # Orient the robot in the direction of the goal at the start

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.dist_history = []
        self.past_obstacle = False
        self.slow_recovery = True
        self.tti_min = 3.5
        self.tti_avoid = 5
        self.tti_safe = 7.0
        self.tti_counter = 0
        self.slow_recovery_counter = 0
        self.slow_recovery_max = 30 # 20 counts of slow recovery before going to direct yaw error

        # NN states
        self.nn_max_normalizer = 9.9951 # Synthetic dataset max value
        self.nn_found_obstacle = False
        self.model = SimpleClassifier()
        self.model.load_state_dict(torch.load("/tmp/urad_classifier_time.pt"))
        self.model.eval()

    def sigmoid_linear_vel(self, alpha=0.4, kp=0.8, x_0=1):
        dx = self.goal_x - self.curr_x
        dy = self.goal_y - self.curr_y
        goal_dist = math.hypot(dx, dy)
        return self.max_linear_vel * 1/(1+kp*np.exp(-(goal_dist-x_0)/alpha))
    
    def sigmoid_angular_vel(self, alpha=0.4, kp=0.8, x_0=1):
        dx = self.goal_x - self.curr_x
        dy = self.goal_y - self.curr_y

        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # Normalize

        temp = np.clip(self.max_angular_vel * 1/(1+kp*np.exp((abs(yaw_error)-x_0)/alpha)), 0.02, self.max_angular_vel)
        # temp = math.copysign(temp, yaw_error)
        return temp

    def odom_callback(self, msg):
        previous_x = self.curr_x
        previous_y = self.curr_y
        now = time.time()
        time_diff = now - self.odom_prev_time
        self.odom_prev_time = now
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        # For simplicity, only yaw (from quaternion)
        self.measured_vel = np.sqrt((self.curr_x - previous_x)**2 + (self.curr_y - previous_y)**2) / time_diff
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def pose_callback(self, msg):
        self.curr_x = msg.transform.translation.x
        self.curr_y = msg.transform.translation.y

        q = msg.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        dist = msg.ranges[0]
        self.previous_distance = self.latest_distance
        self.latest_distance = dist

    def urad_callback(self, msg):
        dist = msg.data
        # print(dist)
        self.previous_distance = self.latest_distance
        self.latest_distance = dist
    
    def urad_vel_callback(self, msg):
        vel = msg.data[0]
        # print(dist)
        self.measured_vel = vel

    def time_logic(self, time_diff, yaw_error):
        if self.start:
            return
        tti = self.latest_distance / self.measured_vel

        nn_avoid_obstacle = False
        prob = 0
        with torch.no_grad():
            prob = self.model(torch.tensor([[tti / self.nn_max_normalizer]], dtype=torch.float32)).item()
            nn_avoid_obstacle = True if prob > 0.5 else False
        print(self.latest_distance, self.measured_vel, tti, prob, self.slow_recovery)
        if nn_avoid_obstacle:
            if tti < self.tti_min:
                self.curr_angular_vel = self.max_angular_vel
            elif tti < self.tti_avoid:
                k = 2.0
                self.curr_angular_vel = k * (1/tti - 1/self.tti_safe)
                self.slow_recovery_counter = 0
                self.slow_recovery = True
                self.tti_counter += 0.5
            elif tti < self.tti_safe:
                blend = (self.tti_safe - tti) / (self.tti_safe - self.tti_avoid)
                self.curr_angular_vel = (1 - blend) * yaw_error + blend * self.curr_angular_vel
        else:
            if self.slow_recovery:
                temp = self.max_angular_acc * time_diff
                temp = math.copysign(temp, yaw_error)
                # self.curr_angular_vel = self.curr_angular_vel + temp * sigmoid_pi_range(abs(yaw_error))
                # self.curr_angular_vel = min(max(self.curr_angular_vel, -self.max_angular_vel), self.max_angular_vel)
                self.curr_angular_vel = self.sigmoid_angular_vel()
                # self.curr_linear_vel = self.max_linear_vel # self.sigmoid_linear_vel()
                self.curr_linear_vel = self.sigmoid_linear_vel(x_0=0.5)
                self.slow_recovery_counter += 1
                if self.slow_recovery_counter >= (self.slow_recovery_max + self.tti_counter):
                    self.slow_recovery_counter = 0
                    self.slow_recovery = False
                    self.tti_counter = 0
            else:
                temp = math.copysign(self.max_angular_vel, yaw_error)
                self.curr_angular_vel = temp if abs(yaw_error) > 0.1 else yaw_error

        self.curr_linear_vel = self.sigmoid_linear_vel(x_0=0.5)
        self.curr_angular_vel = np.clip(self.curr_angular_vel, -self.max_angular_vel, self.max_angular_vel)

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
        tmp_distance = self.latest_distance / self.nn_max_normalizer
        tmp_distance = torch.tensor([[tmp_distance]], dtype=torch.float32)
        with torch.no_grad():
            prob = self.model(tmp_distance).item()
            self.nn_found_obstacle = True if prob > 0.5 else False

        if self.nn_found_obstacle:
        # if self.latest_distance < self.distance_threshold:
            # Distance check
            self.curr_angular_vel = self.max_angular_vel
            self.slow_recovery_counter = 0
            self.slow_recovery = True
        else:
            # Yaw error
            if self.slow_recovery:
                temp = self.max_angular_acc * time_diff
                temp = math.copysign(temp, yaw_error)
                # self.curr_angular_vel = self.curr_angular_vel + temp * sigmoid_pi_range(abs(yaw_error))
                # self.curr_angular_vel = min(max(self.curr_angular_vel, -self.max_angular_vel), self.max_angular_vel)
                self.curr_angular_vel = self.sigmoid_angular_vel()
                # self.curr_linear_vel = self.max_linear_vel # self.sigmoid_linear_vel()
                self.curr_linear_vel = self.sigmoid_linear_vel(x_0=0.5)
                self.slow_recovery_counter += 1
                if self.slow_recovery_counter >= self.slow_recovery_max:
                    self.slow_recovery_counter = 0
                    self.slow_recovery = False
            else:
                temp = math.copysign(self.max_angular_vel, yaw_error)
                self.curr_angular_vel = temp if abs(yaw_error) > 0.1 else yaw_error
                # self.curr_linear_vel = self.max_linear_vel # self.sigmoid_linear_vel()
                self.curr_linear_vel = self.sigmoid_linear_vel(x_0=0.5)

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
            return

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
        # self.distance_logic(time_diff, yaw_error)
        self.time_logic(time_diff, yaw_error)
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
