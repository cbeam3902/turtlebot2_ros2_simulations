import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan  # or use radar equivalent
from std_msgs.msg import Bool
import math
import time
import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.transform import Rotation as R
# from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener

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

class GNCNode(Node):
    def __init__(self):
        super().__init__('gnc_node')

        # --- Parameters ---
        self.goal_x = 4.0
        self.goal_y = 5.0
        self.intermediate_goal = None
        self.scan_radius = 1.0
        self.scan_resolution_deg = 10
        self.obstacle_threshold = 0.7
        # self.nn_max_normalizer = 9.540007 # Obtained while training
        self.nn_max_normalizer = 9.9995 # Synthetic dataset max value
        self.nn_found_obstacle = False
        self.model = SimpleClassifier()
        self.model.load_state_dict(torch.load("/tmp/urad_classifier.pt"))
        self.model.eval()

        # --- State ---
        self.state = 'NAVIGATE_TO_GOAL'
        self.x = self.y = self.yaw = 0.0
        self.scan_distance = None
        self.scan_map = {}
        self.scan_counter = 0
        self.scan_state = 'START'
        self.start_yaw = 0.0
        self.desired_angles = None
        self.logged_angles = None
        self.scan_turn_left = True
        self.intermediate_orientation_check = True
        

        # --- ROS Interfaces ---
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.warn_pub = self.create_publisher(Bool, '/proximity_warning', 10)
        self.create_timer(0.1, self.control_loop)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_yaw_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform("odom", "base_footprint", rclpy.time.Time())
            rot = t.transform.rotation
            r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
            _, _, yaw = r.as_euler('xyz', degrees=False)
            # _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            return yaw
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # siny = 2.0 * (q.w * q.z + q.x * q.y)
        # cosy = 1.0 - 2.0 * (q.y**2 + q.z**2)
        # self.yaw = math.atan2(siny, cosy)
        r = R.from_quat([q.x, q.y, q.z, q.w])
        _, _, yaw = r.as_euler('xyz', degrees=False)
        # _,_,yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw
        # self.get_logger().info(f"Yaw updated: {math.degrees(self.yaw):.2f}")

    def scan_callback(self, msg):
        if len(msg.ranges) > 0:
            r = msg.ranges[0]
            self.scan_distance = r if msg.range_min <= r <= msg.range_max else None

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def rotate_and_scan(self):
        
        if self.scan_state == 'START':
            self.get_logger().info("🔄 Rotating using odometry-based yaw...")
            self.scan_map.clear()
            self.start_yaw = self.yaw
            self.logged_angles = set()
            self.scan_counter = 0
            self.scan_state = 'SCAN'
            self.scan_turn_left = True
        elif self.scan_state == 'SCAN':
            # start_yaw = self.yaw
            last_logged_angle = 0.0
            # logged_angles = set()

            # while True:
                # rclpy.spin_once(self, timeout_sec=0.01)
                # self.yaw = self.get_yaw_from_tf()
                # self.get_logger().info(f"📏 Logged at yaw deg°: {math.degrees(self.yaw):.2f}")
                # Compute relative angle rotated
            delta_yaw = self.normalize_angle(self.yaw - self.start_yaw)
            delta_deg = math.degrees(delta_yaw)
            
            msg = Twist()
            if self.scan_turn_left:
                msg.angular.z = 0.2
            else:
                msg.angular.z = -0.2
                
            if delta_deg > 90 and self.scan_turn_left:
                self.scan_turn_left = False
            elif delta_deg < -90 and not self.scan_turn_left:
                self.scan_turn_left = True

            if delta_deg < 0:
                delta_deg = 360 + delta_deg

            self.cmd_pub.publish(msg)

                # Round to the nearest resolution step
            # print(delta_deg)
            # print(delta_deg)
            rounded_deg = int(round(delta_deg / self.scan_resolution_deg) * self.scan_resolution_deg)

                # Only log a new angle once
            if rounded_deg not in self.logged_angles and 0 <= rounded_deg <= 360:
                if self.scan_distance is not None:
                    self.scan_map[rounded_deg] = self.scan_distance
                    self.get_logger().info(f"📏 Logged at {rounded_deg}°: {self.scan_distance:.2f} m")
                    self.logged_angles.add(rounded_deg)
                    self.scan_counter = self.scan_counter + 1

                # Stop after full 360°
                # if delta_deg >= 360:
                #     break
            if self.scan_counter == 18:
                self.scan_state = 'END'

                # Stop rotation
                self.cmd_pub.publish(Twist())
                self.get_logger().info("✅ 180° scan complete.")

    def pick_intermediate(self, goal_angle, w_angle=0.01, w_dist=1.0):
        best_angle = None
        best_score = -float('inf')  # higher is better

        for angle_deg, dist in self.scan_map.items():
            # Normalize to [-180, 180]
            angle_diff = ((angle_deg - goal_angle + 180) % 360) - 180
            # print(angle_deg, goal_angle, dist, abs(angle_diff))
            # If angle is more than 90° off from goal, apply penalty
            # if abs(angle_diff) > 100:
            #     direction_score = -1  # strong penalty
            # else:
                # Score favors alignment with goal and greater distance
            direction_score = (
                -w_angle * abs(angle_diff) + w_dist * dist
            )
            # print(direction_score)
            if direction_score > best_score:
                best_score = direction_score
                best_angle = angle_deg
                best_dist = dist
        # print(best_angle, best_score)
        return best_angle, best_dist

    def pick_intermediate_goal(self):
        # print(self.scan_map)
        self.get_logger().info("📍 Selecting intermediate waypoint...")
        # Pick the direction with max clearance
        goal_vector = [self.goal_x - self.x, self.goal_y - self.y]
        robot_heading = self.start_yaw
        goal_angle = math.degrees(math.atan2(goal_vector[1], goal_vector[0])) - math.degrees(robot_heading)
        goal_angle = (goal_angle + 360) % 360  # Normalize to [0, 360)
        # print(type(goal_angle))
        best_angle, best_dist = self.pick_intermediate(goal_angle)
        # best_angle, max_dist = max(self.scan_map.items(), key=lambda x: x[1])
        
        theta = math.radians(best_angle)
        dx = self.scan_radius * math.cos(theta + robot_heading)
        dy = self.scan_radius * math.sin(theta + robot_heading)
        self.intermediate_goal = (self.x + dx, self.y + dy)
        self.get_logger().info(f"🔀 Intermediate goal: {self.intermediate_goal}")

    def pick_intermediate_waypoint(self, min_range=1.5, max_range=5.0, neighborhood=20):
        x, y, yaw = self.x, self.y, self.start_yaw
        x_goal, y_goal = self.goal_x, self.goal_y

        # Direction vector to goal
        goal_theta = math.atan2(y_goal - y, x_goal - x)

        best_score = -float('inf')
        best_direction = None
        RADIUS = 1.0  # fixed radius to travel toward new heading

        for angle_deg, distance in self.scan_map.items():
            if distance < min_range:
                continue  # too close to obstacle

            angle_rad = math.radians(angle_deg)
            scan_theta = self.normalize_angle(yaw + angle_rad)  # global angle

            # Compute score based on goal alignment and clearance
            angle_diff = self.normalize_angle(goal_theta - scan_theta)
            alignment_score = math.cos(angle_diff)  # 1 if aligned with goal, -1 if opposite
            # clearance_score = min(distance, max_range) / max_range  # 0 to 1

            local_dists = []
            for offset in range(-neighborhood, neighborhood + 1):
                neighbor_angle = angle_deg + offset
                if neighbor_angle in self.scan_map:
                    d = self.scan_map[neighbor_angle]
                    if d >= min_range:
                        local_dists.append(min(d, max_range))
            if not local_dists:
                continue
            clearance_score = sum(local_dists) / (len(local_dists) * max_range)  # normalized

            # Weighted scoring
            score = 0.2 * alignment_score + 0.8 * clearance_score
            # print(angle_deg, angle_rad, scan_theta, angle_diff, alignment_score, clearance_score, score)
            if score > best_score:
                best_score = score
                best_direction = scan_theta

        # print(best_direction, best_score)
        if best_direction is None:
            self.intermediate_goal = (self.x, self.y)
            self.get_logger().info(f"🔀 No Intermediate goal: {self.intermediate_goal}")


        # Compute intermediate waypoint RADIUS meters away in chosen direction
        new_x = x + RADIUS * math.cos(best_direction)
        new_y = y + RADIUS * math.sin(best_direction)
        self.intermediate_goal = (new_x, new_y)
        self.get_logger().info(f"🔀 Intermediate goal: {self.intermediate_goal}")

    def control_loop(self):
        msg = Twist()
        self.warn_pub.publish(Bool(data= (self.nn_found_obstacle == 1)))
        # === Step 0: NN Classification ===
        if self.scan_distance is not None:
            tmp_distance = self.scan_distance / self.nn_max_normalizer
            tmp_distance = torch.tensor([[tmp_distance]], dtype=torch.float32)
            with torch.no_grad():
                prob = self.model(tmp_distance).item()
                self.nn_found_obstacle = 1 if prob > 0.5 else 0
                # print(self.scan_distance, prob, self.nn_found_obstacle)

        # === Step 1: Detect obstacle ===
        # if self.state == 'NAVIGATE_TO_GOAL' and self.scan_distance is not None and self.scan_distance <= self.obstacle_threshold:
        if self.state == 'NAVIGATE_TO_GOAL' and self.nn_found_obstacle:
            self.get_logger().warn("⚠️ Obstacle too close! Initiating avoidance...")
            self.state = 'SCAN_SURROUNDINGS'
            # return

        # if (self.state == 'NAVIGATE_TO_INTERMEDIATE' and not self.intermediate_orientation_check) and self.scan_distance is not None and self.scan_distance <= self.obstacle_threshold:
        if (self.state == 'NAVIGATE_TO_INTERMEDIATE' and not self.intermediate_orientation_check) and self.nn_found_obstacle:
            self.get_logger().warn("⚠️ Obstacle too close! Initiating avoidance...")
            self.state = 'SCAN_SURROUNDINGS'
            # return
        # === Step 2: Rotate and scan ===
        if self.state == 'SCAN_SURROUNDINGS':
            self.rotate_and_scan()
            # self.pick_intermediate_goal()
            # print(self.state, self.scan_state)
            if self.scan_state == 'END':
                self.scan_state = 'START'
                self.state = 'PICK_INTERMEDIATE'
            # return

        # === Step 2.5: Pick an intermediate goal
        if self.state == 'PICK_INTERMEDIATE':
            self.pick_intermediate_waypoint()
            self.state = 'NAVIGATE_TO_INTERMEDIATE'
            self.intermediate_orientation_check = True
        # === Step 3: Navigate to intermediate waypoint ===
        target = None
        if self.state == 'NAVIGATE_TO_INTERMEDIATE' and self.intermediate_goal:
            target = self.intermediate_goal
            goal_angle = math.degrees(math.atan2(target[1] - self.y, target[0] - self.x)) - math.degrees(self.yaw)
            # print(math.degrees(math.atan2(target[1] - self.y, target[0] - self.x)), math.degrees(self.yaw), goal_angle)
            if abs(goal_angle) < 13:
                self.intermediate_orientation_check = False
            if self.distance_to(*target) < 0.2:
                self.state = 'NAVIGATE_TO_GOAL'
                self.intermediate_goal = None
                # return

        # === Step 4: Navigate to main goal ===
        if self.state == 'NAVIGATE_TO_GOAL':
            target = (self.goal_x, self.goal_y)
            if self.distance_to(*target) < 0.2:
                self.get_logger().info("🎯 Goal reached!")
                self.state = 'STOP'

        # === Control logic ===
        if target and self.state not in ['STOP', 'SCAN_SURROUNDINGS']:
            dx = target[0] - self.x
            dy = target[1] - self.y
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            angle_diff = math.atan2(math.sin(angle - self.yaw), math.cos(angle - self.yaw))

            msg.linear.x = 0.2 if abs(angle_diff) < 0.3 else 0.0
            msg.angular.z = angle_diff
            self.cmd_pub.publish(msg)
        elif self.state == 'STOP':
            self.cmd_pub.publish(Twist())

    def distance_to(self, tx, ty):
        return math.hypot(tx - self.x, ty - self.y)


def main(args=None):
    rclpy.init(args=args)
    node = GNCNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
