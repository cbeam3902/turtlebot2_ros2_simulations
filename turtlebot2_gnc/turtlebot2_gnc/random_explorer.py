import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan  # Replace with radar msg if needed
from std_msgs.msg import Bool
import random
import time

class RandomExplorer(Node):
    def __init__(self):
        super().__init__('random_explorer')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.warn_pub = self.create_publisher(Bool, '/proximity_warning', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.obstacle_too_close = False
        self.timer = self.create_timer(0.2, self.control_loop)

        self.movement_state = 'FORWARD'
        self.last_change_time = time.time()

    def scan_callback(self, msg):
        # Assume single-beam sensor (1 value)
        if len(msg.ranges) > 0:
            r = msg.ranges[0]
            if msg.range_min <= r <= msg.range_max:
                self.obstacle_too_close = (r <= 0.5)
            else:
                self.obstacle_too_close = False
        else:
            self.obstacle_too_close = False

    def control_loop(self):
        msg = Twist()

        if self.obstacle_too_close:
            self.get_logger().info('⚠️ Too close to object! Backing off...')
            self.warn_pub.publish(Bool(data=True))

            # Turn in a new random direction
            msg.linear.x = -0.2
            # msg.angular.z = random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.5)
            msg.angular.z = 3.14
            self.cmd_pub.publish(msg)
            # rclpy.spin_once(self, timeout_sec=1.0)  # brief turn
            self.last_change_time = time.time()
            self.movement_state = 'FORWARD'
        else:
            self.warn_pub.publish(Bool(data=False))

            # Change direction every 3–5 seconds randomly
            if time.time() - self.last_change_time > random.uniform(3.0, 5.0):
                self.last_change_time = time.time()
                self.movement_state = 'TURN'
                msg.linear.x = 0.0
                # msg.angular.z = random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.0)
                msg.angular.z = random.uniform(-3.14, 3.14) * 1 #* random.uniform(0.5, 1.0)
            elif self.movement_state == 'TURN':
                # Continue turning briefly
                msg.linear.x = 0.0
                msg.angular.z = msg.angular.z
                self.movement_state = 'FORWARD'
            else:
                # Normal forward motion
                msg.linear.x = 0.4
                msg.angular.z = 0.0

        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RandomExplorer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
