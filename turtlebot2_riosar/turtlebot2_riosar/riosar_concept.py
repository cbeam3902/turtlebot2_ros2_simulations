import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist
from std_msgs.msg import Float32, Float32MultiArray
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# For simplicity (and to visualize it), this is a concept visualizer to see how it might look using a LiDAR based implementation.
# This is simply just using the range measurements obtained from the LiDAR (360 points) and range binning the respective ranges to get range image
##  The reason for the concept is due to the paper (RIO-SAR), they used 4 RADAR sensors to generate a 360 degree SAR image (after trajectory estimation)
##  This will visualize how (in simulation hopefully) how it might look as a 360 degree image just for sanity reasons

# The real implementation might convert this 360 LiDAR scan (or the simple uRAD +-30 deg) as, instead of ranges from the angles, it's a binned form of the ranges to match a FFT'd version of the I/Q measurements

class RIOSARConcept(Node):
    def __init__(self):
        super().__init__('riosar_concept')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.dist = None
        self.range_min = None
        self.range_max = None

    def scan_callback(self, msg):
        self.dist = np.array(msg.ranges)
        self.range_min = msg.range_min
        self.range_max = msg.range_max

    def control_loop(self):
        if self.dist is not None:
            # Let's go with a square image, so 360x360
            temp = self.dist
            temp[temp == np.inf] = self.range_max
            temp = (temp.reshape(1,-1) - self.range_min) / (self.range_max - self.range_min) * 359
            temp = temp.astype(np.int32)
            I = np.zeros((360,360))
            for idx in range(360):
                # 0 is top left, need to be bottom left
                I[359 - temp[0,idx], idx] = 255

            plt.imshow(I)
            plt.show()
        
def main(args=None):
    rclpy.init(args=args)
    node = RIOSARConcept()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()