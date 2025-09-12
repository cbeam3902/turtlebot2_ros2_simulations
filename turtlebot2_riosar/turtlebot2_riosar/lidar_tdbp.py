import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# ChatGPT generated

class SARBackprojection(Node):
    def __init__(self):
        super().__init__('sar_backprojection')

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Parameters
        self.image_size = 256
        self.grid_extent = 5.0  # meters, half-size of image scene
        self.pixel_size = (2*self.grid_extent) / self.image_size

        # SAR accumulation image
        self.sar_image = np.zeros((self.image_size, self.image_size), dtype=np.complex64)

        # Pose of sensor
        self.sensor_pose = np.array([0.0, 0.0, 0.0])

        self.get_logger().info("SAR TDBP Node started")

    def odom_callback(self, msg):
        # Update platform (sensor) position
        self.sensor_pose = np.array([msg.pose.pose.position.x,
                                     msg.pose.pose.position.y,
                                     msg.pose.pose.position.z])

    def lidar_callback(self, msg):
        angles = np.arange(msg.angle_min - msg.angle_min, msg.angle_max + msg.angle_increment - msg.angle_min, msg.angle_increment)
        ranges = np.array(msg.ranges)
        intensities = np.array(msg.intensities) if msg.intensities else np.ones_like(ranges)

        # Convert LiDAR ranges to Cartesian points in sensor frame
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.stack([xs, ys], axis=-1)

        # Backprojection step
        self.backproject(points, intensities)

    def backproject(self, points, intensities):
        # Grid coordinates
        x = np.linspace(-self.grid_extent, self.grid_extent, self.image_size)
        y = np.linspace(-self.grid_extent, self.grid_extent, self.image_size)
        X, Y = np.meshgrid(x, y)

        # Flatten pixel grid
        pixels = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # Compute ranges from sensor to each pixel
        sensor_xy = self.sensor_pose[:2]
        pixel_ranges = np.linalg.norm(pixels - sensor_xy, axis=1)

        # Backproject LiDAR points onto grid
        for p, intensity in zip(points, intensities):
            # Range of LiDAR hit
            r = np.linalg.norm(p)

            # Find pixels whose slant range is near LiDAR return
            mask = np.isclose(pixel_ranges, r, atol=self.pixel_size)

            # Add contribution (complex exponential phase for SAR, simplified here)
            self.sar_image.ravel()[mask] += intensity * np.exp(-1j * 2 * np.pi * r)

    def save_image(self):
        import matplotlib.pyplot as plt
        plt.imshow(np.abs(self.sar_image), cmap='gray')
        plt.title("SAR Backprojection (LiDAR Simulated)")
        plt.colorbar()
        plt.show()
        plt.savefig("sar_image.png")


def main(args=None):
    rclpy.init(args=args)
    node = SARBackprojection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_image()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
