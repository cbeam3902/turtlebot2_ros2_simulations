import numpy as np
import cv2 as cv
import pickle

class ImageLocalization():
    def __init__(self):
        # Essentials
        self.img = None
        self.H = -1
        self.W = -1
        self.mpp = -1
        self.mx_min = -1
        self.my_min = -1

        self.dist_holder = None
        self.angle_array = None
        self.numAngles = -1

    def genMapFromImage(self, image_string, numAngles, mpp=-1, mx_min=-1, my_min=-1):
        # Read image and get width and height (image will be black/white so color technically doesn't matter)
        self.img = cv.imread(image_string) # NOTE: (H, W, D)
        self.H, self.W, _ = self.img.shape
        self.mpp = mpp
        self.mx_min = mx_min
        self.my_min = my_min

        # Set up the angle array
        self.numAngles = numAngles
        self.angle_array = np.linspace(0, 2*np.pi, self.numAngles, endpoint=False)

        # Set up the distance holder
        self.dist_holder = np.zeros((self.numAngles, self.H * self.W))

        for i in range(self.H * self.W):
            # i = y * W + x
            x = i % self.W
            y = i // self.W

            # Check for black pixel
            if self.img[y,x,0] == 0:
                continue
            
            # For each angle
            for n in range(self.numAngles):
                r = 0
                x2 = x
                y2 = y
                theta = self.angle_array[n] # NOTE: Angle increments counter-clockwise

                coord = np.array([y2, x2])
                direction = np.array([np.cos(theta), -1*np.sin(theta)])
                coord = coord + direction
                temp_coord = np.round(coord).astype(np.uint64)

                # Check for out of bounds
                if temp_coord[0] < 0 or temp_coord[0] > self.H-1:
                    continue
                
                if temp_coord[1] < 0 or temp_coord[1] > self.W-1:
                    continue

                # Increment r until black pixel
                while self.img[temp_coord[0], temp_coord[1], 0] != 0:
                    r = r + 1
                    coord = coord + direction
                    temp_coord = np.round(coord).astype(np.uint64)

                    # Check for out of bounds
                    if temp_coord[0] < 0 or temp_coord[0] > self.H-1:
                        break
                    
                    if temp_coord[1] < 0 or temp_coord[1] > self.W-1:
                        break

                # Assign distance to that coordinate/angle
                self.dist_holder[n,i] = r
        
    def estimateLocalization(self, ranges, pixels=False):
        # I'm assuming that the ranges are in an np.array (numAngles, 1)
        test_ranges = ranges.copy()
        est_x = -1
        est_y = -1
        est_theta = -1

        # Convert distances to equivalent pixels
        if not pixels:
            test_ranges = test_ranges / self.mpp

        # Get the magnitude of the fft for both the map and range input
        fft_ranges = np.fft.fft(test_ranges, axis=0)
        fft_map = np.fft.fft(self.dist_holder, axis=0)
        mag_ranges = np.abs(fft_ranges)
        mag_map = np.abs(fft_map)

        # Figure out which index it belongs to
        error = mag_map - mag_ranges
        error = error**2
        error = np.sum(error, 0)
        est_idx = np.argmin(error)

        # Get the estimated coordinates
        est_x = est_idx % self.W
        est_y = est_idx // self.W

        # Convert pixel coordinates into world coordinates
        if not pixels:
            # est_x = self.mx_min + est_x * self.mpp
            # est_y = self.my_min + est_y * self.mpp
            temp = est_x.copy()
            est_x = self.mx_min + (self.H - est_y - 1) * self.mpp
            est_y = self.my_min + (self.W - temp - 1) * self.mpp

        # Get the estimated orientation
        k = 1
        while mag_ranges[k, 0] < 1e-6:
            k = k + 1
            if k > self.numAngles:
                k = -1
                break
        if k != -1:
            t_0 = -1j * self.W / (2 * np.pi * k) * np.log(fft_map[k, est_idx] / fft_ranges[k, 0])
            est_theta = 2 * np.pi * t_0 / self.numAngles
        est_theta = np.real(est_theta) % (2 * np.pi)

        return est_x, est_y, est_theta

    def saveDistanceMap(self, map_name):
        with open(map_name, 'wb') as f:
            pickle.dump([self.H, self.W, self.mpp, self.mx_min, self.my_min, self.dist_holder, self.angle_array, self.numAngles], f)
    
    def loadDistanceMap(self, map_name):
        with open(map_name, 'rb') as f:
            self.H, self.W, self.mpp, self.mx_min, self.my_min, self.dist_holder, self.angle_array, self.numAngles = pickle.load(f)


if __name__ == "__main__":
    # Let's test stuff out bit by bit

    # Generate a distance map
    il = ImageLocalization()
    il.genMapFromImage("obstacle_map.png", 32, mpp=0.070073, mx_min=-4.77, my_min=-7)

    # # Localize to a random coordinate
    # print("Test:")
    # shift = np.random.randint(il.numAngles-1) + 1
    # x = np.random.randint(il.W)
    # y = np.random.randint(il.H)

    # while il.img[y,x,0] == 0:
    #     x = np.random.randint(il.W)
    #     y = np.random.randint(il.H)
    
    # print(f"\tPixel Coordinate: ({y},{x})")
    # print(f"\tArray shift: ({shift}, {il.angle_array[shift]} rad.)")

    # ranges = il.dist_holder[:, y * il.W + x].copy()
    # ranges = np.roll(ranges, -shift) # Because apparently numpy does it backwards
    # ranges = ranges.reshape((-1,1))
    # est_x, est_y, est_theta = il.estimateLocalization(ranges, pixels=True)

    # print("\nOutcome:")
    # print(f"\tEstimated Coordinate: ({est_y}, {est_x})")
    # print(f"\tEstimate theta: {est_theta}")

    # # Test some QoL functions (because generating everytime will take a while rather than loading from a txt file or something)
    il.saveDistanceMap("obstacle_map.pkl")
    # il.loadDistanceMap("obstacle_map.pkl")

    # # Test2
    # print("\nTest 2:")
    # shift = np.random.randint(il.numAngles-1) + 1
    # x = np.random.randint(il.W)
    # y = np.random.randint(il.H)

    # while il.img[y,x,0] == 0:
    #     x = np.random.randint(il.W)
    #     y = np.random.randint(il.H)
    
    # print(f"\tPixel Coordinate: ({y},{x})")
    # print(f"\tArray shift: ({shift}, {il.angle_array[shift]} rad.)")

    # ranges = il.dist_holder[:, y * il.W + x].copy()
    # ranges = np.roll(ranges, -shift) # Because apparently numpy does it backwards
    # ranges = ranges.reshape((-1,1))
    # est_x, est_y, est_theta = il.estimateLocalization(ranges, pixels=True)

    # print("\nOutcome:")
    # print(f"\tEstimated Coordinate: ({est_y}, {est_x})")
    # print(f"\tEstimate theta: {est_theta}")