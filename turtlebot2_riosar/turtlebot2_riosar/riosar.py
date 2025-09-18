import rclpy
import copy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist, TwistWithCovarianceStamped
from std_msgs.msg import Float32, Float32MultiArray
from rclpy.time import Time
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2

plt.ion()

class RIOSARBasic(Node):
    def __init__(self):
        super().__init__('riosar_basic')

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.placer_timer = self.create_timer(0.1, self.control_loop)
        self.sar_timer = self.create_timer(0.1, self.sar_loop)
        self.vel_pub = self.create_publisher(TwistWithCovarianceStamped, '/lidar_vel', 10)

        self.scan_msg = None
        self.odom_msg = None
        self.prev_ranges = None
        self.scan_vel = 0.0
        self.scan_time = time.time()

        # Variables to use
        self.N_FFT = 256
        self.dist_jump_threshold = 1.0

        # Needed for SAR
        self.n_pulses = 32
        self.ph_left = np.zeros((self.N_FFT, self.n_pulses), np.float) # N Pulses of N_FFT ranges
        self.ph_right = np.zeros((self.N_FFT, self.n_pulses), np.float) # N Pulses of N_FFT ranges
        self.position = np.zeros((3, self.n_pulses), np.float) # Position: X, Y, Z
        self.orientation = np.zeros((4, self.n_pulses), np.float) # Orientation: X, Y, Z, W
        self.pulse_counter = 0 # Need to keep track of how many control calls been made since the beginning
        # self.image = np.zeros((self.N_FFT, self.N_FFT), np.float) # Single image
        self.image = np.ones((self.N_FFT, self.N_FFT*2), np.float) * 0.00001 # Left & right image


    # For now, let's assume that the scan and odom messages happen relatively close to each other
    def odom_callback(self, msg):
        self.odom_msg = msg

    def scan_callback(self, msg):
        self.scan_msg = msg
        curr_time = time.time()

        if self.prev_ranges is not None:
            # # NOTE: Previous velocity using all ranges
            # # Calculate the angles for each LiDAR scan
            # angles = [msg.angle_min + i * msg.angle_increment for i in range(len(self.prev_ranges))]
            # angles = np.array(angles)

            # # Get the difference between the current ranges and previous ranges over time (velocity)
            # temp = (np.array(msg.ranges) - self.prev_ranges) / (curr_time - self.scan_time)
            
            # # Convert radial velocity to real velocity and average it
            # self.scan_vel = np.mean(temp / np.cos(angles))
            # # print(self.scan_vel)

            self.scan_vel = (msg.ranges[0] - self.prev_ranges[0]) / (curr_time - self.scan_time)
            msg_vel = TwistWithCovarianceStamped()
            time_ros = Time()
            msg_vel.header.frame_id = 'base_scan'
            (msg_vel.header.stamp.sec, msg_vel.header.stamp.nanosec) = time_ros.seconds_nanoseconds()

            msg_vel.twist.twist.linear.x = self.scan_vel
            self.vel_pub.publish(msg_vel)
            
        self.scan_time = curr_time
        self.prev_ranges = np.array(msg.ranges)

    def control_loop(self):
        if self.scan_msg is not None and self.odom_msg is not None:
            # Get a copy of each message in case they change from the callbacks
            # Also assume that these are close enough time wise that
            temp_odom = copy.deepcopy(self.odom_msg)
            temp_scan = copy.deepcopy(self.scan_msg)

            # Roll the pose and range holders once so the recent "pulse" is at the end
            self.position = np.roll(self.position, -1, 1)
            self.orientation = np.roll(self.orientation, -1, 1)
            self.ph_left = np.roll(self.ph_left, -1, 1)
            self.ph_right = np.roll(self.ph_right, -1, 1)
            
            # Assign the position and orientation at the end
            pos = temp_odom.pose.pose.position
            quat = temp_odom.pose.pose.orientation

            pos = np.array([pos.x, pos.y, pos.z])
            quat = np.array([quat.x, quat.y, quat.z, quat.w])

            self.position[:,-1] = pos
            self.orientation[:,-1] = quat

            # Get the LiDAR ranges for "left" side
            dist = np.array(temp_scan.ranges[75:105]).reshape(-1)
            range_min = temp_scan.range_min
            range_max = temp_scan.range_max

            # Find the targets
            # Get the indexes for big jumps
            dist_diff = np.abs(np.diff(dist)) > self.dist_jump_threshold
            target_ranges = None
            # If there are not any then just get the mean of the ranges
            if not np.any(dist_diff):
                target_ranges = np.array(np.mean(dist))
            else:
                diff_idx = [i for i, x in enumerate(dist_diff) if x]
                diff_idx = np.array(diff_idx, np.int32) + 1
                target_ranges = np.zeros(sum(dist_diff)+1)
                start_idx = 0
                end_idx = diff_idx[0]
                for idx in range(sum(dist_diff)):
                    target_ranges[idx] = np.mean(dist[start_idx:end_idx])
                    start_idx = end_idx
                    if idx != sum(dist_diff) - 1:
                        end_idx = diff_idx[idx+1]
                target_ranges[-1] = np.mean(dist[start_idx:])
            
            # Make a range profile similar to the one from a RADAR FFT
            range_profile = np.ones(self.N_FFT*2) * 0.00001
            # print(range_max)
            # target_ranges[target_ranges > range_max] = range_max
            target_ranges = (target_ranges - range_min) / (range_max - range_min) * self.N_FFT
            target_ranges = target_ranges.astype(np.int32).tolist()
            # print(target_ranges)
            if np.all(np.array(target_ranges) >= 0):
                range_profile[target_ranges] = 1.0

            # for target in target_ranges:
            #     range_profile[target] = 0
            #     range_profile[-target] = 0
            # try:
            #     for target in target_ranges:
            #         range_profile[target] = 0
            # except:
            #     range_profile[target_ranges] = 0
            
            self.ph_left[:,-1] = range_profile[0:256]

            # Get the LiDAR ranges for "right" side
            dist = np.array(temp_scan.ranges[255:285]).reshape(-1)
            range_min = temp_scan.range_min
            range_max = temp_scan.range_max

            # Find the targets
            # Get the indexes for big jumps
            dist_diff = np.abs(np.diff(dist)) > self.dist_jump_threshold
            target_ranges = None
            # If there are not any then just get the mean of the ranges
            if not np.any(dist_diff):
                target_ranges = np.array(np.mean(dist))
            else:
                diff_idx = [i for i, x in enumerate(dist_diff) if x]
                diff_idx = np.array(diff_idx, np.int32) + 1
                target_ranges = np.zeros(sum(dist_diff)+1)
                start_idx = 0
                end_idx = diff_idx[0]
                for idx in range(sum(dist_diff)):
                    target_ranges[idx] = np.mean(dist[start_idx:end_idx])
                    start_idx = end_idx
                    if idx != sum(dist_diff) - 1:
                        end_idx = diff_idx[idx+1]
                target_ranges[-1] = np.mean(dist[start_idx:])
            
            # Make a range profile similar to the one from a RADAR FFT
            range_profile = np.ones(self.N_FFT*2) * 0.00001
            # print(range_max)
            # target_ranges[target_ranges > range_max] = range_max
            target_ranges = (target_ranges - range_min) / (range_max - range_min) * self.N_FFT
            target_ranges = target_ranges.astype(np.int32).tolist()
            # print(target_ranges)
    
            if np.all(np.array(target_ranges) >= 0):
                range_profile[target_ranges] = 1.0

            # for target in target_ranges:
            #     range_profile[target] = 0
            #     range_profile[-target] = 0
            # try:
            #     for target in target_ranges:
            #         range_profile[target] = 0
            # except:
            #     range_profile[target_ranges] = 0
            
            self.ph_right[:,-1] = range_profile[0:256]

            # # I/Q? Need to comment the plt.ion from above
            # IQ = np.fft.ifft(np.array(range_profile))
            # # plt.plot(np.arange(0,512), range_profile)
            # plt.plot(np.arange(0,512),IQ.real, label="I")
            # plt.plot(np.arange(0,512),IQ.imag, label="Q")
            # plt.legend()
            # plt.show()

            # # Visualize the range bin
            # fig, axs = plt.subplots(2)
            # x_plot = np.array([x for x in range(dist.shape[0])])
            # range_plot = np.array([x for x in range(256)])
            # print(range_profile)
            # axs[0].plot(x_plot, dist)
            # axs[1].plot(range_plot, range_profile)
            # plt.show()

            if self.pulse_counter < self.n_pulses:
                self.pulse_counter = self.pulse_counter + 1
    
    def sar_bp(self, ph, position):
        # As the input is already in a range-compressed format, this will need to be done similarly to stripmap SAR
        # NOTE: Only have experience with spotlight SAR, need to double check the differences

        # Idea:
        #   Assume moving only linearly
        #   Assume that each pulse is equally spaced out
        #   Assume that they each illuminate their own respective portions of the image
        #   Figure out how which pixel has a certain distance away from that "pulse" position
        #   Figure out how much that pulse contributes to the image

        # Figure out spacing between pulses
        avg_spacing = np.mean(np.abs(np.diff(position)))
        
        # Get the distance for both x and y locations
        y_vector = np.linspace(self.scan_msg.range_min, self.scan_msg.range_max, self.N_FFT) # Shouldn't change
        x_vector = np.linspace(0, avg_spacing * self.n_pulses, self.N_FFT)

        # For each pulse, generate a distance grid
        for idx in range(self.n_pulses):
            X, Y = np.meshgrid(x_vector, y_vector)
            distance_grid = np.sqrt(X**2 + Y**2)
            I = np.where((distance_grid > self.scan_msg.range_min) & (distance_grid < self.scan_msg.range_max))
            self.image[I] = np.interp(distance_grid[I], y_vector.reshape(-1), ph[:,idx])
            x_vector = x_vector - idx * avg_spacing

        plt.imshow(self.image)
        plt.show()
    
    def sar_tdbp(self, ph, position):
        # Need to convert the LiDAR range profiles into a I/Q representation
        ph_ifft = np.zeros((self.N_FFT, self.n_pulses), np.complex64)

        for idx in range(self.n_pulses):
            ph_ifft[:,idx] = np.fft.ifft(ph[:,idx])
        plt.plot(ph_ifft[:,0].real)
        plt.plot(ph_ifft[:,0].imag)
        plt.show()
        # TODO: Generate an image grid to figure out how long it takes to reach that pixel

    def sar_rda(self, S_rc, trajectory):
        S_rc = S_rc.T
        trajectory = trajectory.T
        c = 299792458
        fc = None # Need to come up with a substitute
        PRF = None # Need to come up with a substitue
        dr = None # Need to come up with a substitute
        """
        Basic Range-Doppler Algorithm (RDA) for SAR stripmap processing.

        Parameters:
        -----------
        S_rc : ndarray (Na x Nr)
            Range-compressed phase history data (slow-time x fast-time).
        trajectory : ndarray (Na x 3)
            Platform positions [x, y, z] for each azimuth pulse.
        fc : float
            Carrier frequency (Hz).
        c : float
            Speed of light (m/s).
        PRF : float
            Pulse repetition frequency (Hz).
        dr : float
            Range sampling spacing (m).
        
        Returns:
        --------
        img : ndarray (Na x Nr)
            Focused SAR image (magnitude).
        """

        Na, Nr = S_rc.shape
        # Wavenumber
        k = 4 * np.pi * fc / c

        # ---- 1. Motion compensation ----
        # Reference track: straight line at constant height
        ref_traj = np.mean(trajectory, axis=0)
        R_ref = np.linalg.norm(trajectory - ref_traj, axis=1)
        phase_correction = np.exp(-1j * k * R_ref)
        S_mocomp = (S_rc.T * phase_correction).T  # apply to each range bin

        # ---- 2. Azimuth FFT (to Doppler domain) ----
        S_doppler = np.fft.fftshift(np.fft.fft(S_mocomp, axis=0), axes=0)
        fa = np.fft.fftshift(np.fft.fftfreq(Na, d=1.0/PRF))  # azimuth frequency axis

        # ---- 3. Range Cell Migration Correction (RCMC) ----
        # Range migration: drc = f(fa) (approximate quadratic)
        # Create new array for corrected data
        S_rcmc = np.zeros_like(S_doppler, dtype=complex)
        r_axis = np.arange(Nr) * dr

        for i, f in enumerate(fa):
            # Compute differential range for this Doppler freq (simplified)
            # For a straight trajectory at speed v: v = PRF * platform_spacing
            # Here we approximate migration as negligible unless you have wide swath
            # For full implementation, compute R(f) from geometry
            delta_r = 0.0  # <-- replace with actual RCMC if needed
            
            # Interpolate along range
            r_corrected = r_axis + delta_r
            interp_func = interp1d(r_corrected, S_doppler[i, :],
                                kind='linear', bounds_error=False, fill_value=0)
            S_rcmc[i, :] = interp_func(r_axis)

        # ---- 4. Azimuth compression ----
        # Matched filter in Doppler domain is conjugate of expected phase history
        # For linear motion at speed v, phase ~ exp(-j*pi*Kr*t^2)
        # For a first pass, skip custom matched filter if platform is straight & uniform
        S_azcomp = S_rcmc  # in simple case

        # ---- 5. Inverse Azimuth FFT ----
        img = np.fft.ifft(np.fft.ifftshift(S_azcomp, axes=0), axis=0)
        self.image = np.abs(img)
        # return np.abs(img)

    def sar_rma_omega_k(self, S_rc, trajectory):
        S_rc = S_rc.T
        trajectory = trajectory.T
        c = 299792458
        fc = None # Need to come up with a substitute
        PRF = None # Need to come up with a substitue
        dr = None # Need to come up with a substitute
        r0=0.0
        scene_center=None
        """
        Range Migration Algorithm (omega-k / Stolt) for near-linear flight.

        Parameters
        ----------
        S_rc : ndarray (Na, Nr)
            Range-compressed slow-time vs fast-time complex data.
        trajectory : ndarray (Na, 3)
            Platform (x,y,z) positions in meters for each slow-time sample (Na rows).
        fc : float
            Carrier frequency (Hz).
        c : float
            Speed of light (m/s).
        PRF : float
            Pulse Repetition Frequency (Hz).
        dr : float
            Range sample spacing (m) (spacing between adjacent range bins).
        r0 : float, optional
            Range corresponding to the first range bin (m). Default 0.0.
        scene_center : tuple (x,y,z) optional
            Scene reference point used for motion compensation. If None, the mean of trajectory
            projected forward by the mean slant-range is used (simple heuristic).
        
        Returns
        -------
        img : ndarray (Na, Nr)
            Focused complex image (along-track × slant-range).
        r_axis : ndarray (Nr,)
            Slant-range axis in meters for output image.
        x_axis : ndarray (Na,)
            Along-track axis (meters) — positions along the mean-track direction.
        """
        Na, Nr = S_rc.shape

        # ------------------------------
        # 1) Build range axis (slant-range)
        # ------------------------------
        r_axis = r0 + np.arange(Nr) * dr  # slant range for each bin (meters)

        # ------------------------------
        # 2) Estimate along-track axis & speed v
        #    (project trajectory onto mean-track direction)
        # ------------------------------
        # Compute along-track direction by linear regression on positions
        # Here we use principal axis (first principal component) of x,y coordinates
        xy = trajectory[:, :2]  # ignore z for along-track estimation
        mean_xy = xy.mean(axis=0)
        # subtract mean
        xy0 = xy - mean_xy
        # SVD to get principal direction
        U, Svals, Vt = np.linalg.svd(xy0, full_matrices=False)
        track_dir = Vt[0]  # unit vector along-track in (x,y)
        # Project positions onto track_dir to form scalar along-track coordinate x_pos
        x_pos = (xy - mean_xy) @ track_dir
        # Ensure monotonically increasing; if decreasing, reverse sign
        if x_pos[-1] < x_pos[0]:
            x_pos = -x_pos
            track_dir = -track_dir
        # instantaneous along-track spacing and mean speed
        dt = 1.0 / PRF
        distances = np.diff(x_pos)
        mean_dx = np.median(distances)  # median step
        v = mean_dx / dt
        # x axis is along-track positions (meters)
        x_axis = x_pos.copy()

        # ------------------------------
        # 3) Motion Compensation to reference range
        #    We pick reference point (scene_center) or use a simple center guess:
        # ------------------------------
        if scene_center is None:
            # place scene center at mean projection + median slant range in radar boresight
            # simple heuristic: reference point at mean position plus mean slant-range toward nadir
            scene_center = np.array([mean_xy[0] + track_dir[0]*0.0,
                                    mean_xy[1] + track_dir[1]*0.0,
                                    np.mean(trajectory[:,2])])  # keep same altitude
        else:
            scene_center = np.asarray(scene_center)

        # compute reference ranges R_m (range from each platform pos to scene_center)
        Rm = np.linalg.norm(trajectory - scene_center.reshape(1,3), axis=1)  # length Na
        # choose a reference range (mean)
        Rref = np.mean(Rm)

        # two-way wavenumber constant for center freq:
        k_two_way_center = 4.0 * np.pi * fc / c  # equals 2 * (2*pi/lambda) because two-way

        # phase correction: remove exp(-j * k_two_way_center * (Rm - Rref))
        phase_corr = np.exp(-1j * k_two_way_center * (Rm - Rref))
        S_mocomp = (S_rc.T * phase_corr).T  # apply to each range bin (Na x Nr)

        # ------------------------------
        # 4) FFT in azimuth and range (to Doppler freq and range freq)
        # ------------------------------
        # FFT along azimuth (slow-time) -> Doppler domain
        S_fa_tr = fftshift(fft(S_mocomp, axis=0), axes=0)  # (Na x Nr)
        # FFT along range (fast-time) -> range-frequency domain
        S_fa_fr = fftshift(fft(S_fa_tr, axis=1), axes=1)

        # frequency axes
        fa = fftshift(np.fft.fftfreq(Na, d=dt))           # azimuth frequency (Hz)
        fr = fftshift(np.fft.fftfreq(Nr, d=dr * 2.0 / c)) # NOTE: d in seconds? we need mapping; see below

        # --- careful: mapping from range sample spacing to time
        # If dr is in meters, the corresponding fast-time sampling interval is:
        #    dt_fast = 2*dr/c   (two-way travel time)
        # so use dt_fast to get range-frequency axis
        dt_fast = 2.0 * dr / c
        fr = fftshift(np.fft.fftfreq(Nr, d=dt_fast))  # Hz

        # convert range-frequency to two-way wavenumber (k_r)
        k_r = 4.0 * np.pi * (fc + fr) / c  # two-way spatial angular wavenumber (rad/m)
        # azimuth spatial wavenumber kx = 2*pi * fa / v
        if v == 0:
            raise ValueError("Estimated platform speed v == 0. Check trajectory / PRF.")
        kx = 2.0 * np.pi * fa / v  # rad/m

        # prepare 2D grids for interpolation
        KX, KR = np.meshgrid(kx, k_r, indexing='ij')   # shapes (Na, Nr)

        # ------------------------------
        # 5) Stolt interpolation (map KR -> Kz using dispersion relation)
        #    Standard mapping: Kz = sqrt( (KR_total)^2 - KX^2 )
        #    Here KR_total is the absolute two-way k (including carrier)
        # ------------------------------
        # KR currently is two-way k associated with (fc + fr)
        KR_total = KR.copy()

        # compute Kz (positive root)
        # Avoid negative inside sqrt due to numerical reasons: clamp small negative to 0
        inside = KR_total**2 - KX**2
        inside = np.where(inside < 0, 0.0, inside)
        Kz = np.sqrt(inside)  # (Na, Nr)

        # The Stolt mapping maps samples at KR -> new KR' = Kz.
        # We will interpolate the spectrum along the range-wavenumber axis (2nd axis).
        # Because our S_fa_fr is indexed as [fa_index, fr_index] which corresponds to [kx, kr],
        # we need to, for each kx (row), interpolate from original KR (1D) -> Kz_row (1D).
        S_stolt = np.zeros_like(S_fa_fr, dtype=complex)

        # we will work row-by-row (per kx) and interpolate the spectrum along the range axis
        # Original 1D KR axis (monotonic)
        kr_lin = KR_total[0, :]  # should be identical for all rows because k_r is same column wise

        for ix in range(len(kx)):
            # for this kx, target KR' = Kz[ix, :]
            target_kr = Kz[ix, :]

            # if kr_lin not strictly monotonic (should be), ensure sorted order for interp1d
            interp_real = interp1d(kr_lin, S_fa_fr[ix, :].real, kind='linear',
                                bounds_error=False, fill_value=0.0, assume_sorted=True)
            interp_imag = interp1d(kr_lin, S_fa_fr[ix, :].imag, kind='linear',
                                bounds_error=False, fill_value=0.0, assume_sorted=True)

            S_stolt[ix, :] = interp_real(target_kr) + 1j * interp_imag(target_kr)

        # ------------------------------
        # 6) Inverse transforms to image domain
        # ------------------------------
        # inverse range FFT (along second axis)
        S_ifr = ifft(ifftshift(S_stolt, axes=1), axis=1)
        # inverse azimuth FFT
        S_ia = ifft(ifftshift(S_ifr, axes=0), axis=0)

        # final focused complex image (along-track × slant-range)
        img = S_ia
        self.image = np.abs(img)
        # return img, r_axis, x_axis

    def sar_rc(self, ph):
        # Let's try range compression
        # Bascially for a given image size, just roll the image back and used the range compressed values to make the image
        self.image = np.roll(self.image, -1, 1)
        self.image[:,-1] = ph[:,-1]
        plt.imshow(self.image)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def sar_lidar(self):
        ranges = self.scan_msg.ranges
        min_angle = self.scan_msg.angle_min
        max_angle = self.scan_msg.angle_max
        angle_increment = self.scan_msg.angle_increment

        angles = np.arange(min_angle, max_angle + angle_increment, angle_increment)

        # For simplicity, have x_coor be modified by the x odom location
        x_coor = ranges * np.cos(angles) + self.odom_msg.pose.pose.position.x
        y_coor = ranges * np.sin(angles)

        # For now, let's get an "image" of the left side of the map
        # Have the image space be 10m (-5 to 5) by 10m (0 to 10)
        x = np.linspace(-5, 5, self.N_FFT)
        y = np.linspace(0, 10, self.N_FFT)

        X, Y = np.meshgrid(x, y)

        x_idx = [np.argmin(np.abs(x - temp_x)) for temp_x in x_coor]
        y_idx = [np.argmin(np.abs(y - temp_y)) for temp_y in y_coor]

        # print(x_idx[0], y_idx[0])
        for idx in range(len(x_idx)):
            self.image[y_idx[idx], x_idx[idx]] = 1.0

        # Show image or plot
        # plt.plot(x_coor, y_coor)
        plt.imshow(self.image)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def sar_radar(self, ph, position):
        # For now, let's get the distances from the range profile
        distance_axis = np.linspace(self.scan_msg.range_min, self.scan_msg.range_max, self.N_FFT)
        target_idx = ph > -1
        target_distance = distance_axis[target_idx]

        # For now, let's get an "image" of the left side of the map
        # Have the image space be 10m (-5 to 5) by 10m (0 to 10)
        x = np.linspace(-5, 5, self.N_FFT)
        y = np.linspace(0, 10, self.N_FFT)

        X, Y = np.meshgrid(x, y)

        # For each target, figure out which distance is within a certain boundary of the image space
        # In case there's only 1 target
        distance_grid = np.sqrt((X - position[0])**2 + Y**2)
        # try:
        #     for target in target_distance:
        #         contrib_idx = np.abs(distance_grid - target) < 0.01
        #         self.image[contrib_idx] += 1
        # except:
        #     contrib_idx = np.abs(distance_grid - target_distance) < 0.01
        #     self.image[contrib_idx] += 1

        interp_values = np.interp(distance_grid, distance_axis, ph, left=0.00001, right=0.00001)
        self.image += interp_values

        # print(x_idx[0], y_idx[0])
        # for idx in range(len(x_idx)):
        #     self.image[y_idx[idx], x_idx[idx]] = 1.0

        # Show image or plot
        # plt.plot(x_coor, y_coor)
        plt.imshow(self.image)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def sar_radar_both(self, ph_left, ph_right, position, orientation):
        # For now, let's get the distances from the range profile
        distance_axis = np.linspace(self.scan_msg.range_min, self.scan_msg.range_max, self.N_FFT)
        target_idx = ph_left > -1
        target_distance = distance_axis[target_idx]
        image_left = self.image[:,:self.N_FFT]
        image_right = self.image[:,self.N_FFT:]
        yaw = np.arctan2(2.0 * (orientation[3] * orientation[2] + orientation[0] * orientation[1]), 1.0 - 2.0 * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))
        # For now, let's get an "image" of the left side of the map
        # Have the image space be 10m (-5 to 5) by 10m (0 to 10)
        x = np.linspace(-5, 5, self.N_FFT)
        y = np.linspace(0, 10, self.N_FFT)

        X, Y = np.meshgrid(x, y)

        # For each target, figure out which distance is within a certain boundary of the image space
        # In case there's only 1 target
        distance_grid = np.sqrt((X - position[0])**2 + (Y - position[1])**2)
        angle_grid = np.arctan2(Y-position[1],X-position[0])
        angle_idx1 = ((1.308997 + yaw + np.pi) % (2 * np.pi) - np.pi) <= angle_grid
        angle_idx2 = angle_grid <= ((1.832596 + yaw + np.pi) % (2 * np.pi) - np.pi)
        angle_idx = angle_idx1 & angle_idx2

        # try:
        #     for target in target_distance:
        #         contrib_idx = np.abs(distance_grid - target) < 0.01
        #         image_left[contrib_idx] += 1
        # except:
        #     contrib_idx = np.abs(distance_grid - target_distance) < 0.01
        #     image_left[contrib_idx] += 1

        # # Let's do the right side now
        # target_idx = ph_right > -1
        # target_distance = distance_axis[target_idx]

        # # For each target, figure out which distance is within a certain boundary of the image space
        # # In case there's only 1 target
        # distance_grid = np.sqrt((X - position[0])**2 + Y**2)
        # try:
        #     for target in target_distance:
        #         contrib_idx = np.abs(distance_grid - target) < 0.01
        #         image_right[contrib_idx] += 1
        # except:
        #     contrib_idx = np.abs(distance_grid - target_distance) < 0.01
        #     image_right[contrib_idx] += 1

        interp_values_left = np.interp(distance_grid, distance_axis, ph_left, left=0.00001, right=0.00001)
        interp_values_right = np.interp(distance_grid, distance_axis, ph_right, left=0.00001, right=0.00001)
        image_left[angle_idx] += interp_values_left[angle_idx]
        image_right[angle_idx] += interp_values_right[angle_idx]

        # print(x_idx[0], y_idx[0])
        # for idx in range(len(x_idx)):
        #     self.image[y_idx[idx], x_idx[idx]] = 1.0

        # Show image or plot
        # plt.plot(x_coor, y_coor)
        plt.imshow(self.image)
        # plt.imshow(20*np.log10(self.image))
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def sar_radar_map(self, ph_left, ph_right, position, orientation):
        # For now, let's get the distances from the range profile
        distance_axis = np.linspace(self.scan_msg.range_min, self.scan_msg.range_max, self.N_FFT)
        target_idx = ph_left > -1
        target_distance = distance_axis[target_idx]
        image_left = self.image[:,:self.N_FFT]
        image_right = self.image[:,self.N_FFT:]
        yaw = np.arctan2(2.0 * (orientation[3] * orientation[2] + orientation[0] * orientation[1]), 1.0 - 2.0 * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))
        # For now, let's get an "image" of the left side of the map
        # Have the image space be 10m (-5 to 5) by 10m (0 to 10)
        x = np.linspace(-5, 5, self.N_FFT)
        y = np.linspace(-7, 7, self.N_FFT*2)
        # x = np.linspace(-8, 8, self.N_FFT)
        # y = np.linspace(-6, 6, self.N_FFT*2)

        X, Y = np.meshgrid(x, y)

        # For each target, figure out which distance is within a certain boundary of the image space
        # In case there's only 1 target
        distance_grid = np.sqrt((X - position[0])**2 + (Y - position[1])**2)
        angle_grid = np.arctan2(Y-position[1],X-position[0])
        angle_idx1_left = ((1.308997 + yaw + np.pi) % (2 * np.pi) - np.pi) <= angle_grid
        angle_idx2_left = angle_grid <= ((1.832596 + yaw + np.pi) % (2 * np.pi) - np.pi)
        angle_idx_left = angle_idx1_left & angle_idx2_left

        angle_idx1_right = ((4.450590 + yaw + np.pi) % (2 * np.pi) - np.pi) <= angle_grid
        angle_idx2_right = angle_grid <= ((4.974188 + yaw + np.pi) % (2 * np.pi) - np.pi)
        angle_idx_right = angle_idx1_right & angle_idx2_right


        interp_values_left = np.interp(distance_grid, distance_axis, ph_left, left=0.00001, right=0.00001)
        interp_values_right = np.interp(distance_grid, distance_axis, ph_right, left=0.00001, right=0.00001)
        self.image.T[angle_idx_left] += interp_values_left[angle_idx_left]
        self.image.T[angle_idx_right] += interp_values_right[angle_idx_right]

        plt.imshow(self.image)
        # plt.imshow(20*np.log10(self.image))
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def sar_tdbp2(self, ph_left, position, orientation):
        distance_axis = np.linspace(self.scan_msg.range_min, self.scan_msg.range_max, self.N_FFT)
        image_left = self.image[:,:self.N_FFT]
        yaw = np.arctan2(2.0 * (orientation[3] * orientation[2] + orientation[0] * orientation[1]), 1.0 - 2.0 * (orientation[1] * orientation[1] + orientation[2] * orientation[2]))
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        angle_axis = np.linspace(0, np.pi, 256) - np.pi/2
        angle_axis = angle_axis[::-1]

        min_idx = np.argmin(np.abs(angle_axis - yaw))
        image_left[:,min_idx] = ph_left
        # plt.imshow(self.image)
        plt.imshow(20*np.log10(self.image))
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def sar_loop(self):
        if self.pulse_counter >= self.n_pulses:
            # Same as with the control loop, need to make a copy in case the values change later
            temp_ph_left = copy.deepcopy(self.ph_left)
            temp_ph_right = copy.deepcopy(self.ph_right)
            temp_position = copy.deepcopy(self.position)
            temp_orientation = copy.deepcopy(self.orientation)
            # self.sar_lidar()
            # self.sar_radar(temp_ph_left[:,-1], temp_position[:,-1])
            # self.sar_radar_both(temp_ph_left[:,-1], temp_ph_right[:,-1], temp_position[:,-1], temp_orientation[:,-1])
            self.sar_radar_map(temp_ph_left[:,-1], temp_ph_right[:,-1], temp_position[:,-1], temp_orientation[:,-1])
            # self.sar_tdbp2(temp_ph_left[:,-1], temp_position[:,-1], temp_orientation[:,-1])
            # self.sar_bp(temp_ph, temp_position)
            # self.sar_tdbp(temp_ph, temp_position)
            # self.sar_rda(temp_ph, temp_position)
            # self.sar_rma_omega_k(temp_ph, temp_position)
            # self.sar_rc(temp_ph)

        
def main(args=None):
    rclpy.init(args=args)
    node = RIOSARBasic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()