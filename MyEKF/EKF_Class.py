import numpy as np
from maths import get_F, get_P_hat, get_R, get_W
from maths import rot_mat, normalize, compute_jacobian, in_calculate_initial_quaternion

# https://ahrs.readthedocs.io/en/latest/filters/ekf.html#
class EKF_imu:
    def __init__ (self, del_t, accel0, mag0, frame, magnetic_declin_angle, magnetic_inclin_angle, GN, AN, MN):
        self.del_t = del_t
        self.ht = del_t/2
        self.P = np.identity(4)
        self.frame = frame
        self.magnetic_declin_angle_rad = np.radians(magnetic_declin_angle)
        self.magnetic_inclin_angle_rad = np.radians(magnetic_inclin_angle)
        self.frame = frame.upper() # Ensure frame is uppercase for consistency
        gyro_dev = np.array([GN[0]**2, GN[1]**2, GN[2]**2]) # 0.1 - 0.3
        # Spectral noise covariance matrix NOT REQUIRED IF GYRO NOISE SAME
        self.sncm = np.diag(gyro_dev)
        self.accel_dev = np.array([AN[0]**2, AN[1]**2, AN[2]**2]) # 3.6 - 14.4 variance for accelerometer noise
        self.mag_dev = np.array([MN[0]**2, MN[1]**2, MN[2]**2]) # 0.3 - 1.4 variance for gyroscope noise
        self.r = normalize(mag0)
        if self.frame == 'ENU':
            self.g = np.array([0.0, 0.0, 1.0])
            r_east = np.cos(self.magnetic_inclin_angle_rad) * np.sin(self.magnetic_declin_angle_rad)
            r_north = np.cos(self.magnetic_inclin_angle_rad) * np.cos(self.magnetic_declin_angle_rad)
            r_up = -np.sin(self.magnetic_inclin_angle_rad)
            self.r = normalize(np.array([r_east, r_north, r_up]))
        elif self.frame == 'NED':
            self.g = np.array([0.0, 0.0, -1.0])
            r_north = np.cos(self.magnetic_inclin_angle_rad) * np.cos(self.magnetic_declin_angle_rad)
            r_east = np.cos(self.magnetic_inclin_angle_rad) * np.sin(self.magnetic_declin_angle_rad)
            r_down = np.sin(self.magnetic_inclin_angle_rad)
            self.r = normalize(np.array([r_north, r_east, r_down]))
        else:
            raise ValueError("Unsupported frame in EKF_imu init for g and r")

        self.q = in_calculate_initial_quaternion(accel0, mag0,
                                      frame=self.frame,
                                      g_ref=self.g,
                                      m_ref=self.r,
                                      accel_weight=1.5,
                                      mag_weight=0.8)
        
    def predict(self, w):
        self.F = get_F(w, self.ht)  
        self.W = get_W(self.q, self.ht)  
        self.q = self.F @ self.q
        self.q = normalize(self.q)
        self.Q = self.W @ self.sncm @ self.W.T
        self.P = get_P_hat(self.F, self.P, self.Q)

    def correct(self, accel, mag):
        self.z = np.hstack([normalize(accel), normalize(mag)])
        self.a_hat = rot_mat(self.q).T @ self.g
        self.m_hat = rot_mat(self.q).T @ self.r
        self.h = np.hstack([normalize(self.a_hat), normalize(self.m_hat)]) # NO 2*

        self.H = compute_jacobian(self.q, self.g, self.r)
        self.R = get_R(self.accel_dev, self.mag_dev)
        self.v = self.z - self.h
        self.S = (self.H @ self.P @ self.H.T) + self.R
        self.K = self.P @ self.H.T @ np.linalg.pinv(self.S) # Why need pinv to handle 
            # singular matrices if not correcting at every time step????
        self.q = normalize(self.q + (self.K @ self.v))
        self.P = (np.eye(4) - (self.K @ self.H)) @ self.P


