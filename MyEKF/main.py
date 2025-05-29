# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from maths import fix_quat_unknown_frame, normalize, in_calculate_initial_quaternion
from file_reader import load_sensor_data
from plotter import plot_eulers, plot_quaternions
from scipy.spatial.transform import Rotation as sci_R
from scipy.io import savemat
from EKF_Class import EKF_imu 

# Load sensor data from file
accel, mag, w, orient, dt, orient_quat = load_sensor_data('log_files/rand.log', 0, -1)
print(f'dt = {dt/1000}')  # Print time step in seconds

# Define noise parameters for gyroscope, accelerometer, and magnetometer
# GN=[0.0018, 0.0011, 0.0013]
# AN=[0.0111, 0.0071, 0.0162]
# MN=[474.9, 480.04, 610.95]
GN=[0.3, 0.3, 0.3]
AN=[0.5, 0.5, 0.5]
MN=[2000, 2000, 2000]

if True:
    # Magnetic declination and inclination angles
    mda = 12.7297
    mia = -64.4919
    del_t = dt/1000  # Convert dt to seconds

    avg = 5  # Number of samples to average for initial state
    accel_avg = np.mean(accel[0:avg], axis=0)  # Average initial accelerometer readings
    mag_avg = np.mean(mag[0:avg], axis=0)      # Average initial magnetometer readings
# Initialize EKF with initial sensor readings and parameters
EKF = EKF_imu(del_t=del_t, accel0=accel_avg, mag0=mag_avg, frame='ENU', 
            magnetic_declin_angle=mda, magnetic_inclin_angle=mia, GN=GN, MN=MN, AN=AN)

list_q = []  # List to store quaternion estimates
list_q.append(EKF.q)  # Append initial quaternion

# Print shapes of sensor data arrays
print(accel.shape)
print(orient.shape)

# Run EKF over all sensor data
for i in range(1, len(accel)):
    EKF.predict(w[i])  # EKF prediction step using gyroscope data
    EKF.correct(accel[i], mag[i])  # EKF correction step using accelerometer and magnetometer
    # Alternative initialization method (commented out)
    # list_q.append(in_calculate_initial_quaternion(accel[i], mag[i], frame=EKF.frame, g_ref=EKF.g, m_ref=EKF.r, accel_weight=1.5, mag_weight=1.0))
    list_q.append(EKF.q)  # Append updated quaternion

# Convert quaternion list to Euler angles in degrees
q_euler = np.array([
    np.degrees(
        sci_R.from_quat(
            [q_ekf[1], q_ekf[2], q_ekf[3], q_ekf[0]]  # Convert to (x, y, z, w) format
        ).as_euler('xyz')  # Convert to Euler angles
    )
    for q_ekf in list_q
])

offset = -11.15  # Yaw offset correction

# Unwrap Euler angles to avoid discontinuities
q_euler = np.unwrap(np.radians(q_euler), axis=0)
q_euler = np.degrees(q_euler)

# Process ground truth orientation data
orient_radians = np.radians(orient)  # Convert to radians
orient_unwrapped = np.unwrap(orient_radians, axis=0)  # Unwrap
orient_unwrapped = np.degrees(orient_unwrapped)  # Convert back to degrees
time = np.arange(len(accel)) * del_t  # Time vector
orient_unwrapped[:, 2] += offset  # Apply yaw offset

if True:
    # Compute absolute errors between EKF and ground truth
    errors_euler = np.abs(np.subtract(orient_unwrapped, q_euler))

    # Compute Mean Absolute Error (MAE)
    mae_euler = np.mean(errors_euler, axis=0)

    # Compute max and min absolute errors
    max_error_euler = np.max(errors_euler, axis=0)
    min_error_euler = np.min(errors_euler, axis=0)

    # Print error statistics for each angle
    angles = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        print(f"{angles[i]}:")
        print(f"  Mean Absolute Error: {mae_euler[i]:.3f}°")
        print(f"  Max Error: {max_error_euler[i]:.3f}°")
        print(f"  Min Error: {min_error_euler[i]:.3f}°")
else:
    errors_euler = [0]

# Optional analysis and saving (commented out)
'''
# orient_unwrapped[:, 1] += -0.5
# q_euler[:, 1] += -0.4

# Calculate standard deviation for each axis
# std_dev_axes = np.std(q_euler, axis=0)

# Calculate the mean for each axis
# mean_axes = np.mean(q_euler, axis=0)

# Print statistics
# print("--- Gyroscope Data Statistics ---")
# print("Mean for each axis (degrees/s):", np.round(mean_axes, 4))
# print("Standard Deviation for each axis (degrees/s):", np.round(std_dev_axes, 4))

# Calculate Coefficient of Variation
# normalized_std_dev_axes = np.where(mean_axes != 0, std_dev_axes / mean_axes, np.inf)

# print("\n--- Normalized Standard Deviation (Coefficient of Variation) ---")
# print(f"X-axis: {normalized_std_dev_axes[0]:.4f} (CV)")
# print(f"Y-axis: {normalized_std_dev_axes[1]:.4f} (CV)")
# print(f"Z-axis: {normalized_std_dev_axes[2]:.4f} (CV)")

# Save results to .mat file
# mdic = {"My_EKF": q_euler, "Onboard_Filter": orient_unwrapped, "Time": time, "label": "experiment"}
# savemat("30z.mat", mdic)
'''

# Plot Euler angles and errors
poffset = 5
plot_eulers(q_euler[poffset:], orient_unwrapped[poffset:], errors_euler[poffset:], 
            time[poffset:], False, 30, False)

# Optional quaternion plot (commented out)
# plot_quaternions(np.array(list_q), fix_quat_unknown_frame(orient_quat), time)

plt.show()  # Display plots
