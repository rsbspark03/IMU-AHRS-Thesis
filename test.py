import time
import numpy as np
import quaternion
from collections import deque

def parse_latest_orientation(log_file):
    try:
        with open(log_file, 'rb') as file:
            file.seek(0, 2)  # Move to end of file
            file.seek(max(file.tell() - 2000, 0), 0)  # Move back to read last few lines
            lines = file.readlines()[-2:]  # Read last two lines
            
            for line in reversed(lines):  # Check last, then second-last line
                try:
                    decoded_line = line.decode().strip()
                    vals = [v.strip() for v in decoded_line.split(',')]
                    
                    # Ensure all values are valid floats and exactly 6 values are present
                    if len(vals) == 7:
                        return [float(v) for v in vals]
                    
                except ValueError:
                    continue  # Ignore invalid lines and try the next one

    except Exception as e:
        print(f"Error reading log file: {e}")

    return None  # Instead of raising an error, return None

def orientation(a_data, g_data):
    # print(data)
    if (np.linalg.norm(g_data) < 8):
        inv = np.arctan2(a_data[2], a_data[1])
        degree = np.mod(np.degrees(inv), 360)
        print(degree)
    else:
        print("-")
    return

def axis_angle_to_quat(axis, angle):
    """
    Convert an axis–angle representation to a quaternion using the numpy-quaternion library.
    
    :param axis: A NumPy array of shape (3,) representing the rotation axis (will be normalized).
    :param angle: The rotation angle in radians.
    :return: A quaternion.quaternion object representing the rotation.
    """
    # Normalize the axis to ensure it is a unit vector.
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    x, y, z = axis * sin_half
    return quaternion.quaternion(w, x, y, z)

def gyro_to_delta_quat(omega, dt):
    """
    Convert a gyroscope measurement (angular velocity) to a delta quaternion over time step dt.
    
    :param omega: A NumPy array of shape (3,) representing angular velocity (in radians per second).
    :param dt: The time step (in seconds).
    :return: A quaternion.quaternion object representing the rotation over dt.
    """
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-6:
        # If the rotation is very small, return the identity quaternion (no rotation).
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    angle = omega_norm * dt
    axis = omega / omega_norm
    return axis_angle_to_quat(axis, angle)

def update_orientation(q, omega_meas, bias, dt):
    """
    Update the orientation given a gyroscope measurement by converting the angular velocity
    to a delta quaternion and then multiplying it with the current orientation.
    
    :param q: The current orientation as a quaternion.quaternion object.
    :param omega_meas: The measured angular velocity as a NumPy array of shape (3,).
    :param bias: The gyroscope bias as a NumPy array of shape (3,).
    :param dt: The time step (in seconds).
    :return: The updated orientation as a quaternion.quaternion object.
    """
    # Correct the measured angular velocity by subtracting the bias.
    omega_corrected = omega_meas - bias
    # Compute the delta quaternion representing the small rotation over dt.
    delta_q = gyro_to_delta_quat(omega_corrected, dt)
    # Update the orientation using quaternion multiplication (delta_q is applied first).
    q_new = delta_q * q
    # Normalize to ensure the quaternion remains a unit quaternion.
    q_new = q_new.normalized()
    return q_new


# Initial orientation: identity quaternion (no rotation)
q = np.quaternion(1.0, 0.0, 0.0, 0.0)

# Gyroscope bias (for simplicity, we assume zero bias initially)
bias = np.array([0.0, 0.0, 0.0])

log_file = "putty.log"
queue = deque(maxlen=2)  # Create a deque that stores only the last 2 values
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

while True:
    data = parse_latest_orientation(log_file)
    if data:
        queue.append(data)  # Automatically removes the oldest element if at max size
        if len(queue) < 2:
            continue

        dt = (queue[-1][-1] - queue[-2][-1])/1000
        omega_meas = queue[-1][0:3]
        q = update_orientation(q, omega_meas, bias, dt)
    
        # Print the updated quaternion.
        # print("Updated quaternion:", q)
        # If you prefer a NumPy array representation:
        print("Quat:", quaternion.as_float_array(q))

        # print(f"Queue: {list(queue)}")  # Convert to list for printing
        # print(f"Latest Data: {queue[-1]}")  # Most recent entry
        # if len(queue) > 1:
        #     print(f"Previous Data: {queue[-2]}")  # Second most recent entry
        # orientation(data[3:6], data[0:3])
        
    else:
        # print("Waiting for valid data...")  # Prevent unnecessary error messages
        continue

    time.sleep(0.02)

