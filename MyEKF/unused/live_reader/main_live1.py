import numpy as np
from scipy.spatial.transform import Rotation as sci_R
from EKF_Class import EKF_imu 
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Constants and init
mda = 12.7297
mia = -64.4919
del_t = 0.020

# Dummy sensor data for init
from file_reader import load_sensor_data
accel, mag, w, orient = load_sensor_data('data3.txt')

EKF = EKF_imu(del_t=del_t, accel0=accel[0], mag0=mag[0], frame='ENU', 
              magnetic_declin_angle=mda, magnetic_inclin_angle=mia, GN=0.03, AN=0.5, MN=0.8)

r = sci_R.from_euler('xyz', orient[0], degrees=True)
q_scipy = r.as_quat()  # [x, y, z, w]
EKF.q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

# Setup serial
ser = serial.Serial('COM3', 115200, timeout=1)

# Wait for a valid line
line = ''
while not isinstance(line, list) or len(line) < 13:
    try:
        line = ser.readline().decode(errors='ignore').strip().split(',')
    except:
        continue

# Buffers
euler_buf = deque(maxlen=100)

# Setup plot
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=label)[0] for label in ['Roll', 'Pitch', 'Yaw']]
ax.set_ylim(-180, 180)
ax.set_xlim(0, 100)
ax.legend()
ax.set_title("Live EKF Orientation (Euler Angles)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Angle (degrees)")

def update(frame):
    global line
    try:
        line = ser.readline().decode(errors='ignore').strip().split(',')
        if len(line) < 9:
            return lines

        gx, gy, gz = map(float, line[6:9])
        EKF.predict(np.radians([gx, gy, gz]))

        q = EKF.q
        r = sci_R.from_quat([q[1], q[2], q[3], q[0]])  # [x, y, z, w]
        euler_deg = np.degrees(r.as_euler('xyz'))
        euler_buf.append(euler_deg)

        data = np.array(euler_buf)
        x = np.arange(len(data))
        for i in range(3):
            lines[i].set_data(x, data[:, i])
    except Exception as e:
        print("Error reading or processing line:", e)

    return lines

ani = FuncAnimation(fig, update, interval=20, blit=False)
plt.show()
