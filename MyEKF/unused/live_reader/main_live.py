import numpy as np
# from maths import calculate_initial_quaternion, accel_to_quaternion_enu
from file_reader import load_sensor_data
from plotter import plot_eulers
from scipy.spatial.transform import Rotation as sci_R
from EKF_Class import EKF_imu 
import serial
import time

accel, mag, w, orient = load_sensor_data('data3.txt')

mda = 12.7297
mia = -64.4919
del_t = 0.020


EKF = EKF_imu(del_t=del_t, accel0=accel[0], mag0=mag[0], frame='ENU', 
              magnetic_declin_angle=mda, magnetic_inclin_angle=mia, GN=0.03, AN=0.5, MN=0.8)


list_q = []
r = sci_R.from_euler('xyz', orient[0], degrees = True)
q_scipy = r.as_quat() # Get [x, y, z, w] from scipy
EKF.q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
list_q.append(EKF.q)
# -------------------------------------------------------------------------------
ser = serial.Serial('COM3', 115200, timeout=1)

# Wait for a valid line
line = ''
while not isinstance(line, list) or len(line) < 13:
    try:
        line = ser.readline().decode(errors='ignore').strip().split(',')
        print("Le what? ", end="")
    except:
        continue

list_o = []

start = time.time()
end = time.time()
time_elapsed = start - end
while time_elapsed < 10:
    # print([float(line[3]))
    EKF.predict(np.radians([float(line[6]), float(line[7]), float(line[8])]))
    # EKF.correct(line[0:3], line[3:6])
    list_q.append(EKF.q)
    list_o.append([float(line[6]), float(line[7]), float(line[8])])
    end = time.time()
    time_elapsed = start - end
    print(time_elapsed)
    line = ser.readline().decode(errors='ignore').strip().split(',')
    
list_o = np.array(list_o)

# ------------------------------------------------------------------------------
q_euler = np.array([
    np.degrees(
        sci_R.from_quat( # expects [x, y, z, w]
            # Convert EKF's [w, x, y, z] to scipy's [x, y, z, w]
            np.array([ q_ekf[1], q_ekf[2], q_ekf[3], q_ekf[0] ])
        ).as_euler('xyz')
    )
    for q_ekf in np.array(list_q) # Assuming list_q holds [w,x,y,z] quaternions
])

plot_eulers(q_euler, orient)