import numpy as np
from maths import calculate_initial_quaternion
from file_reader import load_sensor_data
from plotter import plot_eulers
from scipy.spatial.transform import Rotation as sci_R
from EKF_Class import EKF_imu 

accel, mag, w, orient = load_sensor_data('adat3.log')

mda = 12.7297
# mda = 40.7297
mia = -64.4919
del_t = 0.020

savg = 0
eavg = 20
accel_avg = np.mean(accel[savg:eavg], axis=0)
mag_avg = np.mean(mag[savg:eavg], axis=0)
# data_to_save = np.vstack((accel_avg, mag_avg))
# np.savetxt("avg_sensors.txt", data_to_save)
# loaded_data = np.loadtxt("avg_sensors.txt")
# accel_avg = loaded_data[0]
# mag_avg = loaded_data[1]

EKF = EKF_imu(del_t=del_t, accel0=accel_avg, mag0=mag_avg, frame='ENU', 
            magnetic_declin_angle=mda, magnetic_inclin_angle=mia, GN=0.3, AN=0.5, MN=0.8)


list_q = []
# list_theta = []
r = sci_R.from_euler('xyz', orient[0], degrees = True)
q_scipy = r.as_quat() # Get [x, y, z, w] from scipy
EKF.q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
# EKF.q = np.array([1, 0, 0, 0])
list_q.append(EKF.q)


for i in range(1, len(accel)):
    EKF.predict(w[i])
    EKF.correct(accel[i], mag[i])
    # theta = np.rad2deg(np.arctan2(mag[i,0], mag[i,1]))
    # list_theta.append([0,0,theta])
    # list_q.append(calculate_initial_quaternion(accel[i], mag[i], 'ENU', mda, mia))
    list_q.append(EKF.q)


q_euler = np.array([
    np.degrees(
        sci_R.from_quat( # expects [x, y, z, w]
            np.array([ q_ekf[1], q_ekf[2], q_ekf[3], q_ekf[0] ])
        ).as_euler('xyz')
    )
    for q_ekf in np.array(list_q) # Assuming list_q holds [w,x,y,z] quaternions
])

plot_eulers(q_euler, orient)
