import numpy as np
from scipy.spatial.transform import Rotation as sci_R

def load_sensor_data(file_path, offset, end):
    accel_list = []
    magneto_list = []
    gyro_list = []
    orient_list = []
    times_list = []
    quat_list = []

    reading_started = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Start processing only after the READING line
            if not reading_started:
                if "READING" in line:
                    reading_started = True
                continue

            # Skip empty or invalid lines
            split_line = line.split(',')
            if len(split_line) != 17:
                continue

            try:
                values = list(map(float, split_line))
                accel_list.append(values[0:3])
                magneto_list.append(np.array(values[3:6]) * 1000)  # Convert µT to nT
                gyro_list.append(np.radians(values[6:9]))  # Convert to radians
                times_list.append(values[9])
                orient_list.append(values[10:13])  # Rearranged
                quat_list.append(values[13:17])
            except ValueError:
                continue  # Skip malformed lines

    # Convert lists to numpy arrays
    accel = np.array(accel_list)
    magneto = np.array(magneto_list)
    # magneto[:, 2] = -1*magneto[:,2]

    gyro = np.array(gyro_list)
    orient = np.array(orient_list)
    # print(orient.shape)
    orient[:, 2] = 360 - (orient[:, 2]+90)
    dt = np.mean(np.diff(np.array(times_list)))
    orient_quat = np.array(quat_list)

    return accel[offset:end], magneto[offset:end], gyro[offset:end], orient[offset:end], dt, orient_quat[offset:end] 

    
def rearrange_data(array, pos0, pos1, pos2, sign0=1, sign1=1, sign2=1):
    aug_array = np.zeros_like(array)
    aug_array[:, 0] = sign0*array[:, pos0]
    aug_array[:, 1] = sign1*array[:, pos1]
    aug_array[:, 2] = sign2*array[:, pos2]
    return aug_array
   


