# Quaternion-Based Orientation Estimation for Human Pose Tracking Using EKF and IMU Sensors

This repository contains the full implementation of an Extended Kalman Filter (EKF) for estimating 3D orientation using a 9-DOF inertial measurement unit (IMU) that includes an accelerometer, gyroscope, and magnetometer. The project is targeted at human body pose estimation applications and is built using:

- Arduino Mega for real-time IMU data acquisition
- Python for EKF implementation, data logging, and visualization
- C++ (Arduino Framework) for microcontroller firmware
- BNO055 Bosch IMU

## Project Structure:

EKF-AHRS/
- arduino/
- MyEKF/
- log_data/
- README.md

## Features:

- Extended Kalman Filter for 3D orientation (quaternion-based)
- Wahba’s problem used for initial quaternion estimation
- Real-time serial communication between Arduino and Python
- Sensor fusion of accelerometer, gyroscope, and magnetometer data
- Optional magnetometer bias estimation
- Offline logging, playback, and analysis
- Time-series plots for debugging and evaluation

## Dependencies include:
- numpy
- scipy
- matplotlib


## EKF Overview:

The system state is represented by a 4D vector:
x = [q]
where:
- q is the orientation quaternion

The filter fuses data at each time step using prediction and update steps based on the IMU measurements. Wahba's problem is solved to initialize the quaternion using gravity and magnetic vectors.


TODO:

- Add magnetometer calibration routine
- Add gyroscope bias state
- Improve real-time visualization
- Integrate with human pose estimation framework (e.g., OpenPose)

Author:

Ruchir Sinh Bais\
Undergraduate Mechatronics Engineering Student, University of Sydney\
LinkedIn: https://www.linkedin.com/in/ruchirsbais2023/ \
Email: ruchirsbais@gmail.com

License:

This project is licensed under the MIT License. See LICENSE for details.
