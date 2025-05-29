Quaternion-Based Orientation Estimation for Human Pose Tracking Using EKF and IMU Sensors

This repository contains the full implementation of an Extended Kalman Filter (EKF) for estimating 3D orientation using a 9-DOF inertial measurement unit (IMU) that includes an accelerometer, gyroscope, and magnetometer. The project is targeted at human body pose estimation applications and is built using:

- Arduino Mega for real-time IMU data acquisition
- Python for EKF implementation, data logging, and visualization
- C++ (Arduino Framework) for microcontroller firmware
- BNO055 Bosch IMU

Project Structure:

EKF-AHRS/
├── arduino/
├── MyEKF/
├── log_data/
└── README.md

Features:

- Extended Kalman Filter for 3D orientation (quaternion-based)
- Wahba’s problem used for initial quaternion estimation
- Real-time serial communication between Arduino and Python
- Sensor fusion of accelerometer, gyroscope, and magnetometer data
- Optional magnetometer bias estimation
- Offline logging, playback, and analysis
- Time-series plots for debugging and evaluation

Setup Instructions:

1. Arduino Setup
- Open arduino/imu_logger.ino in the Arduino IDE.
- Select the correct board and port for your Arduino Mega.
- Upload the firmware.

2. Python Environment

git clone https://github.com/your-username/EKF-AHRS.git
cd EKF-AHRS
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt

Dependencies include:
- numpy
- scipy
- matplotlib
- pyserial
- pandas

3. Run the EKF

Connect the Arduino, then:

python src/ekf.py --port COM3  # Replace COM3 with your serial port

For offline analysis:

python src/plot_results.py --file data/recorded_data.csv

Visualization:

The plot_results.py script produces plots of:
- Quaternion components over time
- Euler angles for interpretability
- Orientation error vs ground truth (if available)

EKF Overview:

The system state is represented by a 7D vector:
x = [ q, b_g ]
where:
- q is the orientation quaternion
- b_g is the gyroscope bias

The filter fuses data at each time step using prediction and update steps based on the IMU measurements. Wahba's problem is solved to initialize the quaternion using gravity and magnetic vectors.

Documentation:

- See notebooks/analysis.ipynb for detailed explanations and plots.
- Inline comments in ekf.py and filters.py explain the mathematical models.

TODO:

- Add magnetometer calibration routine
- Improve real-time visualization
- Integrate with human pose estimation framework (e.g., OpenPose)

Author:

[Your Name]
Undergraduate Mechatronics Engineering Student, University of Sydney
LinkedIn: https://linkedin.com/in/your-profile
Email: your@email.com

License:

This project is licensed under the MIT License. See LICENSE for details.
