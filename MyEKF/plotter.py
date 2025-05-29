import matplotlib.pyplot as plt

def plot_eulers(my_euler, sample_euler, error=0, time=0, plot_error=False, ref=0, plot_ref=False):
    """
    Plot Euler angles with optional error overlay using a time vector on the x-axis.
    """
    plt.figure(figsize=(12, 6))
    angles = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time, sample_euler[:, i], c='b', label=f"Onboard Filter {angles[i]}", linestyle=':')
        plt.plot(time, my_euler[:, i], c='r', label=f"My EKF Filter {angles[i]}", linewidth=1)
        if plot_error:
            plt.plot(time, error[:, i], c='g', label=f"Error {angles[i]}", linestyle='--', linewidth=1)
        if plot_ref:
            plt.axhline(y=ref, color='k', linestyle='--', linewidth=1)  # Add horizontal line at y=30
            plt.axhline(y=0, color='k', linestyle='--', linewidth=1)  # Add horizontal line at y=0
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (°)")
    plt.suptitle("3D Euler Orientation (Degrees)")
    plt.show(block=False)



def plot_quaternions(my_quat, sample_quat, time):
    """
    Plot quaternion components over time.
    """
    plt.figure(figsize=(12, 6))
    angles = ['w', 'x', 'y', 'z']
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.plot(time, sample_quat[:, i], c='b', label=f"Onboard Filter {angles[i]}", linestyle=':')
        plt.plot(time, my_quat[:, i], c='r', label=f"My Filter {angles[i]}", linewidth=1)
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Quaternion Component")
    plt.suptitle("Kalman Filter - 3D Orientation Estimation (Quaternion)")
    plt.show(block=False)



