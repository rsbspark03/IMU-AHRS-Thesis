import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from file_reader import load_sensor_data

def ellipsoid_fit_and_calibrate(mag_data_raw):
    """
    Fits an ellipsoid to the raw magnetometer data and returns calibration parameters
    and calibrated data.

    The method solves for the parameters of the general quadratic form:
    Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    which can be rewritten using matrix notation for D_mat * params = J_vec.

    Args:
        mag_data_raw (np.ndarray): Nx3 array of raw magnetometer readings (x, y, z).

    Returns:
        tuple: (V_hard_iron, W_inv_soft_iron, mag_data_calibrated)
            V_hard_iron (np.ndarray): 3x1 hard-iron offset vector.
            W_inv_soft_iron (np.ndarray): 3x3 soft-iron inverse correction matrix.
            mag_data_calibrated (np.ndarray): Nx3 array of calibrated magnetometer readings.
    """
    if mag_data_raw.shape[1] != 3:
        raise ValueError("Input data must be an Nx3 array.")
    if mag_data_raw.shape[0] < 10:
        raise ValueError("Insufficient data points for robust fitting. Need at least 10.")

    x = mag_data_raw[:, 0]
    y = mag_data_raw[:, 1]
    z = mag_data_raw[:, 2]

    # Design matrix D_mat
    # Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1 (assuming J = -1)
    D_mat = np.zeros((len(x), 9))
    D_mat[:, 0] = x * x
    D_mat[:, 1] = y * y
    D_mat[:, 2] = z * z
    D_mat[:, 3] = x * y # Note: some formulations use 2*xy, adjust Q matrix accordingly if so
    D_mat[:, 4] = x * z
    D_mat[:, 5] = y * z
    D_mat[:, 6] = x
    D_mat[:, 7] = y
    D_mat[:, 8] = z

    # J_vec is a vector of ones (because we set J = -1 and moved it to the RHS)
    J_vec = np.ones(len(x))

    # Solve the linear system D_mat * params = J_vec using least squares
    # params = [A, B, C, D, E, F, G, H, I]^T
    try:
        params, _, _, _ = np.linalg.lstsq(D_mat, J_vec, rcond=None)
    except np.linalg.LinAlgError:
        print("Least squares fitting failed. Check data quality and quantity.")
        return np.zeros(3), np.eye(3), mag_data_raw # Return uncalibrated

    # Extract ellipsoid parameters
    # The quadratic form is x_raw^T * Q_fit * x_raw + U_fit^T * x_raw + J_fit_const = 0
    # where J_fit_const = -1 (from our setup J_vec=ones)
    A, B, C, D_coeff, E_coeff, F_coeff, G_coeff, H_coeff, I_coeff = params

    # Form the symmetric matrix Q_fit
    # If D_mat used xy, xz, yz, then Q off-diagonals are D/2, E/2, F/2
    Q_fit = np.array([
        [A, D_coeff / 2.0, E_coeff / 2.0],
        [D_coeff / 2.0, B, F_coeff / 2.0],
        [E_coeff / 2.0, F_coeff / 2.0, C]
    ])

    # Form the vector U_fit
    U_fit = np.array([G_coeff, H_coeff, I_coeff])

    J_fit_const = -1.0 # Because we solved D_mat * params = 1

    # Calculate hard-iron offset V_hard_iron
    try:
        Q_fit_inv = np.linalg.inv(Q_fit)
    except np.linalg.LinAlgError:
        print("Matrix Q_fit is singular. Calibration failed.")
        return np.zeros(3), np.eye(3), mag_data_raw

    V_hard_iron = -0.5 * np.dot(Q_fit_inv, U_fit)

    # Calculate the constant term k for the centered ellipsoid equation:
    # x_centered^T * Q_fit * x_centered + k = 0
    k = np.dot(V_hard_iron.T, np.dot(Q_fit, V_hard_iron)) + np.dot(U_fit.T, V_hard_iron) + J_fit_const

    # The equation for the centered ellipsoid is x_centered^T * (Q_fit / -k) * x_centered = 1
    # So, M = Q_fit / -k
    if k >= 0: # Should be negative for a real ellipsoid from this formulation
        print(f"Warning: Constant term k is non-negative ({k:.4f}). Ellipsoid fit might be poor or data is not suitable.")
        # Attempt to proceed, but results might be unreliable
        if k == 0 : k = -1e-9 # Avoid division by zero with a tiny negative k
    
    M_transform = Q_fit / -k

    # Find the soft-iron correction matrix W_inv_soft_iron such that
    # M_transform = W_inv_soft_iron^T * W_inv_soft_iron
    # We use eigenvalue decomposition: M_transform = R * Lambda * R^T
    # Then W_inv_soft_iron = sqrt(Lambda) * R^T
    try:
        eigen_vals, eigen_vecs_R = np.linalg.eig(M_transform)
    except np.linalg.LinAlgError:
        print("Eigenvalue decomposition failed. Calibration failed.")
        return V_hard_iron, np.eye(3), mag_data_raw - V_hard_iron # Return only hard-iron corrected

    if np.any(eigen_vals <= 0):
        print(f"Warning: Non-positive eigenvalues found in M_transform ({eigen_vals}). "
              "Soft-iron correction may be unreliable. Check data quality.")
        # Attempt to use absolute values for sqrt, or handle appropriately
        # For simplicity here, if any are non-positive, soft-iron might not work well.
        # A more robust approach might regularize M_transform or use Cholesky with checks.
        # Let's proceed but be cautious. Setting W_inv to identity if problematic.
        sqrt_lambda = np.sqrt(np.abs(eigen_vals))
        # W_inv_soft_iron = np.eye(3) # Fallback
    else:
        sqrt_lambda = np.sqrt(eigen_vals)
        
    W_inv_soft_iron = np.dot(np.diag(sqrt_lambda), eigen_vecs_R.T)
    
    # Ensure W_inv_soft_iron is real if complex numbers arose from sqrt of small negatives
    if np.iscomplexobj(W_inv_soft_iron):
        print("Warning: Soft-iron matrix became complex, taking real part.")
        W_inv_soft_iron = np.real(W_inv_soft_iron)


    # Apply calibration
    mag_data_calibrated = np.zeros_like(mag_data_raw)
    for i in range(len(mag_data_raw)):
        centered_mag = mag_data_raw[i, :] - V_hard_iron
        mag_data_calibrated[i, :] = np.dot(W_inv_soft_iron, centered_mag)
        
    # Optional: Scale calibrated data to have an average magnitude (e.g., 1 or mean of raw norms)
    # This step makes the calibrated data approximate a unit sphere, or a sphere of desired radius.
    # For now, W_inv_soft_iron is designed to map to a sphere whose "radii" are 1 along principal axes.
    # Let's normalize the final output to have an average norm of 1 for better visualization as a unit sphere.
    norms_calibrated = np.linalg.norm(mag_data_calibrated, axis=1)
    avg_norm_calibrated = np.mean(norms_calibrated)
    if avg_norm_calibrated > 1e-6 : # Avoid division by zero
         mag_data_calibrated /= avg_norm_calibrated


    return V_hard_iron, W_inv_soft_iron, mag_data_calibrated

def apply_calibration(mag_data_raw, V_offset, W_inv_matrix):
    """
    Applies pre-calculated hard and soft iron calibration parameters to raw data.

    Args:
        mag_data_raw (np.ndarray): Nx3 array of raw magnetometer readings.
        V_offset (np.ndarray): 3x1 hard-iron offset vector.
        W_inv_matrix (np.ndarray): 3x3 soft-iron inverse correction matrix.

    Returns:
        np.ndarray: Nx3 array of calibrated magnetometer readings.
    """
    if mag_data_raw.ndim == 1: # Single sample
        centered_mag = mag_data_raw - V_offset
        calibrated_mag = np.dot(W_inv_matrix, centered_mag)
    else: # Batch of samples
        calibrated_mag = np.zeros_like(mag_data_raw)
        for i in range(len(mag_data_raw)):
            centered_mag = mag_data_raw[i, :] - V_offset
            calibrated_mag[i, :] = np.dot(W_inv_matrix, centered_mag)
    
    # Normalize the output to have an average norm of 1 if desired (consistent with fitting output)
    if calibrated_mag.ndim > 1 and len(calibrated_mag) > 0 :
        norms_calibrated = np.linalg.norm(calibrated_mag, axis=1)
        avg_norm_calibrated = np.mean(norms_calibrated)
        if avg_norm_calibrated > 1e-6 :
             calibrated_mag /= avg_norm_calibrated
    elif calibrated_mag.ndim == 1: # single sample, normalize its own norm to 1
        norm = np.linalg.norm(calibrated_mag)
        if norm > 1e-6:
            calibrated_mag /= norm
            
    return calibrated_mag

def visualize_magnetometer_data(raw_data, calibrated_data, title_prefix=""):
    """
    Visualizes raw and calibrated magnetometer data in 3D scatter plots.
    """
    fig = plt.figure(figsize=(12, 6))

    # Plot Raw Data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2], c='r', marker='.', label='Raw Data')
    ax1.set_xlabel('Mag X (raw)')
    ax1.set_ylabel('Mag Y (raw)')
    ax1.set_zlabel('Mag Z (raw)')
    ax1.set_title(f'{title_prefix}Raw Magnetometer Readings')
    ax1.legend()
    # Try to make axes scales equal to see the ellipsoidal shape
    ax1.axis('equal')


    # Plot Calibrated Data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(calibrated_data[:, 0], calibrated_data[:, 1], calibrated_data[:, 2], c='b', marker='.', label='Calibrated Data')
    ax2.set_xlabel('Mag X (calibrated)')
    ax2.set_ylabel('Mag Y (calibrated)')
    ax2.set_zlabel('Mag Z (calibrated)')
    ax2.set_title(f'{title_prefix}Calibrated Magnetometer Readings')
    ax2.legend()
    # Try to make axes scales equal to see the spherical shape (approx unit sphere)
    ax2.axis('equal')
    
    # Set limits for calibrated plot to be around a unit sphere if normalization was to 1
    max_lim = np.max(np.abs(calibrated_data)) * 1.1
    if max_lim < 1.0: max_lim = 1.2 # Ensure reasonable scale if data is small
    ax2.set_xlim([-max_lim, max_lim])
    ax2.set_ylim([-max_lim, max_lim])
    ax2.set_zlim([-max_lim, max_lim])


    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    _, raw_mag_demo_data, _, _ = load_sensor_data('putty.log')
    print("Generated raw magnetometer data for demonstration.")
    print(f"Shape of raw data: {raw_mag_demo_data.shape}")

    # 2. Perform Calibration
    print("\nStarting magnetometer calibration...")
    V_cal, W_inv_cal, mag_calibrated_demo_data = ellipsoid_fit_and_calibrate(raw_mag_demo_data)

    if V_cal is not None:
        print("\nCalibration Parameters Found:")
        print(f"Hard-Iron Offset (V_cal): {V_cal}")
        print(f"Soft-Iron Correction Matrix (W_inv_cal):\n{W_inv_cal}")

        # 3. Visualize
        print("\nVisualizing data...")
        visualize_magnetometer_data(raw_mag_demo_data, mag_calibrated_demo_data, title_prefix="Demo ")

        # 4. How to use parameters for new data:
        # Suppose you have a new raw sample or a new batch of raw data
        # new_raw_sample = np.array([some_x, some_y, some_z])
        # calibrated_sample = apply_calibration(new_raw_sample, V_cal, W_inv_cal)
        # print(f"\nExample: Calibrating a new sample {new_raw_sample} -> {calibrated_sample}")
        
        # Test with a single point from the demo data
        test_raw_point = raw_mag_demo_data[0]
        test_calibrated_point_manual = apply_calibration(test_raw_point, V_cal, W_inv_cal)
        print(f"\nTest: Raw point {test_raw_point.round(2)}")
        print(f"      Calibrated by fit func: {mag_calibrated_demo_data[0].round(2)}")
        print(f"      Calibrated by apply_calibration: {test_calibrated_point_manual.round(2)}")
        
    else:
        print("Calibration failed. Cannot visualize.")