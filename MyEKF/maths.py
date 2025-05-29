import numpy as np
from scipy.spatial.transform import Rotation as R

def get_F(w, ht):
    F_mat = np.array([[1,        -ht*w[0],   -ht*w[1],   -ht*w[2]],
                      [ht*w[0],  1,          ht*w[2],    -ht*w[1]],
                      [ht*w[1],  -ht*w[2],   1,          ht*w[0] ],
                      [ht*w[2],  ht*w[1],    -ht*w[0],    1       ]])
    return F_mat

def get_W(q, ht):
    W_mat = ht*np.array([[   -q[1], -q[2], -q[3]],
                         [    q[0], -q[3],  q[2]],
                         [    q[3],  q[0], -q[1]],
                         [   -q[2],  q[1],  q[0]]])
    return W_mat

def get_P_hat(F, P, Q):
    return (F @ P @ F.T) + Q

def rot_mat(q_hat):
    qw, qx, qy, qz = q_hat
    # Rotation matrix C(q_hat)
    C_q = np.array([
        [qw**2 + qx**2 - qy**2 - qz**2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), qw**2 - qx**2 + qy**2 - qz**2, 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw**2 - qx**2 - qy**2 + qz**2]
    ])
    return C_q

def normalize(array):
    norm = np.linalg.norm(array)
    if norm == 0: 
       return array
    return array / norm

def compute_jacobian(q, g, r):
    qw, qx, qy, qz = q
    gx, gy, gz = g
    rx, ry, rz = r
    # Construct the first form of the Jacobian (expanded form)
    jacobian_expanded = 2 * np.array([
        [(gx*qw)+(gy*qz)-(gz*qy), (gx*qx)+(gy*qy)+(gz*qz), (-gx*qy)+(gy*qx)-(gz*qw), (-gx*qz)-(gy*qw)+(gz*qx)],
        [(-gx*qz)+(gy*qw)+(gz*qx), (gx*qy)-(gy*qx)+(gz*qw), (gx*qx)+(gy*qy)+(gz*qz), (-gx*qw)-(gy*qz)-(gz*qy)],
        [(gx*qy)-(gy*qx)+(gz*qw), (gx*qz)-(gy*qw)-(gz*qx), (-gx*qw)-(gy*qz)-(gz*qy), (gx*qx)+(gy*qy)+(gz*qz)],
        [(rx*qw)+(ry*qz)-(rz*qy), (rx*qx)+(ry*qy)+(rz*qz), (-rx*qy)+(ry*qx)-(rz*qw), (-rx*qz)-(ry*qw)+(rz*qx)],
        [(-rx*qz)+(ry*qw)+(rz*qx), (rx*qy)-(ry*qx)+(rz*qw), (rx*qx)+(ry*qy)+(rz*qz), (-rx*qw)-(ry*qz)-(rz*qy)],
        [(rx*qy)-(ry*qx)+(rz*qw), (rx*qz)-(ry*qw)-(rz*qx), (-rx*qw)-(ry*qz)-(rz*qy), (rx*qx)+(ry*qy)+(rz*qz)]
    ])
   
    return jacobian_expanded 

def get_R(sigma_a_squared, sigma_m_squared):
    # Create 3x3 identity matrices
    I3 = np.eye(3)

    # Create 3x3 zero matrices
    O3 = np.zeros((3, 3))

    # Construct the R matrix
    R = np.block([
        [sigma_a_squared * I3, O3],
        [O3, sigma_m_squared * I3]
    ])

    return R

def in_calculate_initial_quaternion(accel, mag, frame='ENU',
                                  g_ref=None, m_ref=None,
                                  accel_weight=1.0, mag_weight=0.5):
    accel = normalize(accel)
    mag = normalize(mag)

    v_b = [accel]
    v_r = [g_ref if g_ref is not None else np.array([0, 0, 1.0] if frame == 'ENU' else [0, 0, -1.0])]
    weights = [accel_weight]

    if mag_weight > 0:
        v_b.append(mag)
        v_r.append(m_ref)
        weights.append(mag_weight)
    # print(f'v_b: {v_b}\nv_r: {v_r}')
    rotation, _ = R.align_vectors(v_r, v_b, weights=weights)
    q = rotation.as_quat()
    return np.roll(q, 1)  # [w, x, y, z]

def fix_quat_unknown_frame(quat_array):

    quat_array = np.asarray(quat_array)
    q_rot = R.from_euler('z', -90, degrees=True)  # Rotation around Z

    quat_fixed = []
    for q in quat_array:
        r = R.from_quat(np.roll(q, -1))  # Convert [w, x, y, z] -> [x, y, z, w]
        r_fixed = q_rot * r
        q_out = r_fixed.as_quat()  # Output in [x, y, z, w]
        q_out = np.roll(q_out, 1)  # Convert to [w, x, y, z]

        # Swap x and y (i.e., q_out[1] and q_out[2])
        q_out[0], q_out[1], q_out[2], q_out[3] = q_out[0], q_out[2], -q_out[1], -q_out[3]

        quat_fixed.append(q_out)

    return np.array(quat_fixed)
