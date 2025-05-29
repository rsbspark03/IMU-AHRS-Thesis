# -*- coding: utf-8 -*-

import numpy as np
from ..common.quaternion import Quaternion
from ..common.orientation import ecompass
from ..common.orientation import acc2q
from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..common.mathfuncs import skew

from ..utils.core import _assert_numerical_iterable
from ..utils.core import _assert_numerical_positive_variable

class EKF:
    """
    Extended Kalman Filter to estimate orientation as Quaternion.

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs.filters import EKF
    >>> from ahrs.common.orientation import acc2q
    >>> ekf = EKF()
    >>> num_samples = 1000              # Assuming sensors have 1000 samples each
    >>> Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    >>> Q[0] = acc2q(acc_data[0])       # First sample of tri-axial accelerometer
    >>> for t in range(1, num_samples):
    ...     Q[t] = ekf.update(Q[t-1], gyr_data[t], acc_data[t])

    The estimation is simplified by giving the sensor values at the
    construction of the EKF object. This will perform all steps above and store
    the estimated orientations, as quaternions, in the attribute ``Q``.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data)
    >>> ekf.Q.shape
    (1000, 4)

    In this case, the measurement vector, set in the attribute ``z``, is equal
    to the measurements of the accelerometer. If extra information from a
    magnetometer is available, it will also be considered to estimate the
    attitude.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data)
    >>> ekf.Q.shape
    (1000, 4)

    For this case, the measurement vector contains the accelerometer and
    magnetometer measurements together: ``z = [acc_data, mag_data]`` at each
    time :math:`t`.

    The most common sampling frequency is 100 Hz, which is used in the filter.
    If that is different in the given sensor data, it can be changed too.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, frequency=200.0)

    Normally, when using the magnetic data, a referencial magnetic field must
    be given. This filter computes the local magnetic field in Munich, Germany,
    but it can also be set to a different reference with the parameter
    ``mag_ref``.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, magnetic_ref=[17.06, 0.78, 34.39])

    If the full referencial vector is not available, the magnetic dip angle, in
    degrees, can be also used.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, magnetic_ref=60.0)

    The initial quaternion is estimated with the first observations of the
    tri-axial accelerometers and magnetometers, but it can also be given
    directly in the parameter ``q0``.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, q0=[0.7071, 0.0, -0.7071, 0.0])

    Measurement noise variances must be set from each sensor, so that the
    Process and Measurement Covariance Matrix can be built. They are set in an
    array equal to ``[0.3**2, 0.5**2, 0.8**2]`` for the gyroscope,
    accelerometer and magnetometer, respectively. If a different set of noise
    variances is used, they can be set with the parameter ``noises``:

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, noises=[0.1**2, 0.3**2, 0.5**2])

    or the individual variances can be set separately too:

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, var_acc=0.3**2)

    This class can also differentiate between NED and ENU frames. By default it
    estimates the orientations using the NED frame, but ENU is used if set in
    its parameter:

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, frame='ENU')

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in nT
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    frame : str, default: 'NED'
        Local tangent plane coordinate frame. Valid options are right-handed
        ``'NED'`` for North-East-Down and ``'ENU'`` for East-North-Up.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    magnetic_ref : float or numpy.ndarray
        Local magnetic reference.
    noises : numpy.ndarray
        List of noise variances for each type of sensor. Default values:
        ``[0.3**2, 0.5**2, 0.8**2]``.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. NOT required
        if ``frequency`` value is given.

    """
    def __init__(self,
            gyr: np.ndarray = None,
            acc: np.ndarray = None,
            mag: np.ndarray = None,
            frequency: float = 100.0,
            frame: str = 'NED',
            **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = frequency
        self.frame: str = frame                          # Local tangent plane coordinate frame
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.q0: np.ndarray = kwargs.get('q0')
        self.P: np.ndarray = kwargs.get('P', np.identity(4))    # Initial state covariance
        self.R: np.ndarray = self._set_measurement_noise_covariance(**kwargs)
        self._set_reference_frames(kwargs.get('magnetic_ref'), self.frame)
        self._assert_validity_of_inputs()
        # Process of data is given
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all(self.frame)

    def _set_measurement_noise_covariance(self, **kw) -> np.ndarray:
        default_noises = kw.get('noises', [0.3**2, 0.5**2, 0.8**2])
        _assert_numerical_iterable(default_noises, 'Spectral noise variances')
        default_noises = np.copy(default_noises)
        if default_noises.ndim != 1:
            raise ValueError(f"Spectral noise variances must be given in a 1-dimensional array. Got {default_noises.ndim} dimensions instead.")
        if default_noises.size != 3:
            raise ValueError(f"Spectral noise variances must be given in a 1-dimensional array with 3 elements. Got {default_noises.size} elements instead.")
        self.noises = [kw.get(label, value) for label, value in zip(['var_gyr', 'var_acc', 'var_mag'], default_noises)]
        self.g_noise, self.a_noise, self.m_noise = self.noises
        return np.diag(np.repeat(self.noises[1:], 3))

    def _set_reference_frames(self, mref: float, frame: str = 'NED') -> None:
        if not isinstance(frame, str):
            raise TypeError(f"Parameter 'frame' must be a string. Got {type(frame)}.")
        if frame.upper() not in ['NED', 'ENU']:
            raise ValueError(f"Invalid frame '{frame}'. Try 'NED' or 'ENU'")
        # Magnetic Reference Vector
        if mref is None:
            # Local magnetic reference of Munich, Germany
            from ..common.constants import MUNICH_LATITUDE, MUNICH_LONGITUDE, MUNICH_HEIGHT
            from ..utils.wmm import WMM
            wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
            self.m_ref = np.array([wmm.X, wmm.Y, wmm.Z]) if frame.upper() == 'NED' else np.array([wmm.Y, wmm.X, -wmm.Z])
        elif isinstance(mref, bool):
            raise TypeError("Invalid magnetic reference. Try a float or a numpy.ndarray.")
        elif isinstance(mref, (int, float)):
            cd, sd = cosd(mref), sind(mref)
            self.m_ref = np.array([cd, 0.0, sd]) if frame.upper() == 'NED' else np.array([0.0, cd, -sd])
        elif isinstance(mref, (list, tuple, np.ndarray)):
            self.m_ref = np.copy(mref)
        else:
            raise TypeError(f"mref must be given as a float, list, tuple or NumPy array. Got {type(mref)}")
        if self.m_ref.ndim != 1:
            raise ValueError(f"mref must be given as a 1-dimensional array. Got {self.m_ref.ndim} dimensions instead.")
        if self.m_ref.size != 3:
            raise ValueError(f"mref must be given as a 1-dimensional array with 3 elements. Got {self.m_ref.size} elements instead.")
        for item in self.m_ref:
            if not isinstance(item, (int, float)):
                raise TypeError(f"mref must be given as a 1-dimensional array of floats. Got {type(item)} instead.")
        self.m_ref /= np.linalg.norm(self.m_ref)
        # Gravitational Reference Vector
        self.a_ref = np.array([0.0, 0.0, 1.0]) if frame.upper() == 'NED' else np.array([0.0, 0.0, -1.0])

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        for item in ["frequency", "Dt"]:
            _assert_numerical_positive_variable(getattr(self, item), item)
        for item in ['q0', 'P', 'R']:
            if self.__getattribute__(item) is not None:
                if isinstance(self.__getattribute__(item), bool):
                    raise TypeError(f"Parameter '{item}' must be an array of numeric values.")
                if not isinstance(self.__getattribute__(item), (list, tuple, np.ndarray)):
                    raise TypeError(f"Parameter '{item}' is not an array. Got {type(self.__getattribute__(item))}.")
                self.__setattr__(item, np.copy(self.__getattribute__(item)))
        if self.q0 is not None:
            if self.q0.shape != (4,):
                raise ValueError(f"Parameter 'q0' must be an array of shape (4,). It is {self.q0.shape}.")
            if not np.allclose(np.linalg.norm(self.q0), 1.0):
                raise ValueError(f"Parameter 'q0' must be a versor (norm equal to 1.0). Its norm is equal to {np.linalg.norm(self.q0)}.")
        for item in ['P', 'R']:
            if self.__getattribute__(item).ndim != 2:
                raise ValueError(f"Parameter '{item}' must be a 2-dimensional array.")
            m, n = self.__getattribute__(item).shape
            if m != n:
                raise ValueError(f"Parameter '{item}' must be a square matrix. It is {m}x{n}.")

    def _compute_all(self, frame: str) -> np.ndarray:
        """
        Estimate the quaternions given all sensor data.

        Attributes ``gyr``, ``acc`` MUST contain data. Attribute ``mag`` is
        optional.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        _assert_numerical_iterable(self.gyr, 'Angular velocity vector')
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        self.gyr = np.array(self.gyr)
        self.acc = np.array(self.acc)
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        Q[0] = self.q0
        if self.mag is not None:
            ###### Compute attitude with MARG architecture ######
            _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
            self.mag = np.array(self.mag)
            if self.mag.shape != self.gyr.shape:
                raise ValueError("mag and gyr are not the same size")
            if self.q0 is None:
                Q[0] = ecompass(self.acc[0], self.mag[0], frame=frame, representation='quaternion')
            Q[0] /= np.linalg.norm(Q[0])
            # EKF Loop over all data
            for t in range(1, num_samples):
                Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
            return Q
        ###### Compute attitude with IMU architecture ######
        if self.q0 is None:
            Q[0] = acc2q(self.acc[0])
        Q[0] /= np.linalg.norm(Q[0])
        # EKF Loop over all data
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t])
        return Q

    def Omega(self, x: np.ndarray) -> np.ndarray:
        """
        Omega operator.

        Given a vector :math:`\\mathbf{x}\\in\\mathbb{R}^3`, return a
        :math:`4\\times 4` matrix of the form:

        .. math::
            \\boldsymbol\\Omega(\\mathbf{x}) =
            \\begin{bmatrix}
                0 & -\\mathbf{x}^T \\\\
                \\mathbf{x} & -\\lfloor\\mathbf{x}\\rfloor_\\times
            \\end{bmatrix} =
            \\begin{bmatrix}
                0   & -x_1 & -x_2 & -x_3 \\\\
                x_1 &    0 &  x_3 & -x_2 \\\\
                x_2 & -x_3 &    0 &  x_1 \\\\
                x_3 &  x_2 & -x_1 & 0
            \\end{bmatrix}

        This operator is constantly used at different steps of the EKF.

        Parameters
        ----------
        x : numpy.ndarray
            Three-dimensional vector.

        Returns
        -------
        Omega : numpy.ndarray
            Omega matrix.
        """
        return np.array([
            [0.0,  -x[0], -x[1], -x[2]],
            [x[0],   0.0,  x[2], -x[1]],
            [x[1], -x[2],   0.0,  x[0]],
            [x[2],  x[1], -x[0],   0.0]])

    def f(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Linearized function of Process Model (Prediction.)

        .. math::
            \\mathbf{f}(\\mathbf{q}_{t-1}, \\Delta t) = \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} =
            \\begin{bmatrix}
            q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
            q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
            q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
            q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        omega : numpy.ndarray
            Angular velocity, in rad/s.
        dt : float
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Linearized estimated quaternion in **Prediction** step.
        """
        Omega_t = self.Omega(omega)
        return (np.identity(4) + 0.5*dt*Omega_t) @ q

    def dfdq(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Jacobian of linearized predicted state.

        .. math::
            \\mathbf{F} = \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1})}{\\partial\\mathbf{q}} =
            \\begin{bmatrix}
            1 & - \\frac{\\Delta t}{2} \\omega_x & - \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_z\\\\
            \\frac{\\Delta t}{2} \\omega_x & 1 & \\frac{\\Delta t}{2} \\omega_z & - \\frac{\\Delta t}{2} \\omega_y\\\\
            \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_z & 1 & \\frac{\\Delta t}{2} \\omega_x\\\\
            \\frac{\\Delta t}{2} \\omega_z & \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_x & 1
            \\end{bmatrix}

        Parameters
        ----------
        omega : numpy.ndarray
            Angular velocity in rad/s.
        dt : float
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        F : numpy.ndarray
            Jacobian of state.
        """
        x = 0.5*dt*omega
        return np.identity(4) + self.Omega(x)

    def h(self, q: np.ndarray) -> np.ndarray:
        """
        Measurement Model

        If only the gravitational acceleration is used to correct the
        estimation, a vector with 3 elements is used:

        .. math::
            \\mathbf{h}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_x (q_w^2 + q_x^2 - q_y^2 - q_z^2) + g_y (q_wq_z + q_xq_y) + g_z (q_xq_z - q_wq_y) \\\\
            g_x (q_xq_y - q_wq_z) + g_y (q_w^2 - q_x^2 + q_y^2 - q_z^2) + g_z (q_wq_x + q_yq_z) \\\\
            g_x (q_wq_y + q_xq_z) + g_y (q_yq_z - q_wq_x) + g_z (q_w^2 - q_x^2 - q_y^2 + q_z^2)
            \\end{bmatrix}

        If the gravitational acceleration and the geomagnetic field are used,
        then a vector with 6 elements is used:

        .. math::
            \\mathbf{h}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_x (q_w^2 + q_x^2 - q_y^2 - q_z^2) + g_y (q_wq_z + q_xq_y) + g_z (q_xq_z - q_wq_y) \\\\
            g_x (q_xq_y - q_wq_z) + g_y (q_w^2 - q_x^2 + q_y^2 - q_z^2) + g_z (q_wq_x + q_yq_z) \\\\
            g_x (q_wq_y + q_xq_z) + g_y (q_yq_z - q_wq_x) + g_z (q_w^2 - q_x^2 - q_y^2 + q_z^2) \\\\
            r_x (q_w^2 + q_x^2 - q_y^2 - q_z^2) + r_y (q_wq_z + q_xq_y) + r_z (q_xq_z - q_wq_y) \\\\
            r_x (q_xq_y - q_wq_z) + r_y (q_w^2 - q_x^2 + q_y^2 - q_z^2) + r_z (q_wq_x + q_yq_z) \\\\
            r_x (q_wq_y + q_xq_z) + r_y (q_yq_z - q_wq_x) + r_z (q_w^2 - q_x^2 - q_y^2 + q_z^2)
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray
            Predicted Quaternion.

        Returns
        -------
        numpy.ndarray
            Expected Measurements.
        """
        C = Quaternion(q).to_DCM().T
        if self.mag is None:
            return C @ self.a_ref
        return np.r_[C @ self.a_ref, C @ self.m_ref]

    def dhdq(self, q: np.ndarray, mode: str = 'normal') -> np.ndarray:
        """
        Linearization of observations with Jacobian.

        If only the gravitational acceleration is used to correct the
        estimation, a :math:`3\\times 4` matrix:

        .. math::
            \\mathbf{H}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_xq_w + g_yq_z - g_zq_y & g_xq_x + g_yq_y + g_zq_z & -g_xq_y + g_yq_x - g_zq_w & -g_xq_z + g_yq_w + g_zq_x \\\\
            -g_xq_z + g_yq_w + g_zq_x & g_xq_y - g_yq_x + g_zq_w &  g_xq_x + g_yq_y + g_zq_z & -g_xq_w - g_yq_z + g_zq_y \\\\
            g_xq_y - g_yq_x + g_zq_w & g_xq_z - g_yq_w - g_zq_x &  g_xq_w + g_yq_z - g_zq_y &  g_xq_x + g_yq_y + g_zq_z
            \\end{bmatrix}

        If the gravitational acceleration and the geomagnetic field are used,
        then a :math:`6\\times 4` matrix is used:

        .. math::
            \\mathbf{H}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_xq_w + g_yq_z - g_zq_y & g_xq_x + g_yq_y + g_zq_z & -g_xq_y + g_yq_x - g_zq_w & -g_xq_z + g_yq_w + g_zq_x \\\\
            -g_xq_z + g_yq_w + g_zq_x & g_xq_y - g_yq_x + g_zq_w &  g_xq_x + g_yq_y + g_zq_z & -g_xq_w - g_yq_z + g_zq_y \\\\
            g_xq_y - g_yq_x + g_zq_w & g_xq_z - g_yq_w - g_zq_x &  g_xq_w + g_yq_z - g_zq_y &  g_xq_x + g_yq_y + g_zq_z \\\\
            m_xq_w + m_yq_z - m_zq_y & m_xq_x + m_yq_y + m_zq_z & -m_xq_y + m_yq_x - m_zq_w & -m_xq_z + m_yq_w + m_zq_x \\\\
            -m_xq_z + m_yq_w + m_zq_x & m_xq_y - m_yq_x + m_zq_w &  m_xq_x + m_yq_y + m_zq_z & -m_xq_w - m_yq_z + m_zq_y \\\\
            m_xq_y - m_yq_x + m_zq_w & m_xq_z - m_yq_w - m_zq_x &  m_xq_w + m_yq_z - m_zq_y &  m_xq_x + m_yq_y + m_zq_z
            \\end{bmatrix}

        If ``mode`` is equal to ``'refactored'``, the computation is carried
        out as:

        .. math::
            \\mathbf{H}(\\hat{\\mathbf{q}}_t) = 2
            \\begin{bmatrix}
            \\mathbf{u}_g & \\lfloor\\mathbf{u}_g+\\hat{q}_w\\mathbf{g}\\rfloor_\\times + (\\hat{\\mathbf{q}}_v\\cdot\\mathbf{g})\\mathbf{I}_3 - \\mathbf{g}\\hat{\\mathbf{q}}_v^T \\\\
            \\mathbf{u}_r & \\lfloor\\mathbf{u}_r+\\hat{q}_w\\mathbf{r}\\rfloor_\\times + (\\hat{\\mathbf{q}}_v\\cdot\\mathbf{r})\\mathbf{I}_3 - \\mathbf{r}\\hat{\\mathbf{q}}_v^T
            \\end{bmatrix}

        .. warning::
            The refactored mode might lead to slightly different results as it
            employs more and different operations than the normal mode,
            originated by the nummerical capabilities of the host system.

        Parameters
        ----------
        q : numpy.ndarray
            Predicted state estimate.
        mode : str, default: ``'normal'``
            Computation mode for Observation matrix. Options are: ``'normal'``,
            or ``'refactored'``.

        Returns
        -------
        H : numpy.ndarray
            Jacobian of observations.
        """
        if mode.lower() not in ['normal', 'refactored']:
            raise ValueError(f"Mode '{mode}' is invalid. Try 'normal' or 'refactored'.")
        qw, qx, qy, qz = q
        if mode.lower() == 'refactored':
            t = skew(self.a_ref)@q[1:]
            H = np.c_[t, q[1:]*self.a_ref*np.identity(3) + skew(t + qw*self.a_ref) - np.outer(self.a_ref, q[1:])]
            if self.mag is not None:
                t = skew(self.m_ref)@q[1:]
                H_2 = np.c_[t, q[1:]*self.m_ref*np.identity(3) + skew(t + qw*self.m_ref) - np.outer(self.m_ref, q[1:])]
                H = np.vstack((H, H_2))
            return 2.0*H
        v = np.r_[self.a_ref, self.m_ref]
        H = np.array([[ v[0]*qw + v[1]*qz - v[2]*qy, v[0]*qx + v[1]*qy + v[2]*qz, -v[0]*qy + v[1]*qx - v[2]*qw, -v[0]*qz + v[1]*qw + v[2]*qx],
                      [-v[0]*qz + v[1]*qw + v[2]*qx, v[0]*qy - v[1]*qx + v[2]*qw,  v[0]*qx + v[1]*qy + v[2]*qz, -v[0]*qw - v[1]*qz + v[2]*qy],
                      [ v[0]*qy - v[1]*qx + v[2]*qw, v[0]*qz - v[1]*qw - v[2]*qx,  v[0]*qw + v[1]*qz - v[2]*qy,  v[0]*qx + v[1]*qy + v[2]*qz]])
        if self.mag is not None:
            H_2 = np.array([[ v[3]*qw + v[4]*qz - v[5]*qy, v[3]*qx + v[4]*qy + v[5]*qz, -v[3]*qy + v[4]*qx - v[5]*qw, -v[3]*qz + v[4]*qw + v[5]*qx],
                            [-v[3]*qz + v[4]*qw + v[5]*qx, v[3]*qy - v[4]*qx + v[5]*qw,  v[3]*qx + v[4]*qy + v[5]*qz, -v[3]*qw - v[4]*qz + v[5]*qy],
                            [ v[3]*qy - v[4]*qx + v[5]*qw, v[3]*qz - v[4]*qw - v[5]*qx,  v[3]*qw + v[4]*qz - v[5]*qy,  v[3]*qx + v[4]*qy + v[5]*qz]])
            H = np.vstack((H, H_2))
        return 2.0*H

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray = None, dt: float = None) -> np.ndarray:
        """
        Perform an update of the state.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori state describing orientation as quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in nT.
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated state describing orientation as quaternion.

        """
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(gyr, 'Tri-axial gyroscope sample')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        if mag is not None:
            _assert_numerical_iterable(mag, 'Tri-axial magnetometer sample')
        dt = self.Dt if dt is None else dt
        if not np.isclose(np.linalg.norm(q), 1.0):
            raise ValueError("A-priori quaternion must have a norm equal to 1.")
        # Current Measurements
        g = np.array(gyr)                       # Gyroscope data (control vector)
        a = np.array(acc)
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return q
        a /= a_norm
        z = np.copy(a)
        if mag is not None:
            m_norm = np.linalg.norm(mag)
            if m_norm == 0:
                raise ValueError("Invalid geomagnetic field. Its magnitude must be greater than zero.")
            z = np.r_[a, mag/m_norm]
        self.R = np.diag(np.repeat(self.noises[1:] if mag is not None else self.noises[1], 3))
        # ----- Prediction -----
        q_t = self.f(q, g, dt)                  # Predicted State
        F   = self.dfdq(g, dt)                  # Linearized Fundamental Matrix
        W   = 0.5*dt * np.r_[[-q[1:]], q[0]*np.identity(3) + skew(q[1:])]  # Jacobian W = df/dω
        Q_t = self.g_noise * W@W.T              # Process Noise Covariance
        P_t = F@self.P@F.T + Q_t                # Predicted Covariance Matrix
        # ----- Correction -----
        y   = self.h(q_t)                       # Expected Measurement function
        v   = z - y                             # Innovation (Measurement Residual)
        H   = self.dhdq(q_t)                    # Linearized Measurement Matrix
        S   = H@P_t@H.T + self.R                # Measurement Prediction Covariance
        K   = P_t@H.T@np.linalg.inv(S)          # Kalman Gain
        self.P = (np.identity(4) - K@H)@P_t     # Updated Covariance Matrix
        q = q_t + K@v                           # Corrected State
        q /= np.linalg.norm(q)
        return q