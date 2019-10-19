import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6), dtype=np.float64)                # Jacobian.

    #--- FILL ME IN ---
    def f(K, H, Wpt):
        """
        Helper function, returns [u, v] from [u, v, 1] = f(K, H, Wpt) from the estimated H
        """
        HWpt = np.linalg.inv(H).dot( np.vstack( (Wpt, 1 )) )
        KHWpt = K.dot(HWpt[:-1])
        f = KHWpt/KHWpt[-1]
        return f[:-1]

    # Camera intrinsic matrix K
    K = np.array([[564.9, 0,     337.3],
                  [0,     564.3, 226.5],
                  [0,     0,     1    ]])
    # Initial guess of the location/orientation variables in Euler's vectors
    E = epose_from_hpose(Twcg)

    # Update E through building H form the current E, using H to get Jacobian, and update E with it
    for iter in range(maxIters):
        # Build current H from current E
        H = hpose_from_epose(E)
        # Get delta of E by summing up the J^T @ J and J^T @ e(x) for each referece point
        A, b = np.zeros([6, 6], dtype=np.float64), np.zeros([6, 1], dtype=np.float64)
        for i in range(Ipts.shape[1]):
            # print('Iteration', iter, 'Point', i, ':')
            J = find_jacobian(K, H, Wpts[:, i].reshape([3,1]));
            A += J.T.dot(J);
            error = Ipts[:, i].reshape([2,1]) - f(K, H, Wpts[:, i].reshape([3,1]))
            b += J.T.dot(error)
        delta_E = np.linalg.inv(A).dot(b)
        # With delta, we can update
        # print("E:", E)
        E += delta_E

    # At the end, convert back to homogenous form
    Twc = hpose_from_epose(E)

    #------------------

    return Twc

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])

    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1

    return T
