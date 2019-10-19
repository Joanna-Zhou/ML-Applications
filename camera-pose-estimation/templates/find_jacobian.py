import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose.
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    #--- FILL ME IN ---

    def dR_drpy(R):
        """
        Helper function, finds the elememets cprrespnoding to the rotational variables in the Jacobian
        """
        dRdr = R.dot(np.array([[0, 0, 0],
                              [0, 0, -1],
                              [0, 1, 0]]))
        dRdy = np.array([[0, -1, 0],
                         [1, 0, 0],
                         [0, 0, 0]]).dot(R)
        try:
            cp = np.sqrt(1 - R[2,0]*R[2,0])
        except Exception as e:
            print("Chech R matrix: an angle must have been > 1")
        cy, sy = R[:2, 0]/cp
        dRdp = np.array([[0,     0, cy],
                         [0,     0, sy],
                         [-cy, -sy, 0]]).dot(R)
        return dRdr, dRdp, dRdy


    # Extract the camera extrinsic rotation and translation
    R, t = Twc[:3, :3], Twc[:3, -1:];
    dt = Wpt - t;

    # Find each dR or dt over variables
    # Essentially dR and dt are part of df
    #   where f(K, Twc, Wpt) = f_tilde = f normalized by 3rd row
    f = K.dot(R.T).dot(dt)
    f_z = f[-1, 0]

    # To obtain the derivatives, we do chain rule of a fraction
    df = np.zeros([3, 6]) # still in homogeneous form
    df[:, :3] = K.dot(R.T).dot((-np.eye(3)))

    dRdr, dRdp, dRdy = dR_drpy(R)
    df[:, 3:4] = K.dot(dRdr.T).dot(dt)
    df[:, 4:5] = K.dot(dRdp.T).dot(dt)
    df[:, 5:6] = K.dot(dRdy.T).dot(dt)

    df_z = df[-1:, :]

    # Now find J = d(f_tilde)/dq of d(f/f_z)/q, where q = [variables]
    # and ditch the last row to make it Cartesean again
    J  = np.zeros((2, 6), dtype=np.float64)
    J = np.divide((f_z * df - f.dot(df_z)), f_z*f_z)[:-1, :]
    # print('\nJacobian:', J)
    #------------------

    return J
