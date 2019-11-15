import numpy as np
from numpy.linalg import inv, norm


def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline.

    Parameters:
    -----------
    Kl   - 3 x 3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3 x 3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    Cwr, Cwl = Twr[:3, :3], Twl[:3, :3]
    cr, cl = Twr[:3, 3:], Twl[:3, 3:]

    # Compute baseline (right camera translation minus left camera translation).
    b = cr - cl

    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    rayl_unnorm = Cwr @ inv(Kl) @ np.append(pl, 1)
    rayl = rayl_unnorm.reshape((3, 1))/norm(rayl_unnorm)
    rayr_unnorm = Cwl @ inv(Kr) @ np.append(pr, 1)
    rayr = rayr_unnorm.reshape((3, 1))/norm(rayr_unnorm)

    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    bTrl = np.dot(b.T, rayl)
    bTrr = np.dot(b.T, rayr)
    rlTrr = np.dot(rayl.T, rayr)
    ml = (bTrl - bTrr*rlTrr)/(1-rlTrr*rlTrr)
    mr = rlTrr * ml - bTrr

    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = cl + rayl*ml
    Pr = cr + rayr*mr

    # Now fill in with appropriate ray Jacobians. These are
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    def get_Jacobian(r, rhat, C, invK):
         """Helper function that returns the Jacobian with chain rule
         Arguments:
             r {1*3 array} -- rayl or rayr
             rhat {1*3 array} -- rayl or rayr
             C {3*3 array} -- Cwr or Cwl
             invK {3*3 array} -- inv(Kl) or inv(Kr)
         """
         CinvK = C @ invK
         dr_duv = CinvK[:, 0:2]  # 3*2 array

         norm_r = norm(r)
         rhat_rhat_T = rhat.reshape((3, 1)) @ rhat.reshape((1, 3))
         drhat_dr = (np.eye(3) - rhat_rhat_T)/norm_r  # 3*3 array

         return drhat_dr @ dr_duv

    drayl[:, 0:2] = get_Jacobian(rayl_unnorm, rayl, Cwl, inv(Kl))
    drayr[:, 2:4] = get_Jacobian(rayr_unnorm, rayr, Cwr, inv(Kr))

    #------------------
    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
        np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)

    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) +
                                  (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
        np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    #--- FILL ME IN ---

    # 3D point.
    P = (Pl + Pr)/2.0

    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    S_lr = np.zeros((4, 4))
    S_lr[0:2, 0:2] = Sl
    S_lr[2:4, 2:4] = Sr
    S = JP @ S_lr @ JP.T

    #------------------

    return Pl, Pr, P, S
