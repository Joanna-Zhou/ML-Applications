import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---

    # initialize the 8*9 matrix stacked by 4 2*9 matrixes
    A = []

    for i in range(0, len(I1pts[0])):
        # c[u v 1]^T = H[x y 1]^T where c is a non-zero constant
        # in other words, coordinates of one point in I1pts -> (x, y), I2pts -> (u, v)
        x, y = I1pts[0][i], I1pts[1][i]
        u, v = I2pts[0][i], I2pts[1][i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    # H is obtained by Ah = 0, solved by SVD
    # the solution is V's last column (the column corresponding to sigma=0)
    U, S, V = np.linalg.svd(np.array(A))

    # normalize the matrix so that it is homogenous (i.e. the bottom-right is 1)
    H = V[-1,:].reshape(3, 3)/ V[-1,-1]

    #------------------

    return H, A
