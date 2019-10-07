import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    #--- FILL ME IN ---

    m, n = I.shape

    # Muilding the A and b for "minimize Ax-b" based on eqn.4
    A = np.zeros([m*n, 6])
    y = np.zeros([m*n, 1])
    row = 0
    for i in range(m):
        for j in range(n):
            A[row] = np.array([i^2, i*j, j^2, i, j, 1])
            y[row] = I[j][i]
            row += 1

    # Use least square to fit the parameters (ie. solve Ax-y => params=inv(ATA)ATy)
    params = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, y))[0]
    [a], [b], [c], [d], [e], [f] = params

    # Find critical point of the paraboloid
    A = np.array([[2*a, b], [b, 2*c]])
    b = np.array([[d], [e]])
    pt = -np.dot(inv(A), b)


    #------------------

    return pt
