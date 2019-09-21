import numpy as np
from dlt_homography import dlt_homography

if __name__ == "__main__":
    # Input point correspondences (4 points).
    I1pts = np.array([[5, 220, 220,   5], \
                      [1,   1, 411, 411]])
    I2pts = np.array([[375, 420, 420, 450], \
                      [ 20,  20, 300, 290]])

    (H, A) = dlt_homography(I1pts, I2pts)
    print('H = \n', H)

    I1pts = np.array([[5, 220, 220,   5], \
                      [1,   1, 411, 411], \
                      [1,   1,   1,   1]])
    I2pts = np.array([[375, 420, 420, 450], \
                     [ 20,  20, 300, 290], \
                     [1,   1,   1,   1]])
    print('\nI1pts * H = \n', np.dot(H, I1pts))

# Don't forget: the homography operates on homogeneous points!
