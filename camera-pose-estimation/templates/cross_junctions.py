import numpy as np
from scipy.ndimage.filters import *
import matplotlib.pyplot as plt

##################################################################################################################################################################
from imageio import imread
import matplotlib.pyplot as plt
from mat4py import loadmat
##################################################################################################################################################################


def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar
    calibration target, where the target is bounded in the image by the
    specified quadrilateral. The number of cross-junctions identified
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    #--- FILL ME IN ---
    Ipts = np.zeros((2, 48))

    plt.imshow(I)
    plt.scatter(199, 168, s=500, c='red', marker='x')
    plt.scatter(570, 476, s=500, c='red', marker='x')

    plt.show()

    #
    # def dlt_homography(I1pts, I2pts):
    #     """
    #     Helper function, find perspective Homography between two images.
    #     """
    #     A = []
    #     for i in range(0, len(I1pts[0])):
    #         # c[u v 1]^T = H[x y 1]^T where c is a non-zero constant
    #         # In other words, coordinates of one point in I1pts -> (x, y), I2pts -> (u, v)
    #         x, y = I1pts[0][i], I1pts[1][i]
    #         u, v = I2pts[0][i], I2pts[1][i]
    #         A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
    #         A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    #     H = scipy.linalg.null_space(A)
    #     H = H.reshape(3, 3)/ H[-1]
    #     return H, A
    #
    # def bilinear_interp(I, pt):
    #     """
    #     Helper function, performs bilinear interpolation for a given image point.
    #     """
    #     y, x = pt[0], pt[1]
    #     y1, y2, x1, x2 = int(pt[0]), int(pt[0])+1, int(pt[1]), int(pt[1])+1
    #     b11, b12, b21, b22 = I[x1, y1], I[x1, y2], I[x2, y1], I[x2, y2]
    #     b = (b11 * (x2 - x) * (y2 - y) +
    #         b21 * (x - x1) * (y2 - y) +
    #         b12 * (x2 - x) * (y - y1) +
    #         b22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
    #     return int(round(b[0]))
    #
    #
    # # Apply gaussian blur to the image
    # sigma = 1.5
    # I = gaussian_filter(I, sigma)
    # print(Wpts)
    #
    # # Transfor this polygon bounding box to a rectangular one
    # rectangle = np.array([[0, 450, 450, 0], [0, 0, 350, 350]])
    # (H, A) = dlt_homography(rectangle, bounds)
    # Irect = np.zeros([450, 350])
    # for u in range(0, 450):
    #     for v in range(0, 350):
    #         I_pts = np.dot(H, np.array([u, v, 1]))
    #         [x, y, one] = I_pts / I_pts[-1]
    #         intensity = bilinear_interp(I, np.array([[x, y]]).T)
    #         Irect[v][u] = (intensity, intensity, intensity)
    #
    # Irect = Irect.astype(np.uint8)
    # plt.imshow(Irect)
    # plt.show()
    # Slide through the image for saddle points
    # bound = [[199 583 570 178], [168 190 476 465]]

    #------------------



    return Ipts


if __name__ == '__main__':

    # Load the boundary.
    bpoly = np.array(loadmat("../targets/bounds.mat")["bpolyh1"])

    # Load the world points.
    Wpts = np.array(loadmat("../targets/world_pts.mat")["world_pts"])

    # Load the example target image.
    I = imread("../targets/example_target.png")
    Ipts = cross_junctions(I, bpoly, Wpts)

    # You can plot the points to check!
    # print(Ipts)
