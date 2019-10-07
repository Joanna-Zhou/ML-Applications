# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    -----------

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    def homography_transform(l1pts, H):
        """
        Helper function that returns the transformed coordinates of the input point(s)
        It adds the 3rd dimension of ones, performs the homography transformation by H,
            and normalizes the output coordinates as well as extracting the first 2 coordinates to output

        Parameters:
        -----------
        l1pts  - the 2-by-n numpy array representing the n input points to be transformed
                 for example, np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
        H      - the 3-by-3 numpy array obtained from dlt_homography

        Returns:
        --------
        l2pts  - the 2-by-n numpy array transformed and post-processed from l1pts
        """
        l1pts = np.concatenate( [ l1pts, np.ones((1,l1pts.shape[1])) ] )
        l2pts = np.dot(H, l1pts)
        l2pts = l2pts[:-1]/l2pts[-1]

        return l2pts

    # print('Shape of Iyd: \n{}\nShape of Ist: \n{}'.format(Iyd.shape, Ist.shape) )
    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)
    # plt.imshow(Ist)

    # Compute the perspective homography we need...
    (H, A) = dlt_homography(Iyd_pts, Ist_pts)

    # First just extract some csonstants to avoid having to write this ugly thing again and again
    x1, x2, y1, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][2] #404, 490, 38, 354
    # x1, x2, y1, y2 = Iyd_pts[0][0], Iyd_pts[0][1], Iyd_pts[1][0], Iyd_pts[1][2] #2, 218, 2, 409
    print(x1, x2, y1, y2)
    # Create the bounding box
    Iyd_path = Path(Iyd_pts.T)

    # Loop through each point and transform
    for u in range(x1, x2):
        for v in range(y1, y2):
            if Iyd_path.contains_point(np.array([u, v])):
                Ist_pts = np.dot(H, np.array([u, v, 1]))
                [x, y, one] = Ist_pts / Ist_pts[-1]
                # if (x > xx1-2 and x < xx2+1 and y > yy1-2 and y < yy2+1):
                intensity = bilinear_interp(Ist, np.array([[x, y]]).T)
                Ihack[v][u] = (intensity, intensity, intensity)

    Ihack = Ihack.astype(np.uint8)

    #------------------
    # plt.imshow(Ihack)
    # plt.show()

    return Ihack

# billboard_hack()
