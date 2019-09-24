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
    Ist_pts = np.array([[2, 219, 219, 2], [2, 2, 410, 410]])

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

    print('Shape of Iyd: \n{}\nShape of Ist: \n{}'.format(Iyd.shape, Ist.shape) )
    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)
    plt.imshow(Ist)

    # Compute the perspective homography we need...
    (H, A) = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion
    x_top, x_bottom, y_top, y_bottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][2]
    row_width = y_bottom - y_top+1 # just to aboid using y_array.shape again and again
    for x in range(x_top, x_bottom+1):
        # Vectorization on each row of pixels
        y_array = np.array(range(y_top, y_bottom+1))
        x_array = np.ones(row_width) * x
        # Perform the homography transformation with the helper function above
        Iyd_row = np.concatenate([[x_array], [y_array]])
        Ist_row = homography_transform(Iyd_row, H)

        # Now we have the coordinates, we do the per-pixel modifications
        for i in range(row_width):
            Iyd_row_x, Iyd_row_y, Ist_row_x, Ist_row_y = int(Iyd_row[:,i][1]), int(Iyd_row[:,i][0]), Ist_row[:,i][1], Ist_row[:,i][0]
            # Since the dlt is linear, instead of the contain_point method, I simply checked if the corresponding point
            #   is contained by the rectengule defined the reference points Ist_pts
            if Ist_row_x >= 2 and Ist_row_y >= 2 and Ist_row_x < 410 and Ist_row_y < 219:
                # Perform the bilinear interpolation
                intensity = int(bilinear_interp(Ist, np.array([Ist_row_x, Ist_row_y]).reshape(2, 1)))
                # Change the RGB values on the Iyd picture, replacing it with the interpolated Ist values
                Ihack[Iyd_row_x, Iyd_row_y] = (intensity, intensity, intensity)

    #------------------

    plt.imshow(Ihack)
    plt.show()
    imwrite(Ihack, 'billboard_hacked.png');

    return Ihack
billboard_hack()
