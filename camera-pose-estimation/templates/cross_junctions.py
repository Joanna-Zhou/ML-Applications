import numpy as np
from scipy.ndimage.filters import *
import matplotlib.pyplot as plt

# #################################################################################################################################################################
# from imageio import imread
# import matplotlib.pyplot as plt
# from mat4py import loadmat
# #################################################################################################################################################################


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

    # HEPLER FUNCTIONS #######################################################################################################################
    def get_harris_corners(I, sigma, threshold, bbox, window):
        """
        Helper function, computes the Harris response and compares them to
        the threshold to get coordinates of the corners detected

        Parameters:
        -----------
        I           - Single-band (greyscale) image as np.array (e.g., uint8, float).
        sigma       - float, sigma for the Gaussian filter.
        threshold   - float between 0-1, a percentage of the top candidate (i.e. max in R) to compare the Harris response of each pixel to.
        bbox        - the corners defining the bounding box
        window      - tuple (window_x, window_y), the minimum number of pixels separating each corner alone x and y

        Returns:
        --------
        harris_corners  - coordinates of all the detected corners
        """
        ## Compute harris response R
        # First, find 1st derivatives alonge x & y direction
        Ix, Iy = np.zeros(I.shape), np.zeros(I.shape)
        gaussian_filter(I, (sigma, sigma), (0, 1), Ix)
        gaussian_filter(I, (sigma, sigma), (1, 0), Iy)

        # Then, comput componants that make up the Harris matrix
        IxIx = gaussian_filter(Ix * Ix, sigma)
        IxIy = gaussian_filter(Ix * Iy, sigma)
        IyIy = gaussian_filter(Iy * Iy, sigma)

        # Compute determinant and trace to get R
        det = IxIx * IyIy - IxIy * IxIy
        trace = IxIx + IyIy
        R = det/trace # Note that R.shape should = I.shape

        ## Now find a suitable threshold value, and get candidates
        # A suitable threshold = the max R value * threshold percentage chosen
        threshold = np.amax(R) * threshold

        # Compare each pixel's R with  the threshold
        candidates = R > threshold

        # Get the candidates's coordinates and R values, and rank them according to their R values
        candidate_coords = np.array(candidates.nonzero()).T
        candidate_Rs = [R[coord[0], coord[1]] for coord in candidate_coords]
        candidate_rank = np.argsort(candidate_Rs)#[::-1]
        candidate_sorted = candidate_coords[candidate_rank[:2000]] # get only top 1000

        ## The top 48 clusters wtih the most amount of dots will be the ideal points
        # Loop through the canditates to get buckets for clustering
        bound = np.array((bbox[1], bbox[0]))
        print(bound, bbox)
        candidate_buckets = []
        i=0
        dist_threshold = 14 #min(window_x, window_y)
        print(dist_threshold)
        for coord in candidate_sorted:
            i+=1
            too_close = False
            # print("is inside:", inside_bounds(coord, bbox))
            for coord_bucket in candidate_buckets:
                if np.linalg.norm(coord - coord_bucket) < dist_threshold:
                    too_close = True
            # Add the candidate that is inside the bound and also not too close to buckets
            if not too_close and inside_bounds(coord, bound):
                candidate_buckets.append(coord)
        # candidate_buckets = np.array(candidate_buckets)

        # Put each of the canditates into its closest bucket
        candidate_clusters = {i: [] for i in range(len(candidate_buckets))}
        for coord in candidate_sorted:
            if inside_bounds(coord, bound):
                distances = [np.linalg.norm(coord - bucket) for bucket in candidate_buckets]
                closest_bucket = np.argmin(np.array(distances))
                candidate_clusters[closest_bucket].append(coord)

        # Compute the mean of each cluster as the corner
        corners = []
        candidate_clusters = sorted(list(candidate_clusters.values()), key=len)
        for cluster in reversed(candidate_clusters):
            centroid = np.mean(np.array(cluster), axis=0)
            print(centroid)
            corners.append([centroid[1], centroid[0]])
            if len(corners) >= 48:
                return np.array(corners)
        return np.array(corners)


    def inside_bounds(coord, bounds):
        """
        Helper function, determines if the point is inside this 4-edge polygon, returns a boolean
        """
        isOutside = lambda p, a,b: np.cross(p-a, b-a) < 0
        if isOutside(coord, bounds[:, 0], bounds[:, 1]) or \
            isOutside(coord, bounds[:, 1], bounds[:, 2]) or \
            isOutside(coord, bounds[:, 2], bounds[:, 3]) or \
            isOutside(coord, bounds[:, 3], bounds[:, 0]):
            return False
        return True

    def saddle_point(I):
        """
        Helper function, from saddle_point.py, returns a 2*1 array
        """
        m, n = I.shape
        A = np.zeros([m*n, 6])
        y = np.zeros([m*n, 1])
        row = 0
        for j in range(m):
            for i in range(n):
                A[row] = np.array([i*i, i*j, j*j, i, j, 1])
                y[row] = I[j,i]
                row += 1
        params = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, y))[0]
        [a], [b], [c], [d], [e], [f] = params
        A = np.array([[2*a, b], [b, 2*c]])
        b = np.array([[d], [e]])
        print("\n##############\nA:\n", A, "\n##############\n")
        pt = -np.dot(np.linalg.inv(A), b)
        return pt

    def dlt_homography(I1pts, I2pts):
        """
        Helper function, find perspective Homography between two images, returns a 3*3 array
        """
        A = []
        for i in range(0, len(I1pts[0])):
            x, y = I1pts[0][i], I1pts[1][i]
            u, v = I2pts[0][i], I2pts[1][i]
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        U, S, V = np.linalg.svd(np.array(A))
        H = V[-1,:].reshape(3, 3)/ V[-1,-1]
        return H

    def get_dlt_transform(H, pts):
        """
        Helper function, find corresponding point, returns a 2*n array
        """
        new_pts = np.dot(H, np.array([pts[:,0], pts[:,1], np.ones(pts.shape[0])]))
        new_pts = new_pts/new_pts[-1, :]
        return new_pts[:-1, :]

    ##########################################################################################################################################
    [x1, x2, x3, x4], [y1, y2, y3, y4] = bounds[0], bounds[1]

    # Get the harris corners
    window_x = int(abs((min(x2, x3)-max(x1, x4))/14))
    window_y = int(abs((min(y3, y4)-max(y1, y2))/14))
    corners = get_harris_corners(I, sigma=1.0, threshold=0.4, bbox=bounds, window=(window_y, window_x))
    # Pass the points into a dlt homography for sorting
    brect = np.array([[0, 500, 500, 0], [0, 0, 500, 500]])
    H = dlt_homography(bounds, brect)
    H_back = dlt_homography(brect, bounds)
    corners_homo = get_dlt_transform(H, corners).T
    print('\n#################\nFound', corners.shape, 'corners\n#################\n')

    # Sort the corners along y
    corners_sorted = corners_homo[corners_homo[:, 1].argsort()]
    # Sort along x
    count_x, prev_corner = 8, corners_sorted[0, 1]
    for i in range(count_x): # note that count_x should be 6 or 8
        index = corners_sorted[i*count_x:(i+1)*count_x, 0].argsort()
        corners_sorted[i*count_x:(i+1)*count_x, :] = corners_sorted[i*count_x:(i+1)*count_x, :][index]
    corners = get_dlt_transform(H_back, corners_sorted).T
    print('\n#################\nFound', corners.T, corners.shape, 'sorted corners\n#################\n')
    # print(corners)

    # Refine the corner's location with the saddle point detector
    saddles = []
    window_x = 10 #int(window_x/2)
    window_y = 10 #int(window_y/2)
    for corner in corners:
        print('\n#################\ncorner:\n', corner, '\n#################\n')
        patch_x, patch_y = int(corner[0]), int(corner[1])
        I_patch = I[(patch_y-window_y):(patch_y+window_y), (patch_x-window_x):(patch_x+window_x)]
        saddle = saddle_point(I_patch)
        saddles.append(corner+saddle.T[0]-np.array([window_x, window_y]))
    saddles = np.array(saddles)

    Ipts = saddles.T
    # print(saddles, '\n#################\nFound', saddles.shape, 'saddles\n#################\n')

    # # Visualization
    # plt.scatter(corners[:, 0], corners[:, 1], s=50, c='cyan', marker='x')
    # plt.scatter(saddles[:, 0], saddles[:, 1], s=50, c='red', marker='x')
    # plt.imshow(I)
    # plt.show()


    #------------------

    return Ipts

#
# if __name__ == '__main__':
#
#     # Load the boundary.
#     bounds = np.array(loadmat("../targets/bounds.mat")["bpolyh1"])
#
#     # Load the world points.
#     Wpts = np.array(loadmat("world_pts.mat")["world_pts"])
#
#     # Load the example target image.
#     I = imread("../targets/example_target.png")
#     Ipts = cross_junctions(I, bounds, Wpts)
#
# #     # You can plot the points to check!
# #     # print(Ipts)
