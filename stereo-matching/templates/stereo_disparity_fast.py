import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *


###########################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from mat4py import loadmat
# from imageio import imread
# from stereo_disparity_score import stereo_disparity_score
###########################################################

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    def add_padding(I, padding):
        """Helper function, copy the border pixels to the image paddings"""
        # print(I.shape)
        I = np.hstack((I[:, :padding], I)) # the left columns
        I = np.hstack((I, I[:, -padding:])) # the right columns
        I = np.vstack((I[:padding, :], I)) # the top rows
        I = np.vstack((I, I[-padding:, :])) # the bottom rows
        # print(I.shape)
        return I


    # Deifne parameters
    kernel_size = 13
    half_kernel = int(kernel_size/2)
    maxd -= 1

    # Initialize Id
    Id = np.zeros(Il.shape)
    Il, Ir = add_padding(Il, half_kernel), add_padding(Ir, half_kernel)

    # Preprocess the images by adding paddings
    # Recall: x axis >>, y axis vv, I[y, x]
    x_r1, x_r2 = half_kernel, Ir.shape[-1]+half_kernel
    x_l1, x_l2 = min(bbox[0, 0], bbox[0, 1])+half_kernel,  max(bbox[0, 0], bbox[0, 1])+1+half_kernel
    y_1, y_2 = min(bbox[1, 0], bbox[1, 1])+half_kernel,  max(bbox[1, 0], bbox[1, 1]+1)+1+half_kernel
    Il, Ir = add_padding(Il, half_kernel), add_padding(Ir, half_kernel)

    # Loop through the images and fill in Id
    for row in range(y_1, y_2):
        # Get all the windows from the right image along this row, stored in window_r_list as np.arrays
        window_r_list = [] # will be 1 * width of r
        for col_r in range(x_r1, x_r2):
            window_r = Ir[row-half_kernel:row+half_kernel+1, col_r-half_kernel:col_r+half_kernel+1]
            window_r_list.append(window_r)

        # Loop through all the windows from the left image and store their correspondending x_r
        for col_l in range(x_l1, x_l2):
            SAD_list = [] # will be 1 * maxd
            window_l = Il[row-half_kernel:row+half_kernel+1, col_l-half_kernel:col_l+half_kernel+1]
            # Get this window's corresponding x_r thru comparing with the right windows
            # starting from x_l and ends at maxd pixels to the right of it
            for window_r in window_r_list[col_l-x_r1-maxd : col_l-x_r1+1]:
                SAD_list.append(np.mean(np.absolute(window_l-window_r)))

            # Compute disparsity = x_l - x_r
            x_r = np.argmin(np.array(SAD_list)) + col_l - maxd
            disparsity = col_l - x_r
            Id[row-half_kernel][col_l-half_kernel] = disparsity

    #------------------

    return Id

###################################################################
# if __name__ == '__main__':
#     bboxes = loadmat("/images/bboxes.mat")
#     # Load the stereo images.
#     # Il = imread("../images/cones_image_02.png", as_gray = True)
#     # Ir = imread("../images/cones_image_06.png", as_gray = True)
#     # It = imread("../images/cones_disp_02.png", as_gray = True)

#     Il = imread("/images/teddy_image_02.png", as_gray = True)
#     Ir = imread("/images/teddy_image_06.png", as_gray = True)
#     It = imread("/images/teddy_disp_02.png", as_gray = True)
#     bbox = np.array(bboxes["teddy_02"]["bbox"])

#     Id = stereo_disparity_fast(Il, Ir, bbox, 52)
#     print("Score:", stereo_disparity_score(It, Id, bbox))
#     plt.imshow(Id, cmap = "gray")
#     plt.show()
###################################################################
