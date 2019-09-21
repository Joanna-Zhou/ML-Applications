import numpy as np
import matplotlib.pyplot as plt


def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # Flatten the 2D matrix to a 1D array
    I_flat = I.flatten()

    # Initiate and update the histogram of "counts of pixels at 256 intensity levels"
    histogram = np.zeros(256)
    for pixel in I_flat:
        histogram[pixel] += 1

    # Compute the cumulative distribution function at each intensity level, stored in a list
    cdf = [histogram[0]]
    for bar in histogram:
        cdf.append(cdf[-1] + bar)
    cdf = np.array(cdf)*255/len(I_flat)

    # Transform the new pixel intensities to J which has the same shape of I
    J_flat = cdf[I_flat]
    J = np.reshape(J_flat, I.shape).astype('uint8')

    #------------------

    return J
