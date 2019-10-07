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

    I_flat = I.flatten()
    N = len(I_flat)

    # Flatten the 2D matrix to a 1D array and generate a histogram of  "counts of pixels at 256 intensity levels"
    histogram, bins = np.histogram(I.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function at each intensity level and normalize
    cdf = histogram.cumsum()
    cdf = cdf * histogram.max() / cdf.max()

    # Perform an equalization with0 terms ignored
    # The use of numpy.ma.masked_equal is referenced from:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    cdf = np.ma.masked_equal(cdf, 0)
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Read the values into J as uint8
    J = cdf[I]#.astype('uint8')

    #------------------

    return J
