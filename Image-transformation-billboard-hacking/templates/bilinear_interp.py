import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---


    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # the x, y indeices making up the surrounding 4 pixels' coordinates
    y, x = pt[0], pt[1]
    y1, y2, x1, x2 = int(pt[0]), int(pt[0])+1, int(pt[1]), int(pt[1])+1
    # x1, x2, y1, y2 = floor(pt[0]), ceil(pt[0]), floor(pt[1]), ceil(pt[1])
    b11, b12, b21, b22 = I[x1, y1], I[x1, y2], I[x2, y1], I[x2, y2]
    b = (b11 * (x2 - x) * (y2 - y) +
        b21 * (x - x1) * (y2 - y) +
        b12 * (x2 - x) * (y - y1) +
        b22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
    b = int(round(b[0]))

    #------------------

    return b
