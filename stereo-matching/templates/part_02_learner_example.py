import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from stereo_disparity_best import stereo_disparity_best

# Load the stereo images.
Il = imread("teddy_image_02.png", as_gray = True)
Ir = imread("teddy_image_06.png", as_gray = True)

# Load the appropriate bounding box.
bboxes = loadmat("bboxes.mat")
bbox = np.array(bboxes["teddy_02"]["bbox"])

Id = stereo_disparity_best(Il, Ir, bbox, 52)
plt.imshow(Id, cmap = "gray")
plt.show()