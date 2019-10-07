import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mat4py import loadmat
from cross_junctions import cross_junctions

# Load the boundary.
bpoly = np.array(loadmat("../targets/bounds.mat")["bpolyh1"])

# Load the world points.
Wpts = np.array(loadmat("../targets/world_pts.mat")["world_pts"])

# Load the example target image.
I = imread("../targets/example_target.png")
plt.imshow(I)
plt.scatter(199, 168, s=500, c='red', marker='x')
plt.scatter(570, 476, s=500, c='red', marker='x')

plt.show()

# Ipts = cross_junctions(I, bpoly, Wpts)

# You can plot the points to check!
# print(Ipts)
