import matplotlib.pyplot as plt
from imageio import imread
from histogram_eq import histogram_eq

I = imread("../billboard/uoft_soldiers_tower_dark.png")
print(I)
J = histogram_eq(I)
print(J)

# plt.imshow(I, cmap = "gray", vmin = 0, vmax = 255)
# plt.show()
# plt.imshow(J, cmap = "gray", vmin = 0, vmax = 255)
# plt.show()
