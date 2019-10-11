import numpy as np
from scipy.ndimage.filters import *

def inside_bounds(coord, bounds):
    isOutside = lambda p, a,b: np.cross(p-a, b-a) < 0
    print(coord, bounds[:, 0], bounds[:, 1], np.cross(coord- bounds[:, 0], bounds[:, 1]- bounds[:, 0]))
    if isOutside(coord, bounds[:, 1], bounds[:, 0]):
        return False
    return True

coord = np.array([11, 11])
bounds = np.array([[0, 10, 10, 0], [0, 0, 7, 7]])
print(inside_bounds(coord, bounds))
