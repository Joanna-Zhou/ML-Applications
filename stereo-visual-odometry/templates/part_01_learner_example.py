import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dcm_from_rpy import dcm_from_rpy
from triangulate import triangulate

# Camera intrinsic matrices.
Kl = np.array([[500.0, 0.0, 320], [0.0, 500.0, 240.0], [0, 0, 1]])
Kr = Kl

# Camera poses (left, right).
Twl = np.eye(4)
Twl[:3, :3] = dcm_from_rpy([-np.pi/2, 0, 0])  # Tilt for visualization.
Twr = Twl.copy()
Twr[0, 3] = 0.4  # Baseline.

# Image plane points (left, right).
pl = np.array([[241], [237.0]])
pr = np.array([[230], [238.5]])

# Image plane uncertainties (covariances).
Sl = np.eye(2)
Sr = np.eye(2)

[Pl, Pr, P, S] = triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr)

# Visualize...
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.plot(np.array([Twl[0, 3], Pl[0, 0]]), 
        np.array([Twl[1, 3], Pl[1, 0]]),
        np.array([Twl[2, 3], Pl[2, 0]]), 'b-')
ax.plot(np.array([Twr[0, 3], Pr[0, 0]]),
        np.array([Twr[1, 3], Pr[1, 0]]),
        np.array([Twr[2, 3], Pr[2, 0]]), 'r-')
ax.plot(np.array([Pl[0, 0], Pr[0, 0]]),
        np.array([Pl[1, 0], Pr[1, 0]]),
        np.array([Pl[2, 0], Pr[2, 0]]), 'g-')
ax.plot([P[0, 0]], [P[1, 0]], [P[2, 0]], 'bx', markersize = 8)
plt.show()


# Generate initial and transformed points.
C  = dcm_from_rpy(np.array([10, -8, 12])*np.pi/180)
t  = np.array([[0.5], [-0.8], [1.7]])

#Pi = np.array([[1, 2, 3, 4], [7, 3, 4, 8], [9, 11, 6, 3]])
Pi = np.random.rand(3, 10)
Pf = C@Pi + t
Si = np.dstack((1*np.eye(3),)*10)
Sf = np.dstack((1*np.eye(3),)*10)

Tfi_est = estimate_motion_ls(Pi, Pf, Si, Sf)

# Check that the transforms match...
Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))
print(Tfi - Tfi_est)

# Now try with iteration.
#Tfi_est = estimate_motion_ils(Pi, Pf, Si, Sf, 10)
#print(Tfi - Tfi_est)