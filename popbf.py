from arlpy import bf, utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

td_data = np.array(pd.read_csv("3S3.csv", header=None).T)
pos = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0]]
pos1 = np.arange(0, 48, 1)
a1 = bf.steering_plane_wave(pos, 100, utils.linspace2d(0, 2 * np.pi, 181, 0, 0, 1))
sd = bf.steering_plane_wave(pos1, 200, np.linspace(0, 2 * np.pi, 181))
# print(td_data)
# y = bf.capon(td_data, 256, a1)
y = bf.broadband(
    td_data,
    256,
    2048,
    a1,
    f0=0,
    fmin=3,
    fmax=3,
    overlap=1024,
    beamformer=bf.bartlett,
)
# print(np.linspace(-np.pi / 2, np.pi / 2, 181))
print(y)
print(np.mean(y, 1).shape)
y = np.mean(y, 1)
# print(np.deg2rad(utils.linspace2d(-180, 180, 180, 0, 0, 180)))

plt.axes(projection="polar")


rads = np.linspace(0, 2 * np.pi, 181)

# plotting the spiral
for j in range(y.shape[1]):
    for i, rad in enumerate(rads):
        r = y[i, j]
        plt.polar(rad, r, "g.")

# display the polar plot
plt.show()
