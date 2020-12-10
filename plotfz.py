from bfpy import beam2d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

td_data = np.array(pd.read_csv("Most.csv", header=None).T)
# pos = np.arange(0, 48, 1)
pos = np.zeros([48, 3])
for i in range(len(pos)):
    pos[i, 0] = i
pos1 = [[4.0, 0.0, 0.0], [-2.0, 3.46410, 0.0], [-2.0, -3.46410, 0.0]]
cmin = 50
cmax = 800
cstep = 10
c_serie = np.arange(cmin, cmax + cstep, cstep)
fmin = 10.0
fmax = 30.0
df = 1
f_serie = np.arange(fmin, fmax + df, df)

# beam = beam1d(td_data, pos, 1000, 1000, 0, "bartlett")
# beam = beam2d(td_data, pos, 1000, 1000, 0, "bartlett")
beam = beam2d(td_data, pos1, 256, 2048, 1024, "bartlett")
azimuths = np.linspace(-1 * np.pi, np.pi, 181)
r, theta = np.meshgrid(c_serie, azimuths)
y = np.zeros([len(f_serie), 181, len(c_serie)])

fig, axs = plt.subplots(
    int(np.ceil(len(f_serie) / 5)),
    5,
    figsize=(15, 6),
    facecolor="w",
    edgecolor="k",
    subplot_kw=dict(projection="polar"),
)
fig.subplots_adjust(hspace=0.5, wspace=0.001)
axs = axs.ravel()
vmax = 0
vmin = 0
for j, f in enumerate(f_serie):
    print("perform as frequency = %.2f Hz" % f)
    for i, c in enumerate(c_serie):
        # y[:, i] = beam.beampy(c, fmin, fmax)[:, 0]) #For 1D
        dat = beam.beampy(c, f, f)
        y[j, :, i] = np.mean(dat, 1)  # For 2D
        if np.max(np.mean(dat, 1)) > vmax:
            vmax = np.max(np.mean(dat, 1))
        if np.min(np.mean(dat, 1)) > vmin:
            vmin = np.min(np.mean(dat, 1))

for j, f in enumerate(f_serie):
    axs[j].contourf(
        theta, r, y[j, :, :], 150, vmin=0, vmax=vmax, cmap="jet", shading="gouraud"
    )
    axs[j].set_title("f = %.2f Hz" % f)
plt.show()
