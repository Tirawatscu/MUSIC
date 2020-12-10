from arlpy import bf, utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class arrayConfig:
    def __init__(self, numberSensor):
        self.numberSensor = numberSensor

    def LA(self, start, dx):
        x = np.linspace(start, self.numberSensor * dx, dx)
        return x

    def CA(self, radius):
        xy = np.zeros([self.numberSensor, 2])
        for i, rad in enumerate(np.arange(0, 360, 360 / self.numberSensor)):
            xy[i, 0] = radius * np.cos(np.deg2rad(rad))
            xy[i, 1] = radius * np.sin(np.deg2rad(rad))
        print(xy)
        plt.plot(xy[:, 0], xy[:, 1], "o")
        plt.show()
        return xy


class beam1d:
    def __init__(self, td_data, pos, fs, nfft, overlap, beamformer):
        # td_data is 2D data array the index number of row represent the sensor
        # pos is 1D array of position of sensor in linear
        # fs is sampling rate
        self.td_data = td_data
        self.pos = pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.beamformer = beamformer

    def beampy(self, c, fmin, fmax):
        sd = bf.steering_plane_wave(self.pos, c, np.linspace(0, 2 * np.pi, 181))
        y = bf.broadband(
            self.td_data,
            self.fs,
            self.nfft,
            sd,
            f0=0,
            fmin=fmin,
            fmax=fmax,
            overlap=self.overlap,
            beamformer=self.selectMethod(),
        )
        return y

    def selectMethod(self):
        if self.beamformer == "bartlett":
            return bf.bartlett
        elif self.beamformer == "capon":
            return bf.capon
        elif self.beamformer == "music":
            return bf.music


class beam2d:
    def __init__(self, td_data, pos, fs, nfft, overlap, beamformer):
        # td_data is 2D data array the index number of row represent the sensor
        # pos is 1D array of position of sensor in linear
        # fs is sampling rate
        self.td_data = td_data
        self.pos = pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.beamformer = beamformer

    def beampy(self, c, fmin, fmax):
        sd = bf.steering_plane_wave(
            self.pos, c, utils.linspace2d(-1 * np.pi, np.pi, 181, 0, 0, 1)
        )
        y = bf.broadband(
            self.td_data,
            self.fs,
            self.nfft,
            sd,
            f0=0,
            fmin=fmin,
            fmax=fmax,
            overlap=self.overlap,
            beamformer=self.selectMethod(),
        )
        return y

    def selectMethod(self):
        if self.beamformer == "bartlett":
            return bf.bartlett
        elif self.beamformer == "capon":
            return bf.capon
        elif self.beamformer == "music":
            return bf.music


if __name__ == "__main__":
    td_data = np.array(pd.read_csv("3S3.csv", header=None).T)
    # pos = np.arange(0, 48, 1)
    pos = np.zeros([48, 3])
    for i in range(len(pos)):
        pos[i, 0] = i
    pos1 = [[9.0, 0.0, 0.0], [-4.5, 7.79422863, 0.0], [-4.5, -7.79422863, 0.0]]
    cmin = 50
    cmax = 600
    cstep = 10
    c_serie = np.arange(cmin, cmax + cstep, cstep)
    fmin = 10.0
    fmax = 10.0

    # beam = beam1d(td_data, pos, 1000, 1000, 0, "bartlett")
    # beam = beam2d(td_data, pos, 1000, 1000, 0, "bartlett")
    beam = beam2d(td_data, pos1, 256, 2048, 1024, "bartlett")
    y = np.zeros([181, len(c_serie)])
    for i, c in enumerate(c_serie):
        # y[:, i] = beam.beampy(c, fmin, fmax)[:, 0]) #For 1D
        dat = beam.beampy(c, fmin, fmax)
        y[:, i] = np.mean(dat, 1)  # For 2D
        # print(dat.shape)

    azimuths = np.linspace(-1 * np.pi, np.pi, 181)
    r, theta = np.meshgrid(c_serie, azimuths)
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    cax = ax.contourf(theta, r, y, 150, cmap="jet", shading="gouraud")
    fig.colorbar(cax)
    plt.show()

    """
    beam = beam1d(td_data, pos, 1000, 1000, 0, "bartlett")
    beam2 = beam1d(td_data, pos, 1000, 1000, 0, "music")
    y1 = beam.beampy(200, 1, 4)
    y2 = beam2.beampy(300, 7, 10)

    plt.axes(projection="polar")
    rads = np.linspace(0, 2 * np.pi, 181)

    # plotting the spiral
    for j in range(y1.shape[1]):
        for i, rad in enumerate(rads):
            r = y1[i, j]
            r2 = y2[i, j]
            # plt.polar(rad, r, "g.")
            plt.polar(rad, r2, "b.")

    # display the polar plot
    plt.show()
    """
