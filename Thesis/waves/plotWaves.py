import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

wave_0 = np.load("../data/2d_homogeneous_membrane.npy")
x_int = np.arange(0, 2, 2/wave_0.shape[0])
y_int = np.arange(0, 3, 3/wave_0.shape[1])
ims = []

X, Y = np.meshgrid(x_int, y_int)
"""
fps = 10
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i in range(0, wave_0.shape[2]):
    surf = ax.plot_surface(X, Y, wave_0[:, :, i].T, linewidth=0, antialiased=False,
                           shade=True, alpha=0.5, color='black')
    ims.append([surf])
    print(i)

"""
levels = np.arange(-2.5, 2.75, 0.25)
fig, ax = plt.subplots(2, 3)

plt.subplot(2,3,1)
plt.title('Membrane shape at t=0')
cs = plt.contourf(X, Y, wave_0[:, :, 0].T, levels=levels, cmap=cm.jet)
plt.subplot(2,3,2)
plt.title('Membrane shape at t=0.05')
cs = plt.contourf(X, Y, wave_0[:, :, 1].T, levels=levels, cmap=cm.jet)
plt.subplot(2,3,3)
plt.title('Membrane shape at t=0.1')
cs = plt.contourf(X, Y, wave_0[:, :, 2].T, levels=levels, cmap=cm.jet)
plt.subplot(2,3,4)
plt.title('Membrane shape at t=0.15')
cs = plt.contourf(X, Y, wave_0[:, :, 3].T, levels=levels, cmap=cm.jet)
plt.subplot(2,3,5)
plt.title('Membrane shape at t=0.2')
cs = plt.contourf(X, Y, wave_0[:, :, 4].T, levels=levels, cmap=cm.jet)
plt.subplot(2,3,6)
plt.title('Membrane shape at t=0.25')
cs = plt.contourf(X, Y, wave_0[:, :, 5].T, levels=levels, cmap=cm.jet)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
CB = fig.colorbar(cs, cax=cbar_ax)
CB.set_label('Membrane height', rotation=270)

plt.show()
