import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

wave_0 = np.load("../data/2d_homogeneous_membrane.npy")
x_int = np.linspace(0, 2, wave_0.shape[0])
y_int = np.linspace(0, 3, wave_0.shape[1])
ims = []

X, Y = np.meshgrid(x_int, y_int)
fps = 10
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for i in range(0, wave_0.shape[2]):
    surf = ax.plot_surface(X, Y, wave_0[:, :, i].T, linewidth=0, antialiased=False,
                           shade=True, alpha=0.5, color='black')
    ims.append([surf])
    print(i)

ax.set_zlim(-2.5, 2.5)
ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)

plt.show()
