import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import scipy.io as scipy

# load data
output = np.load('grid_u.npy').T
real = scipy.loadmat('burgers_shock.mat')['usol']
print(output.shape)
realgrads_t = np.gradient(real, 1/100)[1]
realgrads_x = np.gradient(real, 1/128)[0]
realgrads_tt = np.gradient(realgrads_t, 1/100)[1]
realgrads_xx = np.gradient(realgrads_x, 1/128)[0]
output_t = np.gradient(output, 1/100)[1]
output_x = np.gradient(output, 1/128)[0]

x = np.linspace(-1, 1, output.shape[0])
t = np.linspace(0, 1, output.shape[1])

fig = plt.figure()
gs = GridSpec(2, 1)

plt.subplot(gs[0, 0])
vmin, vmax = -1, 1
plt.pcolormesh(t, x, real, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title(r'$\phi(x, t)$')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[1, 0])
vmin, vmax = -1, 1
plt.pcolormesh(t, x, output, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('PINN solution')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.show()

fig = plt.figure()
gs = GridSpec(3, 1)

plt.subplot(gs[0, 0])
vmin, vmax = -0.01, 0.01
plt.pcolormesh(t, x, real-output, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('absolute error')
cbar.mappable.set_clim(vmin, vmax)

plt.subplot(gs[1, 0])
vmin, vmax = -50, 50
plt.pcolormesh(t, x, realgrads_x, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title(r'$\phi_x$')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[2, 0])
vmin, vmax = -10, 10
plt.pcolormesh(t, x, realgrads_t, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title(r'$\phi_t$')
cbar.mappable.set_clim(vmin, vmax)

plt.show()