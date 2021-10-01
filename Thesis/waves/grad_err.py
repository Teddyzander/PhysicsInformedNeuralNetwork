import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# load data
true = np.load('../data/waveformTruecis1.npy')
pinn = np.load('../data/waveformPINNcis1_64_32_16_8.npy')

# set up grid for plotting
x = np.linspace(0, 2, true.shape[0])
t = np.linspace(0, 4, true.shape[1])

# calc error
truegrad = np.gradient(true, 1/100)
pinngrad = np.gradient(pinn, 1/100)
truegrad_x = truegrad[0]
truegrad_t = truegrad[1]
pinn_x = pinngrad[0]
pinn_t = pinngrad[1]

error = (np.sum((truegrad[0] + truegrad[1])**2) - (np.sum((pinngrad[0] + pinngrad[1])**2)))

print(error)

absrange_x = np.max(np.abs(truegrad[0]- pinngrad[0]))
absrange_t = np.max(np.abs(truegrad[1]- pinngrad[1]))
relrange_x = np.max(np.abs((truegrad[0] / pinngrad[0]) - 1))
relrange_t = np.max(np.abs((truegrad[0] / pinngrad[0]) - 1))


fig = plt.figure()
gs = GridSpec(3, 2)
plt.subplot(gs[0, 0])
vmin, vmax = -2.5, 2.5
plt.pcolormesh(t, x, pinn_x, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('pinn_x')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[0, 1])
vmin, vmax = -2.5, 2.5
plt.pcolormesh(t, x, pinn_t, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('pinn_t')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[1, 0])
vmin, vmax = -2.5, 2.5
plt.pcolormesh(t, x, truegrad[0], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('u_x(t,x)')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[1, 1])
vmin, vmax = -2.5, 2.5
plt.pcolormesh(t, x, truegrad[1], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('u_t(t,x)')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[2, 0])
vmin, vmax = -0.2, 0.2
plt.pcolormesh(t, x, truegrad_x - pinn_x, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('u_x error')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[2, 1])
vmin, vmax = -0.2, 0.2
plt.pcolormesh(t, x, truegrad_t - pinn_t, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('u_t error')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.show()