import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# load data
true = np.load('../data/waveformTruecis1.npy')
pinn_edge = np.load('../data/waveformPINNcis1_64_32_16_8.npy')
pinn = np.load('../data/data_mid.npy')

# set up grid for plotting
x = np.linspace(0, 2, true.shape[0])
t = np.linspace(0, 4, true.shape[1])

# calculate L" norm error
trueL2 = np.linalg.norm(true, ord=2)
pinnL2 = np.linalg.norm(pinn, ord=2)
L2error = np.linalg.norm((true.flatten() - pinn.flatten()), ord=2) / np.linalg.norm(true.flatten(), ord=2)
abserror = true - pinn
#show L2 norm error

# allocate memory to hold errors
error = np.zeros(true.shape)
mseerror = np.zeros(true.shape)
for i in range(2, error.shape[0]-2):
    for j in range(0, error.shape[1]):
        if abs(true[i, j]) > 1e-12: # avoid division by 0
            error[i,j] = (pinn[i,j] / true[i,j]) - 1
            mseerror[i,j] = (true[i,j] - pinn[i,j]) ** 2

MSE = mseerror.sum() / (mseerror.shape[0] * mseerror.shape[1])

truegrad = np.gradient((true))
pinngrad = np.gradient(pinn)
diffgrad = np.zeros((2,200,400))
diffgrad[0] = truegrad[0] - pinngrad[0]
diffgrad[1] = truegrad[1] - pinngrad[1]

energyerror = np.sum(np.dot(diffgrad[0], diffgrad[0].T) * np.dot(diffgrad[1], diffgrad[1].T))

print('energy error: ', energyerror)
print("MSE: ", MSE)
print("L2 error: ", L2error)
# display data
fig = plt.figure()
gs = GridSpec(3, 2)
plt.subplot(gs[0, 0])
vmin, vmax = -1, +1
plt.pcolormesh(t, x, pinn_edge, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('PINN solution for data at t=0')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[0, 1])
vmin, vmax = -1, +1
plt.pcolormesh(t, x, pinn, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('PINN solution for data at t=2')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[1, 0])
vmin, vmax = -0.05, 0.05
plt.pcolormesh(t, x, true-pinn_edge, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('Error for t=0 data')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[1, 1])
vmin, vmax = -0.05, 0.05
plt.pcolormesh(t, x, abserror, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('Error for t=2 data')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()

plt.subplot(gs[2, :])
vmin, vmax = -0.05, 0.05
plt.pcolormesh(t, x, abserror + (true-pinn_edge), cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
plt.title('Error for t=2 data')
cbar.mappable.set_clim(vmin, vmax)
plt.tight_layout()
plt.savefig('../Graphs/error_test.png')
plt.show()