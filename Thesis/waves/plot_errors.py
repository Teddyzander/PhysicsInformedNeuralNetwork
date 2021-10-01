import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

true = np.load('../data/waveformTruecis1.npy')
pinn50 = np.load('../data/data_it_mid50.npy')
pinn100 = np.load('../data/data_it_mid100.npy')
pinn150 = np.load('../data/data_it_mid150.npy')
pinn200 = np.load('../data/data_it_mid200.npy')
pinn250 = np.load('../data/data_it250.npy')
pinn300 = np.load('../data/data_it300.npy')
pinn400 = np.load('../data/data_it_mid400.npy')
pinn500 = np.load('../data/data_it_mid500.npy')
pinn600 = np.load('../data/data_it_mid600.npy')
pinn750 = np.load('../data/data_it_mid750.npy')
pinn1000 = np.load('../data/data_it1000.npy')
pinn2000 = np.load('../data/data_it2000.npy')
pinn3000 = np.load('../data/data_it3000.npy')

abserror50 = true-pinn50
abserror100 = true-pinn100
abserror150 = true-pinn150
abserror200 = true-pinn200
abserror250 = true-pinn250
abserror300 = true-pinn300
abserror400 = true-pinn400
abserror500 = true-pinn500
abserror600 = true-pinn600
abserror750 = true-pinn750
abserror1000 = true-pinn1000
abserror2000 = true-pinn2000
abserror3000 = true-pinn3000

errs = [abserror50,
        abserror100,
        abserror150,
        abserror200,
abserror400,
        abserror500,
        abserror600,
abserror750
        ]

titles = [
    '50 iterations',
'100 iterations',
'150 iterations',
'200 iterations',
'400 iterations',
'500 iterations',
'600 iterations',
'750 iterations'
]

# set up grid for plotting
x = np.linspace(0, 2, true.shape[0])
t = np.linspace(0, 4, true.shape[1])

fig = plt.figure()
gs = GridSpec(4, 2)
k=0
for i in range(0, 4):
    for j in range(0, 2):
        plt.subplot(gs[i, j])
        vmin, vmax = -0.05, 0.05
        plt.pcolormesh(t, x, errs[k], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        plt.title(titles[k])
        cbar.mappable.set_clim(vmin, vmax)
        plt.tight_layout()
        k = k+1

plt.show()