import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# solution for a 1d vibrating membrane u_tt = c^2u_xx
# 2 units long rectangle with c=6 and IC f(x)=x(2-x) with u=0 at the edges.
# u_t(t=0)=0

def u_sol(x, t, n, c=1):

    u = 0
    for i in range(1, n+1):
        num = -(8*np.pi*i * np.sin(np.pi*i) +
               16 * np.cos(np.pi * i)-16)
        denom = np.pi**3 * i**3
        coef1 = np.cos((c * i * np.pi * t) / 2)
        coef2 = np.sin((i * np.pi * x) / 2)

        term = ((num / denom) * coef1 * coef2)

        u = u + term
    return u


x = np.linspace(0, 2, 200)
t = np.linspace(0, 4, 400)
"""
wave = np.zeros((x.__len__(), t.__len__()))
for i in range(0, x.__len__()):
    for j in range(0, t.__len__()):
        ans = u_sol(x[i], t[j], 100)
        wave[i, j] = ans

    print(i)

np.save('../data/waveformTruecis1.npy', np.asarray(wave))
"""
wave = np.load('../data/waveformPINNcis1_64_64.npy')
fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 4)
plt.subplot(gs[0, :])
vmin, vmax = -1, +1
plt.pcolormesh(t, x, wave, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(vmin, vmax)
t_cross_sections = [0, 1, 2, 3]
for i, t_cs in enumerate(t_cross_sections):
    plt.subplot(gs[1, i])
    if i == 0:
        plt.plot(x, wave[:, (i)])
    else:
        plt.plot(x, wave[:, (i * 100)-1])
    plt.title('t={}'.format(i))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.ylim([-1.2, 1.2])
plt.tight_layout()
plt.savefig('1Dwaveproblempinncis1.png', transparent=True)
plt.show()