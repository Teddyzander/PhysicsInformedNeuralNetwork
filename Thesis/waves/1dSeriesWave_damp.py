import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# solution for a 1d vibrating membrane u_tt = c^2u_xx
# 2 units long rectangle with c=6 and IC f(x, y)=x(2-x) with u=0 at the edges.
# u_t(t=0)=0
pi = np.pi
def u_sol(x, t, n, c, b):

    u = 0
    for i in range(1, n+1):

        mu = (c * np.sqrt(4 * i ** 2 * pi ** 2 - b ** 2 * c ** 2)) / 2
        beta = (2 * (
                (10 * pi ** 4 * i ** 4 - 360 * pi ** 2 * i ** 2 + 1200) * np.sin(pi * i) +
                     (80 * pi**3 * i**3 - 960 * pi * i) * np.cos(pi * i) -
                (240 * pi * i)) / (pi**6 * i ** 6))
        alpha = 5 * b * c ** 2 * beta

        term = (np.sin(i * pi * x) *
                np.exp(-c ** 2 * b * t / 2) *
                (alpha * np.sin(mu * t) + beta * np.cos(mu * t)))

        u = u + term
    return u


x = np.linspace(0, 1, 200)
t = np.linspace(0, 4, 400)
"""
wave = np.zeros((x.__len__(), t.__len__()))
for i in range(0, x.__len__()):
    for j in range(0, t.__len__()):
        ans = u_sol(x[i], t[j], 50, 3, 0.1)
        wave[i, j] = ans

    print(i)

np.save('../data/waveformdamp_cis1_bis01.npy', np.asarray(wave))
"""
wave = np.load('../data/waveformdamp_cis1_bis01.npy')

fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 4)
plt.subplot(gs[0, :])
vmin, vmax = -3.5, +3.5
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
        plt.plot(x, wave[:, (i * 100)])
    plt.title('t={}'.format(i))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.ylim([-1.2, 1.2])
plt.tight_layout()
plt.savefig('1Dwaveproblemtruedamp_cis3_bis01.png', transparent=True)
plt.show()