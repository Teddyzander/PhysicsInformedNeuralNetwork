import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import math
from scipy.integrate import quad

# plot's a solution to the wave equation using BCs and ICs with Dirichlet BCs

# define the conditions
L = 1
BC = 0
c = 6
T = 6


def IC_u(x):
    return x * (1 - x)


def IC_u_t(x):
    return 0


# define coefficients for series solution

def alpha_int(x, n):
    return IC_u(x) * math.sin((c * n * math.pi * x) / L)


def beta_int(x, n):
    return IC_u_t(x) * math.sin((c * n * math.pi * x) / L)


def alpha(n):
    return 2 / L * quad(alpha_int, 0, L, args=n)[0]


def beta(n):
    return 2 / (n * math.pi * c) * quad(beta_int, 0, L, args=n)[0]


# define the solution for the wave up to some some finite series

def u_series(x, t, n):
    wave = 0
    """
    for i in range(1, n):
        alpha_val = alpha(i) * math.cos((i * c * math.pi * t) / L)
        beta_val = beta(i) * math.sin((i * c * math.pi * t) / L)
        total_val = (alpha_val + beta_val) * math.sin((i * math.pi * x) / L)
        wave = wave + total_val
    """

    for i in range(1, n * 2):
        if i % 2 != 0:
            wave += (8 / (i ** 3 * math.pi ** 3)) * math.sin(i * math.pi * x) * math.cos(c * i * math.pi * t)
    return wave


x_points = 100
t_points = 1200
x_flat = np.linspace(0, L, x_points)
t_flat = np.linspace(0, T, t_points)
wave = np.zeros((x_flat.__len__(), t_flat.__len__()))

for i in range(0, x_flat.__len__()):
    for j in range(0, t_flat.__len__()):
        wave[i, j] = u_series(i / x_points, j / t_points, 100)
    if i % 10 == 0:
        print(i)

t, x = np.meshgrid(t_flat, x_flat)
# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(7, 4))
gs = GridSpec(2, 4)
plt.subplot(gs[0, :])
vmin, vmax = -0.5, +0.5
plt.pcolormesh(t, x, wave, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(vmin, vmax)
t_cross_sections = [0, 1, 2, 3]
time = 0
for i, t_cs in enumerate(t_cross_sections):
    plt.subplot(gs[1, i])
    tx = np.stack([np.full(t.shape, t_cs), x], axis=-1)
    u = wave[:, time]
    plt.plot(x_flat, u)
    plt.title('t={}'.format(str(time * T / t_points)))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.ylim([-0.5, 0.5])
    time = time + round(t_points / T)
    plt.tight_layout()

plt.savefig('result_anytc_wave', transparent=True)
plt.show()
