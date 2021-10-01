import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sym

# define wave velocity
c = 2
# define small shift for estimating derivative
eps = 1e-14


# define the initial condition
def f(x):
    return sym.sin(2 * x)


s = sym.Symbol('s')


# define wave using d'alembert's solution
def u(x, t):
    g_prime = sym.integrate(0, (s, x - c * t, x + c * t))
    return 0.5 * (f(x + c * t) + f(x - c * t)) + (1 / (2 * c)) * g_prime


x, t = sym.symbols('x t')

u_tt = sym.diff(u(x, t), t, t)
u_xx = sym.diff(u(x, t), x, x)


def check_phys(x_in, t_in):
    u_diff_t = u_tt.evalf(subs={x: x_in, t: t_in})
    u_diff_x = u_xx.evalf(subs={x: x_in, t: t_in})
    return u_diff_t - c ** 2 * u_diff_x


time_int = np.linspace(0, 2, 200)
x_int = np.linspace(0, math.pi, 200)
wave = np.ones((time_int.__len__(), x_int.__len__()))

for i in range(0, time_int.__len__()):
    print(i)
    for j in range(0, x_int.__len__()):
        wave[i, j] = u(x_int[j], time_int[i])

grad1 = np.gradient(wave)
dx = grad1[0]
dt = grad1[1]
dxx = np.gradient(dx)[0]
dtt = np.gradient(dt)[1]
phys_loss = dtt - c ** 2 * dxx

X, T = np.meshgrid(x_int, time_int)
fps = 10
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, T, wave, linewidth=0, antialiased=False,
                       shade=True, alpha=0.75, cmap=cm.jet)

ax.view_init(0, 0)
plt.xlabel("x")
plt.ylabel("time")
plt.title("D'Alembert's Solution for Simple Wave")
fig.colorbar(surf, shrink=0.75)

plt.show()

# np.save("../data/1d_dalambert_wave.npy", wave)
