import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import math
from scipy.integrate import quad

# plot's a solution to the wave equation using BCs and ICs with Dirichlet BCs

# define the conditions
L = 1
BC = 0
c = 1


def IC_u(x):
    return math.sin(x)


def IC_u_t(x):
    return 2*x


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
    for i in range(1, n):
        alpha_val = alpha(i) * math.cos((i * c * math.pi * t) / L)
        beta_val = beta(i) * math.sin((i * c * math.pi * t) / L)
        total_val = (alpha_val + beta_val) * math.sin((i * math.pi * x) / L)
        wave = wave + total_val

    return wave


points = 200
wave = np.linspace(-L, L, points)

for i in range(0, wave.__len__()):
    wave[i] = u_series(i / points, 0.5, 200)
    if i % 10 == 0:
        print(i)

plt.plot(np.linspace(-L, L, points), wave, label="initial wave")
plt.show()
