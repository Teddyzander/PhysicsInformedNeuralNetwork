import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import math
from scipy.integrate import quad

# plot's a solution to the wave equation using BCs and ICs with Dirichlet BCs

# define the conditions
L = 1
BC = 0


def IC_u(x):
    return math.sin(4*x)


def IC_u_t(x):
    return 1


# define coefficients for series solution

def alpha_int(x, n):
    return IC_u(x) * math.sin(n * math.pi * x)


def beta_int(x, n):
    return IC_u_t(x) * math.sin(n * math.pi * x)


def alpha(n):
    return 2 * quad(alpha_int, 0, L, args=n)[0]


def beta(n):
    return 2 / (n * math.pi) * quad(beta_int, 0, L, args=n)[0]


# define the solution for the wave up to some some finite series

def u_series(x, t, n):
    wave = 0
    for i in range(1, n):
        alpha_val = alpha(i) * math.cos(i * math.pi * t)
        beta_val = beta(i) * math.sin(i * math.pi * t)
        total_val = (alpha_val + beta_val) * math.sin(i * math.pi * x)
        wave = wave + total_val

    return wave
