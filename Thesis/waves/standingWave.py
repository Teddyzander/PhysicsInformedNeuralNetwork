import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import math
from scipy.integrate import quad

# define some conditions
amp = 2  # amplitude of the wave
vel = 20  # wave velocity


def u_sol(x, t):
    return amp * (math.sin(x - vel * t) + math.sin(x + vel * t))


def wave(x, t):
    points = x.__len__()
    wave = np.ones(points)
    for i in range(0, points):
        wave[i] = u_sol(x[i], t)

    return wave


t = 0
for i in range(0, 50):
    x = np.linspace(0, 2 * math.pi, 100)
    u = wave(x, t)
    plt.plot(x, u, label="initial wave")
    plt.axis([0, 2 * math.pi, -5, 5])
    plt.show()
    t = t + 0.01
