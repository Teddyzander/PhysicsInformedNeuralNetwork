import numpy as np
import math

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
