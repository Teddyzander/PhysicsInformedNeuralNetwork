import numpy as np
import math

# solution for a rectangular vibrating membrane u_tt = c^2(u_xx + u_yy)
# 2x3 rectangle with c=6 and IC f(x, y)=xy(2-x)(3-y) with u=0 at the edges. Initial velocity of g(x, y)=sin(2*pi*x)

# define series solution
def u_sol(x, y, t, n, m):
    u = 0
    coef = 576 / (math.pi ** 6)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            termmn = ((1 + (-1) ** (j + 1)) * (1 + (-1) ** (i + 1))) / (j ** 3 * i ** 3)
            termx = math.sin((j * math.pi) / 2 * x) * x
            termy = math.sin((i * math.pi) / 3 * y)
            termt = math.cos(math.pi * math.sqrt(9 * j ** 2 + 4 * i ** 2) * t)

            u = u + termmn * termx * termy * termt

    return coef * u


def u_vol(x, y, t, n):
    u = 0
    coef = math.sin(2 * math.pi * x) / math.pi ** 2
    for i in range(1, n + 1):
        termn = (1 + (-1) ** (i + 1)) / (i * math.sqrt(36 + i ** 2))
        termy = math.sin((n * math.pi) / 3 * y)
        termt = math.sin(2 * math.pi * math.sqrt(36 + i ** 2) * t)

        u = u + termn * termy * termt

    return coef * u


# define the space and preallocate space for series solution
x_int = np.linspace(0, 2, 100)
y_int = np.linspace(0, 3, 150)
time = np.linspace(0, 2, 200)
wave_0 = np.ones((x_int.__len__(), y_int.__len__(), time.__len__()))
wave_vol = np.ones((x_int.__len__(), y_int.__len__(), time.__len__()))
n_it = 40
m_it = 40

for i in range(0, x_int.__len__()):
    for j in range(0, y_int.__len__()):
        for k in range(0, time.__len__()):
            wave_0[i][j][k] = u_sol(x_int[i], y_int[j], time[k], n_it, m_it)
            wave_vol[i][j][k] = u_vol(x_int[i], y_int[j], time[k], n_it)

    print(i)

# some plotting stuff
# plt.imshow(wave.T, extent=(0, 2, 0, 3), origin='lower', interpolation='nearest')
# plt.show()

# plt.contourf(x_int, y_int, wave[:, :, 0].T)
# plt.show()

#X, Y = np.meshgrid(x_int, y_int)
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# # surf = ax.plot_surface(X, Y, wave_0.T, linewidth=0, antialiased=False, cmap=cm.plasma)
# # plt.show()

# save the output
np.save("../data/2d_homogeneous_membrane.npy", wave_0)
np.save("../data/2d_velocity_membrane.npy", wave_0 + wave_vol)
