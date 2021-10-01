import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

x_1 = np.random.rand(100) / 100
y_1 = np.random.rand(100) / 80
x_2 = (np.random.rand(100) + 1) / 400
y_2= (np.random.rand(100)+4) / 220

x_3 = (np.random.rand(100) +1.3) / 50
y_3= (np.random.rand(100)+1.1) / 150

x = np.concatenate((x_1, x_2, x_3))
y = np.concatenate((y_1, y_2, y_3))

fig = plt.figure()

plt.plot(x, y, '.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()