import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B


def u0(tx, c=1, k=2, sd=0.5):
    """
    Initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.
        c: wave velocity.
        k: wave number.
        sd: standard deviation.

    Returns:
        u(t, x) as tf.Tensor.
    """

    t = tx[..., 0, None]
    x = tx[..., 1, None]
    u = np.ones((len(x), len(t)))
    for i in range(0, len(t)):
        for j in range(0, len(x)):
            z = k * x[j] - (c * k) * t[i]
            sqr = (0.5 * z / sd)**2
            u[i, j] = tf.sin(z) * tf.exp(-sqr)

    return u

def du0_dt(tx):
    """
    First derivative of t for the initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.

    Returns:
        du(t, x)/dt as tf.Tensor.
    """

    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt

if __name__ == '__main__':

    samples = 1000
    time_limit = 4
    length_limit = 2
    tx_ini = np.random.rand(samples, 2)
    tx_ini[..., 0] = np.linspace(0, time_limit, samples)           # t = 0 ~ +4
    tx_ini[..., 1] = np.linspace(-length_limit/2, length_limit, samples)           # x = -1 ~ +1
    u = u0(tf.constant(tx_ini))

    t_flat = np.linspace(0, time_limit, samples)
    x_flat = np.linspace(-length_limit/2, length_limit/2, samples)
    t, x = np.meshgrid(t_flat, x_flat)
    u = u.reshape(t.shape)

    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    vmin, vmax = -0.5, +0.5
    plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(vmin, vmax)
    plt.show()