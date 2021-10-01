import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.pyplot import figure
import math


# define the initial condition
def u0(x, y):
    return x * y * (2 - x)*(3 - y)


# define velocity
vol = 6

# fetch the data set
data = np.load("../data/2d_velocity_membrane.npy")
# allocate space for sparse dataset
sparse_data = data[::10, ::10, ::10]

# the estimate the derivative we require a very small value
eps = np.sqrt(np.finfo(np.float32).eps)

# define the learning parameters
rate = 0.01  # when we find the direction of descent, how far will we go in one step
phy_rate = 0.1
steps = 5000  # how many training steps to complete
batch_size = 100  # how many training points to use each training cycle
display_step = steps / 10  # how often to display training step
points = 50  # number of training points from the ODE used

# define the layout of the network
nn_input = 3  # number of neurons for the input layer
nn_hidden_1 = 32  # number of neurons in hidden layer 1
nn_hidden_2 = 16  # number of neurons in hidden layer 2
nn_hidden_3 = 8  # number of neurons in hidden layer 3
nn_hidden_4 = 4  # number of neurons in hidden layer 4
#nn_hidden_5 = 16  # number of neurons in hidden layer 5
#nn_hidden_6 = 16  # number of neurons in hidden layer 6
#nn_hidden_7 = 8  # number of neurons in hidden layer 5
#nn_hidden_8 = 8  # number of neurons in hidden layer 6
nn_output = 1  # number of neurons in the output layer

# initialise the weights and biases for the network as random matrices of the correct size
weights = {
    'h1': tf.Variable(tf.random.normal([nn_input, nn_hidden_1])),
    'h2': tf.Variable(tf.random.normal([nn_hidden_1, nn_hidden_2])),
    'h3': tf.Variable(tf.random.normal([nn_hidden_2, nn_hidden_3])),
    'h4': tf.Variable(tf.random.normal([nn_hidden_3, nn_hidden_4])),
    #'h5': tf.Variable(tf.random.normal([nn_hidden_4, nn_hidden_5])),
    #'h6': tf.Variable(tf.random.normal([nn_hidden_5, nn_hidden_6])),
    #'h7': tf.Variable(tf.random.normal([nn_hidden_6, nn_hidden_7])),
    #'h8': tf.Variable(tf.random.normal([nn_hidden_7, nn_hidden_8])),
    'out': tf.Variable(tf.random.normal([nn_hidden_4, nn_output]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([nn_hidden_1])),
    'b2': tf.Variable(tf.random.normal([nn_hidden_2])),
    'b3': tf.Variable(tf.random.normal([nn_hidden_3])),
    'b4': tf.Variable(tf.random.normal([nn_hidden_4])),
    #'b5': tf.Variable(tf.random.normal([nn_hidden_5])),
    #'b6': tf.Variable(tf.random.normal([nn_hidden_6])),
    #'b7': tf.Variable(tf.random.normal([nn_hidden_7])),
    #'b8': tf.Variable(tf.random.normal([nn_hidden_8])),
    'out': tf.Variable(tf.random.normal([nn_output]))
}

# we wish to use stochastic gradient descent with the defined learning rate
optimiser = tf.optimizers.SGD(rate)


# create the network
def nn(x, y, t):
    t = np.array([[[x, y, t]]], dtype='float32')
    # Hidden fully connected layer with 64 neurons
    layer_1 = tf.add(tf.matmul(t, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden fully connected layer with 64 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    # Hidden fully connected layer with 64 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
    # Hidden fully connected layer with 64 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.tanh(layer_4)
    # Hidden fully connected layer with 64 neurons
    #layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    #layer_5 = tf.nn.tanh(layer_5)
    # Hidden fully connected layer with 64 neurons
    #layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    #layer_6 = tf.nn.tanh(layer_6)
    # Hidden fully connected layer with 64 neurons
    #layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    #layer_7 = tf.nn.tanh(layer_7)
    # Hidden fully connected layer with 64 neurons
    #layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    #layer_8 = tf.nn.tanh(layer_8)
    # Output fully connected layer
    output = tf.matmul(layer_4, weights['out']) + biases['out']
    return output


# define the universal approximator using the network and the initial condition
def g(x, y, t):
    return u0(x, y) + t * u0(x, y) * nn(x, y, t)


# define the physics loss
def g_phys(x, y, t, c):
    g_2 = 2 * g(x, y, t)
    g_tt = (g(x, y, t+eps) - g_2 + g(x, y, t-eps)) / eps ** 2
    g_xx = (g(x+eps, y, t) - g_2 + g(x-eps, y, t)) / eps ** 2
    g_yy = (g(x, y + eps, t) - g_2 + g(x, y - eps, t)) / eps ** 2
    return g_tt - c**2 * (g_xx + g_yy)


# create the loss function, which compares the derivative of g against u
def loss_function():
    # arrays to hold loss
    data_summation = []
    phy_summation = []
    # intervals in time and space
    x_int = np.linspace(0, 2, 10)
    y_int = np.linspace(0, 3, 15)
    time = np.linspace(0, 2, 20)
    for i in range(0, x_int.__len__()):
        for j in range(0, y_int.__len__()):
            for k in range(0, time.__len__()):
                NN = g(x_int[i], y_int[j], time[k])
                data_summation.append(abs(NN - sparse_data[i, j, k]) ** 2 +
                                      phy_rate * (g_phys(x_int[i], y_int[j], time[k], vol)) ** 2)

    ans = tf.reduce_mean(tf.abs(data_summation))
    return ans


def train_step():
    with tf.GradientTape() as tape:
        loss = loss_function()
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimiser.apply_gradients(zip(gradients, trainable_variables))
    return loss


# train the network
for i in range(steps):
    train_step()
    print(str(i) + " steps: Average loss: %f " % (loss_function()))

a_file = open("weights.pkl", "wb")
b_file = open("biases.pkl", "wb")
pickle.dump(weights, a_file)
pickle.dump(biases, b_file)
a_file.close()
b_file.close()
