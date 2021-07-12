import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import math


# define the parameters and functions of the ODE
def u(t):
    return math.cos(2 * math.pi * t)


u0 = 1  # create an initial condition

# the estimate the derivative we require a very small value
eps = np.sqrt(np.finfo(np.float32).eps)

# define the learning parameters
rate = 0.001  # when we find the direction of descent, how far will we go in one step
steps = 5000  # how many training steps to complete
batch_size = 100  # how many training points to use each training cycle
display_step = steps / 10  # how often to display training step
points = 50  # number of training points from the ODE used

# define the layout of the network
nn_input = 1  # number of neurons for the input layer
nn_hidden_1 = 32  # number of neurons in hidden layer 1
nn_hidden_2 = 32  # number of neurons in hidden layer 2
nn_output = 1  # number of neurons in the output layer

# initialise the weights and biases for the network as random matrices of the correct size
weights = {
    'h1': tf.Variable(tf.random.normal([nn_input, nn_hidden_1])),
    'h2': tf.Variable(tf.random.normal([nn_hidden_1, nn_hidden_2])),
    'out': tf.Variable(tf.random.normal([nn_hidden_2, nn_output]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([nn_hidden_1])),
    'b2': tf.Variable(tf.random.normal([nn_hidden_2])),
    'out': tf.Variable(tf.random.normal([nn_output]))
}

# we wish to use stochastic gradient descent with the defined learning rate
optimiser = tf.optimizers.SGD(rate)


# create the network
def nn(t):
    t = np.array([[[t]]], dtype='float32')
    # Hidden fully connected layer with 32 neurons
    layer_1 = tf.add(tf.matmul(t, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden fully connected layer with 32 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    # Output fully connected layer
    output = tf.matmul(layer_2, weights['out']) + biases['out']
    return output


# define the universal approximator using the network and the initial condition
def g(t):
    return u0 + t * nn(t)


# create the loss function, which compares the derivative of g against u
def loss_function():
    summation = []
    for x in np.linspace(0, 1, points):
        dNN = (g(x + eps) - g(x)) / eps
        summation.append(abs(dNN - u(x)) ** 2)

    ans = tf.reduce_mean(tf.abs(summation))
    return ans


def train_step():
    with tf.GradientTape() as tape:
        loss = loss_function()
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimiser.apply_gradients(zip(gradients, trainable_variables))


# train the network
for i in range(steps):
    train_step()
    if i % display_step == 0:
        print("Average loss: %f " % (loss_function()))

# display the figure
figure(figsize=(10, 10))


# True Solution
def true_solution(x):
    return 1 + np.sin(2 * np.pi * x)/(2 * np.pi)


X = np.linspace(0, 1, points * 10)
result = []
S = true_solution(X)
for i in X:
  result.append(g(i).numpy()[0][0][0])

plt.plot(X, S, label="Original Function u(t)")
plt.plot(X, result, label="Neural Net Approximation g(t)")
plt.legend(loc=1, prop={'size': 15})
plt.xlabel("input t")
plt.ylabel("output")
plt.show()
