import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.pyplot import figure
import math
from matplotlib import cm


# define the initial condition
def u0(x):
    return np.sin(2 * x)


# define velocity
vol = 2

# fetch the data set
data = np.load("../data/1d_dalambert_wave.npy").T
sparse_data = data[::5, ::5]

# define the learning parameters
rate = 0.01  # when we find the direction of descent, how far will we go in one step
phy_rate = 0.1
steps = 500  # how many training steps to complete
batch_size = 100  # how many training points to use each training cycle
display_step = steps / 10  # how often to display training step
points = 50  # number of training points from the ODE used

# define the layout of the network
nn_input = 2  # number of neurons for the input layer
nn_hidden_1 = 20  # number of neurons in hidden layer 1
nn_hidden_2 = 20  # number of neurons in hidden layer 2
nn_hidden_3 = 20  # number of neurons in hidden layer 3
nn_hidden_4 = 20  # number of neurons in hidden layer 4
nn_hidden_5 = 20  # number of neurons in hidden layer 5
nn_hidden_6 = 20  # number of neurons in hidden layer 6
#nn_hidden_7 = 4  # number of neurons in hidden layer 5
#nn_hidden_8 = 4  # number of neurons in hidden layer 6
nn_output = 1  # number of neurons in the output layer

# initialise the weights and biases for the network as random matrices of the correct size
weights = {
    'h1': tf.Variable(tf.random.normal([nn_input, nn_hidden_1])),
    'h2': tf.Variable(tf.random.normal([nn_hidden_1, nn_hidden_2])),
    'h3': tf.Variable(tf.random.normal([nn_hidden_2, nn_hidden_3])),
    'h4': tf.Variable(tf.random.normal([nn_hidden_3, nn_hidden_4])),
    'h5': tf.Variable(tf.random.normal([nn_hidden_4, nn_hidden_5])),
    'h6': tf.Variable(tf.random.normal([nn_hidden_5, nn_hidden_6])),
    #'h7': tf.Variable(tf.random.normal([nn_hidden_6, nn_hidden_7])),
    #'h8': tf.Variable(tf.random.normal([nn_hidden_7, nn_hidden_8])),
    'out': tf.Variable(tf.random.normal([nn_hidden_6, nn_output]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([nn_hidden_1])),
    'b2': tf.Variable(tf.random.normal([nn_hidden_2])),
    'b3': tf.Variable(tf.random.normal([nn_hidden_3])),
    'b4': tf.Variable(tf.random.normal([nn_hidden_4])),
    'b5': tf.Variable(tf.random.normal([nn_hidden_5])),
    'b6': tf.Variable(tf.random.normal([nn_hidden_6])),
    #'b7': tf.Variable(tf.random.normal([nn_hidden_7])),
    #'b8': tf.Variable(tf.random.normal([nn_hidden_8])),
    'out': tf.Variable(tf.random.normal([nn_output]))
}

# we wish to use stochastic gradient descent with the defined learning rate
optimiser = tf.optimizers.SGD(rate)


# create the network
def nn(x, t):
    t = np.array([[[x, t]]], dtype='float32')
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
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.tanh(layer_5)
    # Hidden fully connected layer with 64 neurons
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.tanh(layer_6)
    # Hidden fully connected layer with 64 neurons
    #layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    #layer_7 = tf.nn.tanh(layer_7)
    # Hidden fully connected layer with 64 neurons
    #layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    #layer_8 = tf.nn.tanh(layer_8)
    # Output fully connected layer
    output = tf.matmul(layer_6, weights['out']) + biases['out']
    return output


# define the universal approximator using the network and the initial condition
def g(x, t):
    return u0(x) + t * u0(x) * nn(x, t)


# define the physics loss
def g_phys(data):
    return 0


# create the loss function, which compares the derivative of g against u
def loss_function(training_cycle):
    # arrays to hold loss
    data_summation = []
    phys_summation = []
    # intervals in time and space
    time_int = np.linspace(0, 2, 40)
    x_int = np.linspace(0, math.pi, 10)
    results = np.ones((x_int.__len__(), time_int.__len__()))
    for i in range(0, x_int.__len__()):
        for j in range(0, time_int.__len__()):
            NN = g(x_int[i], time_int[j])
            results[i, j] = NN
            data_loss = abs(NN - data[i, j]) ** 2
            data_summation.append(abs(data_loss))

    time_int = np.linspace(0, 2, 100)
    x_int = np.linspace(0, math.pi, 100)
    results = np.ones((x_int.__len__(), time_int.__len__()))
    for i in range(0, x_int.__len__()):
        for j in range(0, time_int.__len__()):
            NN = g(x_int[i], time_int[j])
            results[i, j] = NN
    # if training_cycle > steps / 2:
    grad1 = np.gradient(results)
    dx = grad1[0]
    dt = grad1[1]
    dxx = np.gradient(dx)[0]
    dtt = np.gradient(dt)[1]
    phys_loss = phy_rate * (np.abs(dtt - vol ** 2 * dxx)) ** 2
    for i in phys_loss:
        for j in i:
            phys_summation.append(abs(tf.constant(np.array([[[np.float32(j)]]]))))

    ans = data_summation + phys_summation
    ans = tf.abs(ans)
    ans = tf.reduce_mean(ans)
    return ans


def train_step(training_cycle):
    with tf.GradientTape() as tape:
        loss = loss_function(training_cycle)
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimiser.apply_gradients(zip(gradients, trainable_variables))
    return loss


# train the network
for i in range(steps):
    train_step(i)
    if i % display_step == 0:
        print("Average loss: %f " % (loss_function(i)))

a_file = open("weights_d.pkl", "wb")
b_file = open("biases_d.pkl", "wb")
pickle.dump(weights, a_file)
pickle.dump(biases, b_file)
a_file.close()
b_file.close()

t_int = np.linspace(0, 2, 200)
x_int = np.linspace(0, math.pi, 200)
result = np.ones((x_int.__len__(), t_int.__len__()))

for i in range(0, x_int.__len__()):
    for j in range(0, t_int.__len__()):
        prediction = g(x_int[i], t_int[j]).numpy()[0][0][0]
        result[i, j] = prediction

X, T = np.meshgrid(x_int, t_int)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, T, result, linewidth=0, antialiased=False,
                       shade=True, cmap=cm.jet)

ax.view_init(0, 0)
plt.xlabel("x")
plt.ylabel("time")
plt.title("D'Alembert's Solution for Simple Wave from neural network")
fig.colorbar(surf, shrink=0.75)

plt.show()

