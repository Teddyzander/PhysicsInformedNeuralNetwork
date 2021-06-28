import torch
import torch.nn as nn
import math

test = torch.arange(0, 1, 0.01, dtype=torch.float)
X = torch.tensor(([0], [0.1], [0.2], [0.3], [0.4], [0.5],
                  [0.6], [0.7], [0.8], [0.9], [1]), dtype=torch.float) # 3 X 1 tensor
xPredicted = torch.tensor(([0.3]), dtype=torch.float) # 1 X 2 tensor

u_0 = 1
eps = (1<<31)-1

def f(X):
    return torch.cos(2*math.pi*X)

y = f(X)

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        self.inputSize = 1
        self.outputSize = 1
        self.hiddenSize = 32

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 1 X 32 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 32 X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)  # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z)  # activation function
        o = torch.matmul(self.z2, self.W2)
        #o = self.sigmoid(self.z3)  # final activation function
        return o

    def g_forward(self, X):
        return u_0 + torch.mul(X, self.forward(X))

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = torch.abs(((self.g_forward(X+eps) - self.g_forward(X))/eps) - y)  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.g_forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))


NN = Neural_Network()
for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - (NN.g_forward(X+eps)-NN.g_forward(X)/eps))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
NN.predict()

print(NN.g_forward(torch.tensor(([0], [0.25], [1]), dtype=torch.float)))