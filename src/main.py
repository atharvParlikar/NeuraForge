from NeuraForge import NeuralNet, debug
from activation import tanh, sigmoid
from loss import MSEloss
import numpy as np
import timeit

start = timeit.default_timer()

nn = NeuralNet(28*28, 10, [64, 64], True, tanh)
nn.setWeights()
y = nn.forward(np.random.rand(28*28))

print(f'[forward pass] {timeit.default_timer() - start} seconds')

start = timeit.default_timer()

input_ = np.random.rand(28*28)
weights1 = np.random.rand(28*28, 64)
hidden1 = np.dot(input_, weights1)
weights2 = np.random.rand(64, 64)
hidden2 = np.dot(hidden1, weights2)
weights3 = np.random.rand(64, 10)
output = np.dot(hidden2, weights3)

print(f'[matrix dotproduct] {timeit.default_timer() - start} seconds')
