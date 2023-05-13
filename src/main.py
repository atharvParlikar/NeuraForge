from NeuraForge import NeuralNet
from activation import tanh, sigmoid
from loss import MSEloss
import numpy as np
import timeit

# start = timeit.default_timer()

# nn = NeuralNet(28*28, 10, [64, 64], True, tanh)
# nn.setWeights()
# y = nn.forward(np.random.rand(28*28))

# print(f'[forward pass] {timeit.default_timer() - start} seconds')

# start = timeit.default_timer()
# loss = MSEloss(y, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

# print(f'[backward pass] {timeit.default_timer() - start} seconds')



