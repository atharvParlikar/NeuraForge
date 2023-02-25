from main import NeuralNetwork
from main import Value
import numpy as np

net = NeuralNetwork(2, 1, [3], True)
net.setWeights()
out = net.forward([Value(np.random.rand()) for _ in range(2)])

print(out)
