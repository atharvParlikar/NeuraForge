from NeuraForge import NeuralNet
from activation import tanh, sigmoid
from loss import MSEloss

nn = NeuralNet(2, 1, [3], True, tanh)
nn.setWeights()
y = nn.forward([1,2])
loss = MSEloss(y, [1])

loss.backward()

nn.printWeights()
