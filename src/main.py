from NeuraForge import NeuralNet
from loss import MSEloss
import numpy as np

nn = NeuralNet(3, 1, [2], False)
print(nn.activation)
nn.setWeights()

input_ = [1,2,3]

y = nn.forward(input_)

loss = MSEloss(y, [3])
print(loss)

loss.backward()
