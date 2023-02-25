from NeuraForge import NeuralNet, Value
from loss import MSEloss
from activation import tanh


nn = NeuralNet(2, 1, [3], True, tanh)
nn.setWeights()
nn.printWeights()
print("===" * 15)
out = nn.forward([1, 2])
loss = MSEloss(out, [Value(0)])
loss.grad = 1
loss.backward()
nn.printWeights()
