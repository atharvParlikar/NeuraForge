from NeuraForge import NeuralNet, Value
from loss import MSEloss
from activation import tanh

nn = NeuralNet(2, 1, [3], True, tanh)
nn.setWeights()
nn.printWeights()
print("=" * 50)
out = nn.forward([1, 2])
loss = MSEloss(out, [Value(1)])
loss.backward()
print(f"Loss: {loss}")
nn.printWeights()

