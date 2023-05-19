from NeuraForge import NeuralNet, Value
from loss import MSEloss
import random
import matplotlib.pyplot as plt
from optim import gradient_decent
from utils import depth

net = NeuralNet(1, 1)
net.setWeights()

data = [(x, x*2) for x in range(500,1000)]
x, y_true = [x[0] for x in data], [x[1] for x in data]

losses = []

print(net.weights)

for x_, yt in zip(x, y_true):
    y = net.forward([x_])
    loss = MSEloss([yt], [y])
    loss.backward()
    gradient_decent([net.weights], 0.000001)
    print(f"loss := {loss} ; y := {y}")
    net.reset_grad()
    losses.append(loss)
    y.grad = 0

plt.plot([x.value for x in losses])
plt.savefig('loss.png')

print("==== Examples ====")
for i in range(10):
    random_number = random.randint(100, 200)
    print(f"x := {random_number} ; actual := {random_number * 2} ; predicted := {net.forward([random_number]).value}")
