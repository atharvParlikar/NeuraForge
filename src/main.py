from NeuraForge import NeuralNet, Value, argmax
from activation import softmax, tanh
from loss import MSEloss
import pandas as pd
from optim import gradient_decent
import random
import matplotlib.pyplot as plt

data = pd.read_csv('winequality-red.csv')

x_train, y_train = data[data.columns[:-1]][:1200] ,data[data.columns[-1]][:1200]
x_test, y_test = data[data.columns[:-1]][1200:] ,data[data.columns[-1]][1200:]

y_train = [[1 if i == x - 3 else 0 for i in range(6)] for x in y_train]
y_test = [[1 if i == x - 3 else 0 for i in range(6)] for x in y_test]

losses = []

net = NeuralNet(11, 6, [20], activation=[tanh, softmax], add_biases=True)
net.setWeights()


for i in range(0, 300):
    y = net.forward(x_train.iloc[i])
    loss: Value = MSEloss(y, y_train[i])
    loss.backward()
    gradient_decent([net.weights, net.biases], 0.01)
    net.reset_grad()
    losses.append(loss.value)
    print(f"epoch: {i + 1} ; loss := {loss}")

plt.plot(losses)
plt.savefig('loss-plt.png')
plt.close()

index = random.randint(1, 100)

print(f"\n\ny := {net.forward(x_test.iloc[index])}; actual := {y_test[index]}")
