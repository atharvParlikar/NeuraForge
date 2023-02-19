import numpy as np
from loss import MSEloss
import activation

class NeuralNetwork:
    def __init__(self, inputs=0, outputs=0, hidden=[], add_biases=False, activation=None):
        self.input = inputs
        self.output = outputs
        self.hidden = hidden
        self.add_biases = add_biases
        self.activation = activation
        self.biases = []

    # sets the weights randomly
    def setWeights(self):
        if self.input != 0 and self.output != 0:
            if self.hidden == []:
                self.weights = np.random.rand(self.output, self.input)
            elif len(self.hidden) == 1:
                self.weights = [np.random.rand(self.hidden[0], self.input), np.random.rand(
                    self.output, self.hidden[0]
                )]
            else:
                self.weights = [np.random.rand(self.hidden[0], self.input)]
                for h in range(1, len(self.hidden)):
                    self.weights.append(np.random.rand(
                        self.hidden[h], self.hidden[h - 1]
                    ))
                self.weights.append(np.random.rand(
                    self.output, self.hidden[-1]
                ))
        if self.add_biases:
            self.biases = np.random.rand(len(self.weights))

    def dotproduct(self, layer, weights):
        product = []
        for i in weights:
            product.append(np.dot(layer, i))
        return product

    def forward(self, x):
        last_layer = x
        if self.add_biases:
            if self.activation != None:
                for (weight, bias) in zip(self.weights, self.biases):
                    last_layer = self.activation(self.dotproduct(last_layer, weight) + bias)
            else:
                for (weight, bias) in zip(self.weights, self.biases):
                    last_layer = self.dotproduct(last_layer, weight) + bias

        else:
            for weight in self.weights:
                last_layer = self.dotproduct(last_layer, weight)

        return last_layer


# Example of forward a pass
nn = NeuralNetwork()
nn.input = 28*28
nn.output = 2
nn.hidden = [2, 2]
nn.add_biases = True
nn.setWeights()
input_layer = np.random.rand(28*28)
forward = nn.forward(input_layer)
nn.activation = activation.tanh
forward_with_activation = nn.forward(input_layer)
print(forward)
print(forward_with_activation)
