import numpy as np
from loss import MSEloss
import activation


class Value:
    def __init__(self, value, children=()):
        self.value = value
        self.grad = 0
        self._backward = lambda: None
        self.children = children
        self.prev = set(self.children)

    def __add__(self, other):
        out = Value(self.value + other.value, (self, other))

        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.value * other.value, (self, other))

        def _backward():
            self.grad += out.grad * other.grad
            other.grad += out.grad * self.grad
        out._backward = _backward()
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for i in reversed(topo):
            i._backward()

    def __repr__(self):
        return f'[{self.value} {self.grad}]'


class NeuralNetwork:
    def __init__(self, inputs=0, outputs=0, hidden=[], add_biases=False, activation=None):
        self.input = inputs
        self.output = outputs
        self.hidden = hidden
        self.add_biases = add_biases
        self.activation = activation
    # sets the weights randomly

    def random(self, x, y=0):
        mat = []
        if y == 0:
            return [Value(np.random.rand()) for _ in range(x)]
        for i in range(y):
            mat.append([Value(np.random.rand()) for _ in range(x)])
        return mat

    def setWeights(self):
        if self.input != 0 and self.output != 0:
            if self.hidden == []:
                self.weights = self.random(self.output, self.input)
            elif len(self.hidden) == 1:
                self.weights = [self.random(self.hidden[0], self.input), self.random(
                    self.output, self.hidden[0]
                )]
            else:
                self.weights = [
                    self.random(self.hidden[0], self.input)]
                for h in range(1, len(self.hidden)):
                    self.weights.append(self.random(
                        self.hidden[h], self.hidden[h - 1]
                    ))
                self.weights.append(self.random(
                    self.output, self.hidden[-1]
                ))
        if self.add_biases:
            self.biases = self.random(len(self.weights))

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
                    last_layer = self.activation(
                        self.dotproduct(last_layer, weight) + bias)
            else:
                for (weight, bias) in zip(self.weights, self.biases):
                    last_layer = self.dotproduct(last_layer, weight) + bias

        else:
            for weight in self.weights:
                last_layer = self.dotproduct(last_layer, weight)

        return last_layer


# Example of forward a pass
# nn = NeuralNetwork()
# nn.input = 2
# nn.output = 1
# nn.hidden = [3]
# nn.add_biases = True
# nn.setWeights()
# input_layer = [Value(np.random.rand()) for _ in range(2)]
# forward = nn.forward(input_layer)
# nn.activation = activation.tanh
# forward_with_activation = nn.forward(input_layer)
# print(forward)
# print(forward_with_activation)
