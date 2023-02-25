import numpy as np
from activation import tanh

class Value:
    def __init__(self, value, children=()):
        self.value = value
        self.grad = 0
        self._backward = lambda: 1
        self.children = children
        self.prev = set(self.children)

    def __add__(self, other):
        out = Value(self.value + other.value, (self, other))

        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Value(self.value - other.value, (self, other))

        def _backward():
            self.grad = out.grad
            other.grad = -out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.value * other.value, (self, other))

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = Value(self.value / other.value, (self, other))

        def _backward():
            self.grad += out.grad / other.value
            other.grad += out.grad * -(self.value / other.value ** 2)
        out._backward = _backward

        return out

    def __neg__(self):
        return self * Value(-1)

    def exp(self):
        out = Value(np.e ** self.value, (self,))

        def _backward():
            self.grad += out.value
        out._backward = _backward
        
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

    def tanh(self):
        def _tanh(x):
            return (np.e ** 2*x - 1) / (np.e ** 2*x + 1)

        out = Value(_tanh(self.value), (self))

        def _backward():
            self.grad += 1 - _tanh(self.value) ** 2
        self._backward = _backward
        
        return out

    def __repr__(self):
        return '{' + f'{self.value} {self.grad}' + '}'


class NeuralNet:
    def __init__(self, input, output_layer, hidden=[], add_biases=False):
        self.input = input
        self.output_layer = output_layer
        self.hidden = hidden
        self.add_biases = add_biases
        self.out = Value(0)

    def random(self, x, y=0):
        mat = []
        if y == 0:
            return [Value(np.random.rand()) for _ in range(x)]
        for _ in range(y):
            mat.append([Value(np.random.rand()) for _ in range(x)])
        return mat

    def setWeights(self):
        if self.input != 0 and self.output_layer != 0:
            if self.hidden == []:
                self.weights = self.random(self.input, self.output_layer)
            elif len(self.hidden) == 1:
                self.weights = [self.random(self.input, self.hidden[0]), self.random(
                    self.hidden[0], self.output_layer
                )]
            else:
                self.weights = [
                    self.random(self.input, self.hidden[0])]
                for h in range(1, len(self.hidden)):
                    self.weights.append(self.random(
                        self.hidden[h-1], self.hidden[h]
                    ))
                self.weights.append(self.random(
                    self.hidden[-1], self.output_layer
                ))
        if self.add_biases:
            self.biases = self.random(len(self.weights))

    def dotproduct(self, layer, weights):
        product = []
        for i in weights:
            product.append(np.dot(layer, i))
        return product

    def forward(self, x):
        last_layer = [Value(i) for i in x]
        if self.add_biases:
            for (weight, bias) in zip(self.weights, self.biases):
                last_layer = self.dotproduct(last_layer, weight)
                last_layer = [x + bias for x in last_layer]
            
        else:
            for weight in self.weights:
                last_layer = self.dotproduct(last_layer, weight)
        return last_layer 

    def printWeights(self):
        for x in self.weights:
            print(x)

    def printBiases(self):
        for x in self.biases:
            print(x)


nn = NeuralNet(2, 1, [3], True)
nn.setWeights()
out = nn.forward([2, 2])[0]
out.grad = 1

x = Value(1)
print(tanh(x))
