import numpy as np
from numpy.lib import math
from activation import no_activation

class Value:
    def __init__(self, value, children=()):
        self.value = value
        self.grad = 0
        self._backward = lambda: 1
        self.children = children
        self.prev = set(self.children)


    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self + Value(other)

        out = Value(self.value + other.value, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self + Value(other)

        out = Value(self.value + other.value, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self - Value(other)

        out = Value(self.value - other.value, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self * Value(other)

        out = Value(self.value * other.value, (self, other))
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * Value(other)

        out = Value(self.value * other.value, (self, other))
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward

        return out

    
    def __truediv__(self, other):
        return self * other ** -1

    
    def __rtruediv__(self, other):
        return other * self ** -1


    def __neg__(self):
        return self * Value(-1)


    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.value ** other, (self,))
        
        def _backward():
            self.grad += (other * self.value ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __rpow__(self, other):
        out = Value(other ** self.value, (self,))

        def _backward():
            self.grad += ((other ** self.value) * math.log(other)) * out.grad
        out._backward = _backward

        return out

    def __repr__(self):
        return '{' + f'{self.value} {self.grad}' + '}'

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
        out = Value((math.e ** self.value - math.e ** -self.value) / (math.e ** self.value + math.e ** -self.value), (self,))
        
        def _backward():
            self.grad = 1 - out.value ** 2
        out._backward = _backward
        
        return out

    def sigmoid(self):
        out = Value(1 / (1 + math.e ** self.value), (self,))

        def _backward():
            self.grad = out.value * (1 - out.value)
        out._backward = _backward

        return out


class NeuralNet:
    def __init__(self, input, output_layer, hidden=[], add_biases=False, activation=no_activation):
        self.input = input
        self.output_layer = output_layer
        self.hidden = hidden
        self.add_biases = add_biases
        self.activation = activation
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
                # Input -> Hidden
                self.weights = [self.random(self.input, self.hidden[0])]
                # Hidden -> Hidden
                for h in range(1, len(self.hidden)):
                    self.weights.append(self.random(
                        self.hidden[h-1], self.hidden[h]
                    ))
                # Hidden -> Output
                self.weights.append(self.random(
                    self.hidden[-1], self.output_layer
                ))
        if self.add_biases:
            self.biases = self.random(len(self.weights))
        else:
            self.biases = [0] * len(self.weights)

    def dotproduct(self, layer, weights): # not in use anymore but still keeping it just in case
        product = []
        for i in weights:
            product.append(np.dot(layer, i))
        return product

    def forward(self, x):
        last_layer = [Value(i) for i in x]
        for (weight, bias) in zip(self.weights, self.biases):
            last_layer = self.activation(np.dot(weight, last_layer) + bias)
        return last_layer

    def printWeights(self):
        if not hasattr(self, 'weights'):
            print("set weights first")
            print("aborting...")
            exit()

        for x in self.weights:
            print(x)

    def printBiases(self):
        for x in self.biases:
            print(x)


def debug(x):
    print(f"[debug] {x}")
