import numpy as np
from numpy.lib import math

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
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self + Value(other)

        out = Value(self.value + other.value, (self, other))
        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self - Value(other)

        out = Value(self.value - other.value, (self, other))

        def _backward():
            self.grad = out.grad
            other.grad = -out.grad
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
        if isinstance(other, (int, float)):
            return self / Value(other)

        out = Value(self.value / other.value, (self, other))
        def _backward():
            self.grad += 1 / other.value
            other.grad += -(self.value / other.value ** 2) 
        out._backward = _backward

        return out

    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Value(other) / self

        out = Value(other.value / self.value, (other, self))

        def _backward():
            self.grad += -(other.value / self.value ** 2)
            other.grad += 1 / self.value
        out._backward = _backward

        return out


    def __neg__(self):
        return self * Value(-1)


    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.value ** other, (self,))
        
        def _backward():
            self.grad = (other * self.value ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __rpow__(self, other):
        out = Value(other ** self.value, (self,))

        def _backward():
            self.grad += (other ** self.value) * math.log(other)
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
    def __init__(self, input, output_layer, hidden=[], add_biases=False, activation=None):
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
            if self.activation != None:
                for (weight, bias) in zip(self.weights, self.biases):
                    last_layer = self.dotproduct(last_layer, weight)
                    last_layer = [self.activation(x + bias) for x in last_layer]
            
        else:
            if self.activation != None:
                for weight in self.weights:
                    last_layer += self.dotproduct(last_layer, weight) 
                    last_layer = [self.activation(x) for x in last_layer]
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


def debug(x):
    print(f"[debug] {x}")
