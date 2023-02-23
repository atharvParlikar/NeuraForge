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
