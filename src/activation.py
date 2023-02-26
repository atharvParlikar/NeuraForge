from math import e

def sigmoid(x):
    return 1 / (1 + e ** -x)

def tanh(x):
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

