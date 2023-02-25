from numpy.lib import math


def sigmoid(x):
    return 1 / (1 + ((-x).exp()))

def tanh(x):
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
