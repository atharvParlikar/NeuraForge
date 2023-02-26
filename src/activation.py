from math import e

def sigmoid(x):
    return 1 / (1 + e ** -x)

def tanh(x):
    return (e ** x - e ** -x) / (e ** x + e ** -x)
