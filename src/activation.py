from math import e

def sigmoid(x):
    return 1 / (1 + e ** -x)

def tanh(x):
    return (e ** x - e ** -x) / (e ** x + e ** -x)

def softmax(layer_x):
    sum_ = sum([e ** i for i in layer_x])
    return [e ** i / sum_ for i in layer_x]

def no_activation(x):
    return x
