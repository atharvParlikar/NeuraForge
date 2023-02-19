from numpy.lib import math


def sigmoid(x):
    return 1 / (1 + math.e ** -x)

def tanh(x):
    return ((math.e ** x) - (math.e ** -x)) / ((math.e ** x + math.e ** -x))
