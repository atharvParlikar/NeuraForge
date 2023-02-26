from activation import sigmoid
from math import e
from NeuraForge import Value

a = Value(2)
b = e ** -a
b.backward()
print(a, b)
