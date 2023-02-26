from activation import sigmoid, tanh
from math import e
from NeuraForge import Value

a = Value(2)
b = a.tanh()
b.backward()
print(f"a := {a} ; b := {b}")
