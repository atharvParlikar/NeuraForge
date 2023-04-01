from NeuraForge import Value
import numpy as np

e = np.e

a = Value(1)
b = 1 / (1 + e ** -a)
b.backward()
print(a)
