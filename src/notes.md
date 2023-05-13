# Forwardpass speed

I have observed that the forward pass function is pretty slow for large network,
currently I have tested with a network of size [28*28, 64, 64, 10] (input, hidden, output)
the time for forward pass currently averages around 1 second.

The forward pass of this size only consists of 3 matrix multiplications,

1. [1, 784] * [784, 64] 
2. [64, 64] * [64]
3. [64] * [10]

These 3 matrix dotproducts shouldn't cost 1 second, or maybe it's about all the shit
that goes in the background when I multiply 2 Value classes, to see if there's a
difference I need to test it on vanilla numpy arrays.
