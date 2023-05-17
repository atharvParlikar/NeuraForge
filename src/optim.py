from typing import List

def gradient_decent(params, lr):
    # for weights
    for x in range(len(params)):
        for i in range(len(params[x])):
            # because the -= operation creates a new Value object and thus resetting the gradint to 0,
            # we need to preserve the gradient of the orignal value and then inject it after the operation is done
            gradient = params[x][i].grad
            params[x][i] -= params[x][i].grad * lr
            params[x][i].grad = gradient

