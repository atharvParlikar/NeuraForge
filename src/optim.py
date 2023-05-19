from utils import depth

def gradient_decent(params, lr):
    # The first loop loops over the params, normally, we would have params as
    # a list of changable parameters such as [net.weights, net.biases]

    # Now the loop inside that should loop through the nested lists and
    # apply the gradient decent on each Value.
    # The current implementation is very rigid and custom to our structure of
    # weights and biases anything else will break the code.

    # TODO: Make the function so that it can apply gradient decent on list / params
    #       of any depth or configuration i.e make it more dynamic.
    
    for i in range(len(params[0])):
        for j in range(len(params[0][i])):
            if depth(params[0]) == 3:
                for k in range(len(params[0][i][j])):
                    gradient = params[0][i][j][k].grad
                    params[0][i][j][k] -= params[0][i][j][k].grad * lr
                    params[0][i][j][k].grad = gradient
            else:
                print('depth 2')
                gradient = params[0][i][j].grad
                params[0][i][j] -= params[0][i][j].grad * lr
                params[0][i][j].grad = gradient
    
    if len(params) ==  2:
        for i in params[1]:
            gradient = params[1][i].grad
            params[1][i] -= gradient * lr
            params[1][i].grad = gradient
