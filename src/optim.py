def gradient_decent(params, lr):
    # for weights
    for i in range(len(params[0])):
        for j in range(len(params[0][i])):
            for k in range(len(params[0][i][j])):
                params[0][i][j][k] -= params[0][i][j][k] * lr
    for i in range(len(params[1])):
        params[1][i] -= params[1][i] * lr
