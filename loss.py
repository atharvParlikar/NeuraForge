def MSEloss(x, y):
    return (1 / len(x)) * sum([(x_ - y_) ** 2 for (x_, y_) in zip(x, y)])
