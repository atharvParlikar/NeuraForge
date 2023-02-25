from NeuraForge import Value

def MSEloss(x, y):
    assert len(x) == len(y)
    sum_ = Value(0)
    for (x_, y_) in zip(x, y):
        sum_ += (x_ - y_) ** 2

    return sum_ / Value(len(x))
