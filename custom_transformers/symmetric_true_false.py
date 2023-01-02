from sklearn.preprocessing import FunctionTransformer


# map 0/1 to -1/1
def symmetric_true_false(X):
    X[X == 0] = -1
    return X


SymmetricTrueFalse = FunctionTransformer(symmetric_true_false, validate=True, check_inverse=False)

