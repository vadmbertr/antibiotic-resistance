from sklearn.preprocessing import FunctionTransformer


# map 0/1 to -1/1
def sym_tf(X):
    X[X == 0] = -1
    return X


symmetric_true_false = FunctionTransformer(sym_tf, validate=True, check_inverse=False)
