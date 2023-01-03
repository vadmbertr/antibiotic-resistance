from sklearn.preprocessing import FunctionTransformer


def std_tf(X):
    return (X - X.mean()) / X.std()


standard_true_false = FunctionTransformer(std_tf, validate=True, check_inverse=False)
