import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, safe_mask


class MultipleTestingTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, selected_regressors):
        self.selected_regressors = selected_regressors

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None, threshold=None):
        return self

    # Return the statistically different regressors
    def get_support(self, X, indices=False):
        mask = self.selected_regressors[X.shape[1]]

        return mask if not indices else np.where(mask)[0]

    # Restrict features to the stable ones
    def transform(self, X):
        X = check_array(X, accept_sparse="csr")

        mask = self.get_support(X)

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting")
        if not mask.any():
            print("No features were selected")
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
