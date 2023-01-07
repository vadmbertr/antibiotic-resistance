from multipy.fdr import lsu
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, safe_mask
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ttest_ind
from sklearn.utils.validation import check_is_fitted


# choose appropriate test based on # of unique values
def stat_test(x, res_idx, sens_idx):
    if len(np.unique(x)) > 2:
        test_res = ttest_ind(x[res_idx], x[sens_idx])
    else:
        test_res = proportions_ztest((sum(x[res_idx]), sum(x[sens_idx])),
                                     (len(x[res_idx]), len(x[sens_idx])))
    return test_res[1]


class MultipleTesting(TransformerMixin, BaseEstimator):

    def __init__(self, alpha=.05):
        self.alpha = alpha
        self.pval = None
        self.mask = None

    def __sklearn_is_fitted__(self):
        return self.pval is not None

    # run individual tests
    def fit(self, X, y=None):
        res_idx = np.where(y == 1)
        sens_idx = np.where(y == 0)

        self.pval = np.apply_along_axis(lambda x: stat_test(x, res_idx, sens_idx), 0, X)
        self.mask = lsu(self.pval, q=self.alpha)

        return self

    # Return the mask of statistically different regressors
    def get_support(self, X, indices=False, alpha=None):
        if alpha is not None and (not isinstance(alpha, float) or not (0.0 < alpha < 1.0)):
            raise ValueError("alpha should be a float in (0, 1), got %s" % self.alpha)

        alpha = self.alpha if alpha is None else alpha
        if alpha == self.alpha:
            mask = self.mask
        else:
            mask = lsu(self.pval, q=alpha)

        return mask if not indices else np.where(mask)[0]

    # Restrict to regressors
    def transform(self, X, alpha=None):
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        mask = self.get_support(X, alpha=alpha)

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting")
        if not mask.any():
            print("No features were selected")
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]

    def fit_transform(self, X, y=None, alpha=None):
        self.fit(X, y)
        return self.transform(X, alpha=alpha)
