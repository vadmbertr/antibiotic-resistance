from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array, check_X_y, safe_mask
from sklearn.utils.validation import check_is_fitted


# Inspired by https://github.com/scikit-learn-contrib/stability-selection


# Return selected variables along all the regularization path
def _get_support_path(estimator, reg_param_name, reg_grid, X, y, boot_idx):
    X_boot = X[safe_mask(X, boot_idx), :]
    y_boot = y[boot_idx]

    selected_variables = [None] * len(reg_grid)
    for idx, reg_value in enumerate(reg_grid):
        selected_variables[idx] = _get_support(clone(estimator), reg_param_name, reg_value, X_boot, y_boot)

    return selected_variables


# Return selected variables for a given regularization value
def _get_support(estimator, reg_param_name, reg_value, X, y):
    estimator.set_params(**{reg_param_name: reg_value})
    estimator.fit(X, y)

    variable_selector = SelectFromModel(estimator=estimator, threshold=1e-8, prefit=True)
    return variable_selector.get_support()


class StabilitySelection(TransformerMixin, BaseEstimator):
    def __init__(self, estimator=LogisticRegression(penalty="l1", class_weight="balanced", solver="liblinear"),
                 reg_param_name="C", reg_grid=list(np.logspace(-2, 4, 25)),
                 n_bootstraps=100, bootstrap_prop=.5, threshold=.6, random_state=15, n_jobs=1,
                 stability_scores=None):
        self.estimator = estimator
        self.reg_param_name = reg_param_name
        self.reg_grid = reg_grid
        self.n_bootstraps = n_bootstraps
        self.bootstrap_prop = bootstrap_prop
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.stability_scores = stability_scores

        np.random.seed(self.random_state)
        self.estimator.set_params(**{"random_state": self.random_state})

    def __sklearn_is_fitted__(self):
        return self.stability_scores is not None

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")

        if not self.__sklearn_is_fitted__():
            # Initial fit: build the stability scores
            stability_scores = Parallel(n_jobs=self.n_jobs)(delayed(_get_support_path)(
                self.estimator, self.reg_param_name, self.reg_grid,
                X, y, np.random.choice(X.shape[0], size=int(X.shape[0] * self.bootstrap_prop))
            ) for _ in range(self.n_bootstraps))

            self.stability_scores = np.array(stability_scores).mean(axis=0).transpose()

        return self

    # Return the stable features for a given stability threshold
    def get_support(self, indices=False, threshold=None):
        if threshold is not None and (not isinstance(threshold, float) or not (0.0 < threshold <= 1.0)):
            raise ValueError("threshold should be a float in (0, 1], got %s" % self.threshold)

        cutoff = self.threshold if threshold is None else threshold
        mask = (self.stability_scores.max(axis=1) > cutoff)

        return mask if not indices else np.where(mask)[0]

    # Restrict features to the stable ones
    def transform(self, X, threshold=None):
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        mask = self.get_support(threshold=threshold)

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting")
        if not mask.any():
            print("No features were selected")
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]

    def fit_transform(self, X, y=None, threshold=None):
        self.fit(X, y)
        return self.transform(X, threshold=threshold)

    # Return the stability path figure
    def plot_path(self, threshold_highlight=None, xscale="log", **kwargs):
        check_is_fitted(self)

        threshold = self.threshold if threshold_highlight is None else threshold_highlight
        paths_to_highlight = self.get_support(threshold=threshold)

        fig, ax = plt.subplots(1, 1, **kwargs)
        if not paths_to_highlight.all():
            ax.plot(self.reg_grid, self.stability_scores[~paths_to_highlight].T, 'k:', linewidth=0.5)

        if paths_to_highlight.any():
            ax.plot(self.reg_grid, self.stability_scores[paths_to_highlight].T, 'r-', linewidth=0.5)

        if threshold is not None:
            ax.plot(self.reg_grid, threshold * np.ones_like(self.reg_grid), 'b--', linewidth=0.5)

        ax.set_ylabel("Stability score")
        ax.set_xlabel("Regularization")
        ax.set_xscale(xscale)

        fig.tight_layout()

        return fig, ax


class StabilitySelectionTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, stability_scores, threshold=.6):
        self.stability_scores = stability_scores
        self.threshold = threshold

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None, threshold=None):
        return self

    # Return the stable features for a given stability threshold
    def get_support(self, X, indices=False, threshold=None):
        if threshold is not None and (not isinstance(threshold, float) or not (0.0 < threshold <= 1.0)):
            raise ValueError("threshold should be a float in (0, 1], got %s" % self.threshold)

        cutoff = self.threshold if threshold is None else threshold
        mask = (self.stability_scores[X.shape[1]].max(axis=1) > cutoff)

        return mask if not indices else np.where(mask)[0]

    # Restrict features to the stable ones
    def transform(self, X, threshold=None):
        X = check_array(X, accept_sparse="csr")

        mask = self.get_support(X, threshold=threshold)

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting")
        if not mask.any():
            print("No features were selected")
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]

    def fit_transform(self, X, y=None, threshold=None):
        return self.transform(X, threshold=threshold)
