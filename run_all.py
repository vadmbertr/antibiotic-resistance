import argparse
from itertools import combinations, product
import os
import pandas as pd
import pickle
import shutil
import traceback

import joblib
from joblib import Memory
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from stability_selection import StabilitySelection


def read_data(data_path):
    with open(os.path.join(data_path, "dataset.pkl"), "rb") as f:
        data = pickle.load(f)
        Y = data["pheno"].iloc[:, 1:]
        X_gpa = data["X_gpa"]
        X_snps = data["X_snps"]
        X_genexp = data["X_genexp"]

    return X_gpa, X_snps, X_genexp, Y


def build_pipeline(X_gpa, X_snps, X_genexp, cache_path):
    gpa_idx = np.arange(0, X_gpa.shape[1] - 1)
    snps_idx = np.arange(0, X_snps.shape[1] - 1) + gpa_idx[-1] + 1
    genexp_idx = np.arange(0, X_genexp.shape[1] - 1) + snps_idx[-1] + 1

    trans_ind = ColumnTransformer(transformers=[("genexp", StandardScaler(), genexp_idx)],
                                  remainder="passthrough")
    dim_red_ind = ColumnTransformer(transformers=[("gpa", "passthrough", gpa_idx),
                                                  ("snps", "passthrough", snps_idx),
                                                  ("genexp", "passthrough", genexp_idx)])

    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.mkdir(cache_path)

    pipe = Pipeline([("trans_ind", trans_ind), ("dim_red_ind", dim_red_ind),
                     ("dim_red", "passthrough"),
                     ("clf", DummyClassifier())],
                    memory=Memory(location=cache_path))

    return pipe


def _create_grid(roots, params):
    def add_to_grid(g, r, p):
        if len(p[0]) > 0:
            r = "__".join([r, p[0]])
        g[r] = p[1]
        for c in p[2]:
            add_to_grid(g, r, c)

    grids = []
    for comb in combinations(product(roots, params), len(roots)):
        valid = True
        grid = {}
        for root, param in comb:
            if root in grid:
                valid = False
                break
            else:
                add_to_grid(grid, root, param)
        if valid:
            grids.append(grid)
    return grids


def _merge_grids(grids):
    merged_grid = grids.pop()
    for grid in grids:
        merged_grid = [{**g1, **g2} for g1, g2 in product(merged_grid, grid)]
    return merged_grid


def build_hp_grid(pipe, seed, n_jobs):
    dim_red_ind_grid_roots = ["dim_red_ind__gpa", "dim_red_ind__snps", "dim_red_ind__genexp"]
    dim_red_ind_grid_params = [("", ["drop", "passthrough"], []),
                               ("", [TruncatedSVD(random_state=seed)],
                                [("n_components", [64, 128, 256], [])]),
                               ("", [KernelPCA(random_state=seed)],
                                [("kernel", ["linear", "poly", "rbf", "sigmoid"], []),
                                 ("n_components", [64, 128, 256], [])]),
                               ("", [StabilitySelection(random_state=seed)],
                                [("threshold", np.linspace(.6, .9, 4), [])])]
    dim_red_ind_grid = _create_grid(dim_red_ind_grid_roots, dim_red_ind_grid_params)

    dim_red_grid_roots = ["dim_red"]
    dim_red_grid_params = [("", ["passthrough",], []),
                           ("", [TruncatedSVD(random_state=seed),],
                            [("n_components", [64, 128, 256], [])]),
                           ("", [KernelPCA(random_state=seed),],
                            [("kernel", ["linear", "poly", "rbf", "sigmoid"], []),
                             ("n_components", [64, 128, 256], [])]),
                           ("", [StabilitySelection(random_state=seed),],
                            [("threshold", np.linspace(.6, .9, 4), [])])]
    dim_red_grid = _create_grid(dim_red_grid_roots, dim_red_grid_params)

    clf_grid_roots = ["clf"]
    clf_grid_params = [("", [AdaBoostClassifier(random_state=seed), GradientBoostingClassifier(random_state=seed)],
                        [("learning_rate", np.logspace(-2, 0, 3), [])]),
                       ("", [BaggingClassifier(random_state=seed)],
                        [("max_features", np.linspace(1/3, 2/3, 3), [])]),
                       ("", [ExtraTreesClassifier(bootstrap=True, oob_score=True, class_weight="balanced",
                                                  random_state=seed),
                             RandomForestClassifier(oob_score=True, class_weight="balanced", random_state=seed)],
                        [("criterion", ["gini", "log_loss"], [])]),
                       ("", [LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
                             PassiveAggressiveClassifier(class_weight="balanced", random_state=seed)],
                        [("C", np.logspace(-1, 1, 3), [])]),
                       ("", [Perceptron(class_weight="balanced", random_state=seed),
                             SGDClassifier(class_weight="balanced", random_state=seed)],
                        [("alpha", np.logspace(-5, -3, 3), [])]),
                       ("", [SVC(class_weight="balanced", max_iter=10000, random_state=seed)],
                        [("C", np.logspace(-1, 1, 3), []), ("kernel", ["linear", "poly", "rbf", "sigmoid"], [])])]
    clf_grid = _create_grid(clf_grid_roots, clf_grid_params)

    final_grid = _merge_grids([dim_red_ind_grid, dim_red_grid, clf_grid])
    cv_grid = GridSearchCV(pipe, final_grid, scoring="balanced_accuracy", n_jobs=n_jobs, pre_dispatch="n_jobs",
                           verbose=2)

    return cv_grid


def save_cv_results(cv_grid, antibiotic, data_path):
    pd.DataFrame(cv_grid.cv_results_).to_csv(os.path.join(data_path, "cv_results__{}.csv".format(antibiotic)))


def run_one(X_gpa, X_snps, X_genexp, y, antibiotic, data_path, seed, n_jobs):
    not_nan_idx = np.argwhere(np.logical_not(np.isnan(y))).flatten()
    X_gpa = X_gpa[not_nan_idx]
    X_snps = X_snps[not_nan_idx]
    X_genexp = X_genexp[not_nan_idx]
    y = y[not_nan_idx].astype(int)

    cache_path = os.path.join(data_path, ".cache")
    pipe = build_pipeline(X_gpa, X_snps, X_genexp, cache_path)
    cv_grid = build_hp_grid(pipe, seed, n_jobs)

    X = np.concatenate([X_gpa, X_snps, X_genexp], axis=1)
    cv_grid = cv_grid.fit(X, y)

    save_cv_results(cv_grid, antibiotic, data_path)
    shutil.rmtree(cache_path)


def main(data_path, seed, n_jobs):
    np.random.seed(seed)
    n_jobs = min(n_jobs, joblib.cpu_count())

    X_gpa, X_snps, X_genexp, Y = read_data(data_path)
    antibiotics = list(Y)

    for antibiotic in antibiotics:
        print("Fitting {}".format(antibiotic))
        y = Y[antibiotic].to_numpy()
        try:
            run_one(X_gpa, X_snps, X_genexp, y, antibiotic, data_path, seed, n_jobs)
        except:
            print("FITTING FAILED FOR {}".format(antibiotic))
            print(traceback.format_exc())
        else:
            print("Fitting done for {}".format(antibiotic))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data_path', help='data path to use', type=str, required=True)
    argparser.add_argument('-s', '--seed', help='seed to use', type=int, required=False, default=15)
    argparser.add_argument('-j', '--n_jobs', help='number of jobs', type=int, required=False, default=1)
    args = argparser.parse_args()
    main(args.data_path, args.seed, args.n_jobs)
