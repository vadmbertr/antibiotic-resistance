import argparse
from itertools import combinations, product
import os
import pandas as pd
import pickle
import traceback

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from custom_transformers.multiple_testing import MultipleTesting
from custom_transformers.stability_selection import StabilitySelectionTransformer


def read_data(data_path):
    with open(os.path.join(data_path, "dataset.pkl"), "rb") as f:
        data = pickle.load(f)
        Y = data["pheno"].iloc[:, 1:]
        X_gpa = data["X_gpa"]
        X_snps = data["X_snps"]
        X_genexp = data["X_genexp"]

    return X_gpa, X_snps, X_genexp, Y


def build_pipeline(X_gpa, X_snps, X_genexp):
    gpa_idx = np.arange(0, X_gpa.shape[1])
    snps_idx = np.arange(0, X_snps.shape[1]) + gpa_idx[-1] + 1
    genexp_idx = np.arange(0, X_genexp.shape[1]) + snps_idx[-1] + 1

    trans_ind = ColumnTransformer(transformers=[("gpa", "passthrough", gpa_idx),
                                                ("snps", "passthrough", snps_idx),
                                                ("genexp", StandardScaler(), genexp_idx)],
                                  remainder="drop")
    sel_ind = ColumnTransformer(transformers=[("gpa", "passthrough", gpa_idx),
                                              ("snps", "passthrough", snps_idx),
                                              ("genexp", "passthrough", genexp_idx)],
                                  remainder="drop")

    pipe = Pipeline([("trans_ind", trans_ind), ("sel_ind", sel_ind), ("dim_red", "passthrough"),
                     ("clf", DummyClassifier())])

    return pipe


def _get_stab_sel_trans(stab_sel_path):
    stab_sel_trans = None

    if os.path.exists(stab_sel_path):
        with open(stab_sel_path, "rb") as f:
            stability_scores = pickle.load(f)
        stab_sel_trans = StabilitySelectionTransformer(stability_scores=stability_scores)

    return stab_sel_trans


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


def build_hp_grid(pipe, seed, n_jobs, stab_sel_path, mul_test_path):
    sel_ind_grid_roots = ["sel_ind__gpa", "sel_ind__snps", "sel_ind__genexp"]
    sel_ind_grid_params = [("", ["drop", "passthrough"], [])]
    sel_ind_grid = _create_grid(sel_ind_grid_roots, sel_ind_grid_params)

    dim_red_grid_roots = ["dim_red"]
    dim_red_grid_params = [("", ["passthrough", ], []),
                           ("", [KernelPCA(random_state=seed), ],
                            [("kernel", ["linear", "poly", "rbf", "sigmoid"], []),
                             ("n_components", [64, 128, 256], [])]),
                           ("", [MultipleTesting(), ], ["alpha", (.01, .05, .1)])]
    stab_sel_trans = _get_stab_sel_trans(stab_sel_path)
    if stab_sel_trans is not None:
        dim_red_grid_params.append(("", [stab_sel_trans, ], [("threshold", np.linspace(.6, .9, 4), [])]))
    dim_red_grid = _create_grid(dim_red_grid_roots, dim_red_grid_params)

    clf_grid_roots = ["clf"]
    clf_grid_params = [("", [AdaBoostClassifier(random_state=seed), GradientBoostingClassifier(random_state=seed)],
                        [("learning_rate", np.logspace(-2, 0, 3), [])]),
                       ("", [RandomForestClassifier(class_weight="balanced", random_state=seed)],
                        [("n_estimators", [100, 300, 500], []), ("max_depth", [None, 10, 100], []),
                         ("max_features", ["sqrt", "log2"], [])]),
                       ("", [LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                                max_iter=10000, random_state=seed)],
                        [("C", np.logspace(-1, 1, 3), [])]),
                       ("", [SGDClassifier(penalty="l1", class_weight="balanced", random_state=seed)],
                        [("loss", ["hinge", "log_loss"], []), ("alpha", np.logspace(-5, -3, 3), [])]),
                       ("", [SVC(class_weight="balanced", random_state=seed)],
                        [("C", np.logspace(-1, 1, 3), []), ("kernel", ["linear", "poly", "rbf", "sigmoid"], [])])]
    clf_grid = _create_grid(clf_grid_roots, clf_grid_params)

    final_grid = _merge_grids([sel_ind_grid, dim_red_grid, clf_grid])
    cv_grid = GridSearchCV(pipe, final_grid, scoring="balanced_accuracy", n_jobs=n_jobs, verbose=2)

    return cv_grid


def save_cv_results(cv_grid, antibiotic, save_path):
    pd.DataFrame(cv_grid.cv_results_).to_csv(os.path.join(save_path, "cv_results__{}.csv".format(antibiotic)))


def run_one(X_gpa, X_snps, X_genexp, Y, antibiotic, seed, n_jobs, stab_sel_path, mul_test_path, save_path):
    y = Y[antibiotic].to_numpy()

    # there is no missing value in the regressors but there are in the target
    mask = np.isfinite(y)
    X_gpa = X_gpa[mask]
    X_snps = X_snps[mask]
    X_genexp = X_genexp[mask]
    y = y[mask].astype(int)

    pipe = build_pipeline(X_gpa, X_snps, X_genexp)
    cv_grid = build_hp_grid(pipe, seed, n_jobs,
                            os.path.join(stab_sel_path, "stability_scores__{}.pkl".format(antibiotic)),
                            os.path.join(mul_test_path, "selected_regressors__{}.pkl".format(antibiotic)))

    X = np.concatenate([X_gpa, X_snps, X_genexp], axis=1)
    cv_grid = cv_grid.fit(X, y)

    save_cv_results(cv_grid, antibiotic, save_path)


def main(data_path, seed, n_jobs):
    np.random.seed(seed)
    n_jobs = min(n_jobs, joblib.cpu_count() - 1)
    stab_sel_path = os.path.join(data_path, "results/stab_sel")
    mul_test_path = os.path.join(data_path, "results/mul_test")
    save_path = os.path.join(data_path, "results/grid_search")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_gpa, X_snps, X_genexp, Y = read_data(data_path)
    antibiotics = list(Y)

    for antibiotic in antibiotics:
        print("Fitting {}".format(antibiotic))

        try:
            run_one(X_gpa.copy(), X_snps.copy(), X_genexp.copy(), Y, antibiotic, seed, n_jobs, stab_sel_path,
                    mul_test_path, save_path)
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
