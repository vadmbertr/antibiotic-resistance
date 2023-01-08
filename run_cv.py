import argparse
import os
import pandas as pd
import pickle
import traceback

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from custom_transformers.stability_selection import StabilitySelection


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


def _custom_parser(params, n_jobs):
    params = params.replace("{", "")
    params = params.replace("}", "")
    chunks = params.split(", '")
    parsed_params = {}
    for chunk in chunks:
        if "StabilitySelectionTransformer" in chunk:
            key = "dim_red"
            value = StabilitySelection(n_jobs=n_jobs)
        else:
            key, value = chunk.split(': ')
            key = key.replace("'", "")
            value = eval(value)  # !! can be dangerous !!
        parsed_params[key] = value
    return parsed_params


def save_cv_scores(cv_scores, antibiotic, save_path):
    np.save(os.path.join(save_path, "cv_scores__{}".format(antibiotic)), cv_scores)


def save_cv_pred(cv_pred, antibiotic, save_path):
    np.save(os.path.join(save_path, "cv_pred__{}".format(antibiotic)), cv_pred)


def run_one(X_gpa, X_snps, X_genexp, Y, antibiotic, seed, n_jobs, grid_search_path, save_path, n_splits=5):
    y = Y[antibiotic].to_numpy()

    # there is no missing value in the regressors but there are in the target
    mask = np.isfinite(y)
    X_gpa = X_gpa[mask]
    X_snps = X_snps[mask]
    X_genexp = X_genexp[mask]
    y = y[mask].astype(int)

    pipe = build_pipeline(X_gpa, X_snps, X_genexp)

    cv_results = pd.read_csv(os.path.join(grid_search_path, "cv_results__{}.csv".format(antibiotic)))
    params = cv_results.sort_values("mean_test_score", axis=0, ascending=False)["params"].iloc[0]
    pipe.set_params(**_custom_parser(params, n_jobs))

    X = np.concatenate([X_gpa, X_snps, X_genexp], axis=1)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba_all = np.zeros(y.shape)
    cv_scores = np.zeros(n_splits)
    for (i, (train_idx, test_idx)) in zip(np.arange(0, n_splits), cv.split(X, y)):
        pipe.fit(X[train_idx], y[train_idx])
        try:
            y_proba = pipe.predict_proba(X[test_idx])[:, 1]
        except AttributeError:
            y_proba = pipe.predict(X[test_idx])
        y_proba_all[test_idx] = y_proba
        cv_scores[i] = balanced_accuracy_score(y[test_idx], (y_proba >= .5).astype(int))

    save_cv_pred(y_proba_all, antibiotic, save_path)
    save_cv_scores(cv_scores, antibiotic, save_path)


def main(data_path, seed, n_jobs):
    np.random.seed(seed)
    n_jobs = min(n_jobs, joblib.cpu_count() - 1)
    grid_search_path = os.path.join(data_path, "results/grid_search")
    save_path = os.path.join(data_path, "results/cv")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_gpa, X_snps, X_genexp, Y = read_data(data_path)
    antibiotics = list(Y)

    for antibiotic in antibiotics:
        print("Fitting {}".format(antibiotic))

        try:
            run_one(X_gpa.copy(), X_snps.copy(), X_genexp.copy(), Y, antibiotic, seed, n_jobs, grid_search_path,
                    save_path)
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
