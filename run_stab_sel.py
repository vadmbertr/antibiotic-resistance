import argparse
from itertools import product
import os
import pickle
import traceback

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from custom_transformers.stability_selection import StabilitySelection
from custom_transformers.standard_true_false import standard_true_false


def read_data(data_path):
    with open(os.path.join(data_path, "dataset.pkl"), "rb") as f:
        data = pickle.load(f)
        Y = data["pheno"].iloc[:, 1:]
        X_gpa = standard_true_false.fit_transform(data["X_gpa"])
        X_snps = standard_true_false.fit_transform(data["X_snps"])
        X_genexp = StandardScaler().fit_transform(data["X_genexp"])

    return X_gpa, X_snps, X_genexp, Y


def save_stability_scores(stability_scores, antibiotic, save_path):
    with open(os.path.join(save_path, "stability_scores__{}.pkl".format(antibiotic)), "rb") as f:
        pickle.dump(stability_scores, f)


def run_one(X_gpa, X_snps, X_genexp, Y, antibiotic, seed, n_jobs, save_path):
    y = Y[antibiotic].to_numpy()

    # there is no missing value in the regressors but there are in the target
    mask = np.isfinite(y)
    X_gpa = X_gpa[mask]
    X_snps = X_snps[mask]
    X_genexp = X_genexp[mask]
    regressors = [X_gpa, X_snps, X_genexp]
    y = y[mask].astype(int)

    stability_scores = {}
    for comb in product([True, False], repeat=3):
        X = []
        for i in range(len(comb)):
            if comb[i]:
                X.append(regressors[i])
        X = np.concatenate(X, axis=1)

        stab_sel = StabilitySelection(random_state=seed, n_jobs=n_jobs)
        stab_sel = stab_sel.fit(X, y)
        stability_scores[X.shape[1]] = stab_sel.stability_scores

    save_stability_scores(stability_scores, antibiotic, save_path)


def main(data_path, seed, n_jobs):
    np.random.seed(seed)
    n_jobs = min(n_jobs, joblib.cpu_count())
    save_path = os.path.join(data_path, "results/stab_sel")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_gpa, X_snps, X_genexp, Y = read_data(data_path)
    antibiotics = list(Y)

    for antibiotic in antibiotics:
        print("Fitting {}".format(antibiotic))

        try:
            run_one(X_gpa.copy(), X_snps.copy(), X_genexp.copy(), Y, antibiotic, seed, n_jobs, save_path)
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
