import argparse
import os
import traceback
import pickle

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from custom_transformers.stability_selection import StabilitySelection
from custom_transformers.standard_true_false import standard_true_false


def read_data(data_path):
    with open(os.path.join(data_path, "dataset.pkl"), "rb") as f:
        data = pickle.load(f)
        Y = data["pheno"].iloc[:, 1:]
        X_gpa = data["X_gpa"]
        X_snps = data["X_snps"]
        X_genexp = data["X_genexp"]

    return X_gpa, X_snps, X_genexp, Y


def get_X_transformer(gpa_idx, snps_idx, genexp_idx):
    return ColumnTransformer(transformers=[("gpa", standard_true_false, gpa_idx),
                                           ("snps", standard_true_false, snps_idx),
                                           ("genexp", StandardScaler(), genexp_idx)],
                             remainder="drop")


def get_f0_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed):
    pipe = Pipeline([("trans", get_X_transformer(gpa_idx, snps_idx, genexp_idx)),
                     ("dim_red", StabilitySelection(random_state=15, threshold=.9)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, criterion="gini"))])
    return pipe


def get_f1_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed):
    pipe = Pipeline([("trans", get_X_transformer(gpa_idx, snps_idx, genexp_idx)),
                     ("dim_red", StabilitySelection(random_state=15, threshold=.9)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, criterion="gini"))])
    return pipe


def get_f2_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed):
    pipe = Pipeline([("trans", get_X_transformer(gpa_idx, snps_idx, genexp_idx)),
                     ("dim_red", StabilitySelection(random_state=15, threshold=.9)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, criterion="gini"))])
    return pipe


def get_f3_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed):
    pipe = Pipeline([("trans", get_X_transformer(gpa_idx, snps_idx, genexp_idx)),
                     ("dim_red", StabilitySelection(random_state=15, threshold=.9)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, criterion="gini"))])
    return pipe


def get_f4_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed):
    pipe = Pipeline([("trans", get_X_transformer(gpa_idx, snps_idx, genexp_idx)),
                     ("dim_red", StabilitySelection(random_state=15, threshold=.9)),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, criterion="gini"))])
    return pipe


def get_voting_clf(f0_pipeline, f1_pipeline, f2_pipeline, f3_pipeline, f4_pipeline):
    return VotingClassifier([("f0", f0_pipeline), ("f1", f1_pipeline), ("f2", f2_pipeline), ("f3", f3_pipeline),
                             ("f4", f4_pipeline)], voting="soft")


def save_cv_results(res, antibiotic, save_path):
    np.save(os.path.join(save_path, "cv_results__{}.npy".format(antibiotic)), res)


def run_one(X_gpa, X_snps, X_genexp, Y, antibiotic, seed, n_jobs, save_path):
    y = Y[antibiotic].to_numpy()

    # there is no missing value in the regressors but there are in the target
    mask = np.isfinite(y)
    X_gpa = X_gpa[mask]
    X_snps = X_snps[mask]
    X_genexp = X_genexp[mask]
    y = y[mask].astype(int)

    gpa_idx = np.arange(0, X_gpa.shape[1])
    snps_idx = np.arange(0, X_snps.shape[1]) + gpa_idx[-1] + 1
    genexp_idx = np.arange(0, X_genexp.shape[1]) + snps_idx[-1] + 1

    f0_pipeline = get_f0_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed)
    f1_pipeline = get_f1_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed)
    f2_pipeline = get_f2_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed)
    f3_pipeline = get_f3_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed)
    f4_pipeline = get_f4_pipeline(antibiotic, gpa_idx, snps_idx, genexp_idx, seed)

    voting_clf = get_voting_clf(f0_pipeline, f1_pipeline, f2_pipeline, f3_pipeline, f4_pipeline)

    res = cross_val_score(voting_clf, np.concatenate([X_gpa, X_snps, X_genexp], axis=1), y, scoring="balanced_accuracy",
                          n_jobs=n_jobs)

    save_cv_results(res, antibiotic, save_path)


def main(data_path, seed, n_jobs):
    np.random.seed(seed)
    n_jobs = min(n_jobs, joblib.cpu_count() - 1)
    save_path = os.path.join(data_path, "results/reg_voting")

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
