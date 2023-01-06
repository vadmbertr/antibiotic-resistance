import argparse
from itertools import product
import os
import pickle
import traceback

from multipy.fdr import lsu
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ttest_ind


def read_data(data_path):
    with open(os.path.join(data_path, "dataset.pkl"), "rb") as f:
        data = pickle.load(f)
        Y = data["pheno"].iloc[:, 1:]
        X_gpa = data["X_gpa"]
        X_snps = data["X_snps"]
        X_genexp = data["X_genexp"]

    return X_gpa, X_snps, X_genexp, Y


def save_selected_regressors(selected_regressors, antibiotic, save_path):
    with open(os.path.join(save_path, "selected_regressors__{}.pkl".format(antibiotic)), "wb") as f:
        pickle.dump(selected_regressors, f)


def run_one(X_gpa, X_snps, X_genexp, Y, antibiotic, save_path):
    y = Y[antibiotic].to_numpy()

    # there is no missing value in the regressors but there are in the target
    mask = np.isfinite(y)
    X_gpa = X_gpa[mask]
    X_snps = X_snps[mask]
    X_genexp = X_genexp[mask]
    y = y[mask].astype(int)
    res_idx = np.where(y == 1)
    sens_idx = np.where(y == 0)

    selected_regressors = {}

    for X in (X_gpa, X_snps):
        pval = np.apply_along_axis(lambda x: proportions_ztest((sum(x[res_idx]), sum(x[sens_idx])),
                                                               (len(x[res_idx]), len(x[sens_idx])))[1], 0, X)
        selected_regressors[X.shape[1]] = lsu(pval)

    pval = np.apply_along_axis(lambda x: ttest_ind(x[res_idx], x[sens_idx])[1], 0, X_genexp)
    selected_regressors[X_genexp.shape[1]] = lsu(pval)

    save_selected_regressors(selected_regressors, antibiotic, save_path)


def main(data_path, seed):
    np.random.seed(seed)
    save_path = os.path.join(data_path, "results/mul_test")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_gpa, X_snps, X_genexp, Y = read_data(data_path)
    antibiotics = list(Y)

    for antibiotic in antibiotics:
        print("Testing {}".format(antibiotic))

        try:
            run_one(X_gpa.copy(), X_snps.copy(), X_genexp.copy(), Y, antibiotic, save_path)
        except:
            print("TESTING FAILED FOR {}".format(antibiotic))
            print(traceback.format_exc())
        else:
            print("Testing done for {}".format(antibiotic))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data_path', help='data path to use', type=str, required=True)
    argparser.add_argument('-s', '--seed', help='seed to use', type=int, required=False, default=15)
    args = argparser.parse_args()
    main(args.data_path, args.seed)
