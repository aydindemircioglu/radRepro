import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed
import random


def generate_data(n_samples, n_features=3):
    X1 = np.zeros((n_samples, n_features))
    X2 = np.zeros((n_samples, n_features))
    for k in range(len(X1)):
        if k > n_samples//4: # ensure we have enough of both classes
            X1[k, np.random.randint(n_features)] = 1
            X2[k, np.random.randint(n_features)] = 1
    y = (np.any(X1 > 0.5, axis=1)).astype(int)
    return X1, X2, y



# https://nirpyresearch.com/concordance-correlation-coefficient/
def cc_coeff(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc



def compute_ccc(features_a, features_b):
    ccc_values = []
    for i in range(features_a.shape[1]):
        x = features_a[:, i]
        y = features_b[:, i]
        ccc = cc_coeff(x, y)
        ccc_values.append(ccc)
    return np.array(ccc_values)



def run_experiment(n_samples, n_repeats=10, n_features=3):
    np.random.seed(n_repeats)
    random.seed(n_repeats)

    aucs = []
    cccs = np.zeros((n_repeats, n_features))

    for i in range(n_repeats):
        X1, X2, y = generate_data(n_samples, n_features)
        ccc_values = compute_ccc(X1, X2)
        cccs[i, :] = ccc_values

        X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.5, random_state=42, stratify=y)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)

    # mean ccc
    ccc_means = np.array([np.mean(cccs)])
    ccc_stds = np.array([np.std(cccs)])

    return pd.DataFrame({
        'Sample Size': [n_samples],
        'AUC Mean': [np.mean(aucs)],
        'AUC Std': [np.std(aucs)],
        'AUC 2.5% CI': [np.percentile(aucs, 2.5)],
        'AUC 97.5% CI': [np.percentile(aucs, 97.5)],
        'CCC Means': [ccc_means],
        'CCC Stds': [ccc_stds]
    })


def plot_results(df):
    sample_sizes = df['Sample Size']
    ccc_means = np.array(df['CCC Means'].to_list())
    ccc_stds = np.array(df['CCC Stds'].to_list())

    plt.figure(figsize=(6.3, 4.5))
    for i in range(ccc_means.shape[1]):
        plt.plot(sample_sizes, ccc_means[:,i], label=f"Mean CCC")
        plt.fill_between(sample_sizes, ccc_means[:,i] - ccc_stds[:,i],
                ccc_means[:,i] + ccc_stds[:,i], alpha=0.15)
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean CCC')
    plt.tight_layout()
    plt.savefig("./results/elephant_house.png", dpi=600)
    plt.close()


def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    sample_sizes = range(15, 200, 1)
    n_repeats = 100
    n_features = 3

    results = Parallel(n_jobs=-1)(delayed(run_experiment)(n_samples, n_repeats=n_repeats, n_features=n_features) for n_samples in sample_sizes)
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_excel('results/elephant_house.xlsx', index=False)
    plot_results(all_results)


if __name__ == "__main__":
    main()
