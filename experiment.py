import os
import glob
import random

import matplotlib
matplotlib.use('Agg') # avoid tkinter errors
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


import pickle


dogsClass = "0"
catsClass = "1"


def get_dataset_pairs(root):
    cats_A, cats_B = list(zip(*[sorted(glob.glob(g+"/*.png"))[0:3:2] for g in glob.glob(root+f"/{catsClass}/*/")]))
    dogs_A, dogs_B = list(zip(*[sorted(glob.glob(g+"/*.png"))[0:3:2] for g in glob.glob(root+f"/{dogsClass}/*/")]))
    dataset_a = cats_A + dogs_A
    dataset_b = cats_B + dogs_B
    assert [os.path.basename(a).replace("_0", "_2") for a in dataset_a] == [os.path.basename(b) for b in dataset_b]
    return dataset_a, dataset_b



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


def extract_features(files, remove_const = True):
    features = []
    for file in files:
        fv = pd.read_csv(file.replace(".png", ".csv"))
        assert len(fv) == 1
        features.append(fv.iloc[0])

    features = pd.DataFrame(features).drop(["Unnamed: 0"], axis = 1)
    # drop what is constant
    if remove_const is True:
        features = features.loc[:, features.nunique() > 1]
    elif isinstance(remove_const, list):
        features = features[remove_const]  # Keep only specified columns
    return features.values, list(features.keys())


def run_ml_pipeline(features, labels):
    fsel = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)).fit(features, labels)
    selected_features = fsel.transform(features)
    clf = RandomForestClassifier(n_estimators = 250).fit(selected_features, labels)
    return fsel, clf



def evaluate_model(model, test_features, test_labels):
    fsel, clf = model
    selected_features = fsel.transform(test_features)
    preds = clf.predict_proba(selected_features)[:, 1]
    return roc_auc_score(test_labels, preds)



def run_single_repeat(r, method, root, ccc_threshold, num_features):
    np.random.seed(r)
    random.seed(r)

    dataset_a, dataset_b = get_dataset_pairs(root)
    dataset_a_labels = [0 if f"/{dogsClass}/" in g else 1 for g in dataset_a]
    dataset_b_labels = [0 if f"/{dogsClass}/" in g else 1 for g in dataset_b]
    assert dataset_a_labels == dataset_b_labels
    #train_a, test_a, train_b, test_b = train_test_split(dataset_a, dataset_b, test_size=50, random_state=r)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=r)
    for train_idx, test_idx in split.split(dataset_a, dataset_a_labels):
        train_a, test_a = [dataset_a[i] for i in train_idx], [dataset_a[i] for i in test_idx]
        train_b, test_b = [dataset_b[i] for i in train_idx], [dataset_b[i] for i in test_idx]

    labels_a = [0 if f"/{dogsClass}/" in g else 1 for g in train_a]
    labels_b = [0 if f"/{dogsClass}/" in g else 1 for g in train_b]
    test_labels_a = [0 if f"/{dogsClass}/" in g else 1 for g in test_a]
    test_labels_b = [0 if f"/{dogsClass}/" in g else 1 for g in test_b]

    # need to save the names of the features to select the very same ones from b
    features_a, names_a = extract_features(train_a)
    features_b, _ = extract_features(train_b, names_a)

    ccc_values = compute_ccc(features_a, features_b)
    repro_feats = np.abs(ccc_values) >= ccc_threshold
    non_repro_feats = np.abs(ccc_values) < ccc_threshold

    filtered_features_a_repro = features_a[:, repro_feats]
    filtered_features_b_repro = features_b[:, repro_feats]
    filtered_features_a_nonrepro = features_a[:, non_repro_feats]
    filtered_features_b_nonrepro = features_b[:, non_repro_feats]

    full_model_a = run_ml_pipeline(features_a, labels_a)
    #full_model_b = run_ml_pipeline(features_b, labels_b)

    test_features_a = extract_features(test_a, names_a)[0]
    test_features_b = extract_features(test_b, names_a)[0]

    auc_full_aa = evaluate_model(full_model_a, test_features_a, test_labels_a)

    restricted_model_repro = run_ml_pipeline(filtered_features_a_repro, labels_a)
    auc_repro_aa = evaluate_model(restricted_model_repro, test_features_a[:, repro_feats], test_labels_a)

    restricted_model_non_repro = run_ml_pipeline(filtered_features_a_nonrepro, labels_a)
    auc_nonrepro_aa = evaluate_model(restricted_model_non_repro, test_features_a[:, non_repro_feats], test_labels_a)

    print(".", end='', flush=True)

    return {
        "Method": method,
        "CCC-Threshold": ccc_threshold,
        "Number-Features": num_features,
        "AUC-Full-A-A": auc_full_aa,
        "AUC-Repro-A-A": auc_repro_aa,
        "AUC-NonRepro-A-A": auc_nonrepro_aa,
    }



def run_experiment(method, root, num_repeats=30, ccc_threshold=0.85, num_features=50):
    results = Parallel(n_jobs=30)(
        delayed(run_single_repeat)(r, method, root, ccc_threshold, num_features)
        for r in range(num_repeats)
    )
    return results



def plot_auc(df, file_path, dataset):
    tmp_df = df.drop(["Method"], axis=1)
    grouped = tmp_df.groupby('CCC-Threshold')
    grouped_df = grouped.mean().reset_index().sort_values('CCC-Threshold', ascending=False)
    se_df = grouped.sem().reset_index().sort_values('CCC-Threshold', ascending=False)

    # Extract AUC values
    auc_full_aa = grouped_df["AUC-Full-A-A"]
    auc_repro_aa = grouped_df["AUC-Repro-A-A"]
    auc_nonrepro_aa = grouped_df["AUC-NonRepro-A-A"]

    # Compute confidence intervals
    z_score = 1.96
    ci_full_aa = z_score * se_df["AUC-Full-A-A"]
    ci_repro_aa = z_score * se_df["AUC-Repro-A-A"]
    ci_nonrepro_aa = z_score * se_df["AUC-NonRepro-A-A"]

    x_values = grouped_df['CCC-Threshold']

    plt.figure(figsize=(6.3, 4.5))

    alpha = 0.1
    plt.plot(x_values, auc_full_aa, label="All Features", color="black")
    plt.fill_between(x_values, auc_full_aa - ci_full_aa, auc_full_aa + ci_full_aa, color="black", alpha=alpha)
    plt.plot(x_values, auc_repro_aa, label="Reproducible Features", color="blue")
    plt.fill_between(x_values, auc_repro_aa - ci_repro_aa, auc_repro_aa + ci_repro_aa, color="blue", alpha=alpha)
    plt.plot(x_values, auc_nonrepro_aa, label="Non-reproducible Features", color="red")
    plt.fill_between(x_values, auc_nonrepro_aa - ci_nonrepro_aa, auc_nonrepro_aa + ci_nonrepro_aa, color="red", alpha=alpha)

    plt.xlim(max(x_values), min(x_values))
    plt.xlabel('CCC-Threshold')
    plt.ylabel('AUC')
    plt.legend(ncol=1, loc='lower center')#, bbox_to_anchor=(0.5, 1.15))
    plt.text(0.02, 0.04, dataset, fontsize=20, transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()



def create_figure2(image_paths, output_path, margin=20):
    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)

    total_width = 2 * max_width + 3 * margin
    total_height = 2 * max_height + 3 * margin

    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    positions = [
        (margin, margin),
        (max_width + 2 * margin, margin),
        (margin, max_height + 2 * margin),
        (max_width + 2 * margin, max_height + 2 * margin)
    ]

    for img, pos in zip(images, positions):
        new_image.paste(img, pos)

    new_image.save(output_path)



def main():
    root = "slices"
    image_paths = []
    for dataset in ["CRLM", "Desmoid", "GIST", "Lipo"]:
        if os.path.exists(f"./results/radiomics_results_{dataset}.xlsx"):
            df_radiomics = pd.read_excel(f"./results/radiomics_results_{dataset}.xlsx")
        else:
            print (f"Computing {dataset}")
            radiomics_results_all = []
            for ccc_threshold in np.arange(0.95, 0.65, -0.01):
                radiomics_results = run_experiment("radiomics", f'{root}/{dataset}', num_repeats=100, ccc_threshold=ccc_threshold)
                if ccc_threshold == 0.80:
                    radiomics_080 = radiomics_results.copy()
                radiomics_results_all.extend(radiomics_results)

            df_radiomics = pd.DataFrame(radiomics_results_all)
            df_radiomics.to_excel(f"./results/radiomics_results_{dataset}.xlsx", index=False)
        image_path = f"./results/radiomics_{dataset}.png"
        plot_auc(df_radiomics, image_path, dataset)
        image_paths.append(image_path)

    create_figure2(image_paths, "./results/Figure_2.png", margin=20)


if __name__ == "__main__":
    main()



#
