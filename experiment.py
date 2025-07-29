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
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, auc, f1_score


import pickle

num_repeats = 100

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
    probas = clf.predict_proba(selected_features)[:, 1]

    auc_score = roc_auc_score(test_labels, probas)

    fpr, tpr, thresholds = roc_curve(test_labels, probas)
    youden_index = tpr - fpr
    optimal_idx = youden_index.argmax()
    optimal_threshold = thresholds[optimal_idx]

    preds_optimal = (probas >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds_optimal).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = f1_score(test_labels, preds_optimal)

    precision, recall, _ = precision_recall_curve(test_labels, probas)
    auprc = auc(recall, precision)

    return auc_score, sensitivity, specificity, f1, auprc



def create_table ():
    tbls = []
    for dataset in ["CRLM", "Desmoid", "GIST", "Lipo"]:
        dataset_a, dataset_b = get_dataset_pairs(f"{root}/{dataset}")
        dataset_a_labels = [0 if f"/{dogsClass}/" in g else 1 for g in dataset_a]
        _, (minorcl, majorcl) = np.unique(dataset_a_labels, return_counts = True)
        if minorcl > majorcl:
            minorcl, majorcl = majorcl, minorcl
        bal = majorcl/minorcl
        tbls.append({"Dataset": dataset,
                "Modality": "",
                "Sample Size": len(dataset_a),
                "Size of minor class": minorcl,
                "Size of major class": majorcl,
                "Class-balance": bal,
                "In-plane resolution": ".",
                "Slice thickness": "."})
    tbl = pd.DataFrame(tbls)
    tbl.to_excel("./results/Table_1_raw.xlsx")



def run_single_repeat(r, root, ccc_threshold):
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

    feature_sets = {
        'Full': features_a,
        'Repro': features_a[:, repro_feats],
        'NonRepro': features_a[:, non_repro_feats]
    }
    # print (ccc_threshold, feature_sets["Full"].shape[1], feature_sets["Repro"].shape[1], feature_sets["NonRepro"].shape[1], features_b.shape[1])

    test_features_a, _ = extract_features(test_a, names_a)
    test_feature_sets = {
        'Full': test_features_a,
        'Repro': test_features_a[:, repro_feats],
        'NonRepro': test_features_a[:, non_repro_feats]
    }

    results = {}
    for fs_name in ['Full', 'Repro', 'NonRepro']:
        current_model = run_ml_pipeline(feature_sets[fs_name], labels_a)
        auc, sens, spec, f1, auprc = evaluate_model(current_model, test_feature_sets[fs_name], test_labels_a)

        # Save to results dictionary with suffix
        results[f'AUC-{fs_name}-A-A'] = auc
        results[f'Sensitivity-{fs_name}-A-A'] = sens
        results[f'Specificity-{fs_name}-A-A'] = spec
        results[f'AUPRC-{fs_name}-A-A'] = auprc
        results[f'F1-{fs_name}-A-A'] = f1

    results.update({
        "CCC-Threshold": ccc_threshold
    })

    print(".", end='', flush=True)
    return results



def run_experiment(root, num_repeats=30, ccc_threshold=0.85):
    results = Parallel(n_jobs=30)(
        delayed(run_single_repeat)(r, root, ccc_threshold)
        for r in range(num_repeats)
    )
    return results


# df, mm, file_path, dataset = df_radiomics.copy(), k, image_path, dataset
def plot_metrics(df, mm, file_path, dataset):
    tmp_df = df.copy()
    grouped = tmp_df.groupby('CCC-Threshold')
    grouped_df = grouped.mean().reset_index().sort_values('CCC-Threshold', ascending=False)
    se_df = grouped.sem().reset_index().sort_values('CCC-Threshold', ascending=False)

    # Extract AUC values
    auc_full_aa = grouped_df[f"{mm}-Full-A-A"]
    auc_repro_aa = grouped_df[f"{mm}-Repro-A-A"]
    auc_nonrepro_aa = grouped_df[f"{mm}-NonRepro-A-A"]

    # Compute confidence intervals
    z_score = 1.96
    ci_full_aa = z_score * se_df[f"{mm}-Full-A-A"]
    ci_repro_aa = z_score * se_df[f"{mm}-Repro-A-A"]
    ci_nonrepro_aa = z_score * se_df[f"{mm}-NonRepro-A-A"]

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
    plt.ylabel(mm)
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
    image_paths = {}
    for k in ["AUC", "Sensitivity", "Specificity", "AUPRC", "F1"]:
        image_paths[k] = []

    for dataset in ["CRLM", "Desmoid", "GIST", "Lipo"]:
        if os.path.exists(f"./results/radiomics_results_{dataset}.xlsx"):
            df_radiomics = pd.read_excel(f"./results/radiomics_results_{dataset}.xlsx")
        else:
            print (f"Computing {dataset}")
            radiomics_results_all = []
            for ccc_threshold in np.arange(0.95, 0.65, -0.01):
            # for ccc_threshold in np.arange(0.65, 0.95, 0.01):
                radiomics_results = run_experiment(f'{root}/{dataset}', num_repeats=num_repeats, ccc_threshold=ccc_threshold)
                if ccc_threshold == 0.80:
                    radiomics_080 = radiomics_results.copy()
                radiomics_results_all.extend(radiomics_results)

            df_radiomics = pd.DataFrame(radiomics_results_all)
            df_radiomics.to_excel(f"./results/radiomics_results_{dataset}.xlsx", index=False)

        for k in ["AUC", "Sensitivity", "Specificity", "AUPRC", "F1"]:
            image_path = f"./results/radiomics_{dataset}_{k}.png"
            plot_metrics(df_radiomics, k, image_path, dataset)
            image_paths[k].append(image_path)

    for k in ["AUC", "Sensitivity", "Specificity", "AUPRC", "F1"]:
        create_figure2(image_paths[k], f"./results/Figure_2_{k}.png", margin=20)

    create_table()


if __name__ == "__main__":
    main()

#
