# Fraud detection pipeline ordering

Code and archived results for *A Pipeline Ordering Framework for Hybrid Data Augmentation and Cleaning in Fraud Detection Under Extreme Class Imbalance*, by Basant Nabil, Ehab Ezzat Hassanein and Noha Nagy Mohy.

The study evaluates 35 configurations, including a non-augmented baseline. Five seeds, three classifiers and two datasets give 1,050 paper runs. The primary analysis uses credit card transactions from the Machine Learning Group of the Université Libre de Bruxelles (ULB). The secondary comparison uses the Base variant of the Bank Account Fraud (BAF) suite, a synthetic dataset of account applications.

## Match the archive to the supplement

| Supplementary table | Archived source |
| --- | --- |
| S1 and S5. ULB and BAF performance | The per-seed results CSV for the corresponding dataset. |
| S2. Cleaning counts | `Results/cleaning_counts_per_seed.csv`. |
| S3 and S4. Paired ULB comparisons | ULB per-seed results, matched by classifier and seed. The S3 calculation is included below. |
| S6(a–h). Threshold sensitivity and descriptive metric rankings | The `threshold_analysis` section of the individual run records for grid thresholds, and the `metrics` section for 0.35. |
| S6(i). BAF F1-score and PR-AUC ranks | BAF per-seed results, ranked across all 35 configurations within each classifier. |
| Implementation details: ULB synthetic sample allocation | The `sample_origins` section of the GAN-SMOTE and SMOTE-GAN run records. |

## Reproduce the reported summaries

Download the [code and results package](https://github.com/basantnabil-hub/fraud-detection-pipeline-ordering/blob/4ec828c507ab9d5ba596087877c9d8154d9ec191/fraud-detection-pipeline-ordering.rar) and extract it with a RAR5-compatible utility such as 7-Zip 23 or later, WinRAR or `unrar`. Some older `unar` versions stop with a read error on two files in `Code/`. The commands below start in the directory containing `Code/` and `Results/`. GitHub’s **Download ZIP** adds an outer ZIP to extract first. The per-seed summaries require pandas (`python -m pip install pandas`). You can run this example without downloading either raw dataset or training a model.

```python
from pathlib import Path

import pandas as pd

root = Path('Results')
pipeline_labels = pd.read_csv(root / 'paper_pipelines.csv')
runs = pd.concat([
    pd.read_csv(root / 'CC Dataset' / 'CC_per_seed_long.csv'),
    pd.read_csv(root / 'BAF Dataset' / 'BAF_per_seed_long.csv'),
], ignore_index=True)

keys = ['dataset', 'classifier', 'pipeline']
seeds = {42, 456, 2566, 1011, 9842}
assert len(pipeline_labels) == 35 and set(runs['pipeline']) == set(pipeline_labels['pipeline'])
assert set(runs['dataset']) == {'CC', 'BAF'}
assert set(runs['classifier']) == {'xgb', 'rf', 'nn'}
assert len(runs) == 1050 and runs['success'].all()
assert not runs.duplicated(keys + ['seed']).any()
assert runs.groupby(keys)['seed'].agg(set).map(lambda group: group == seeds).all()
assert runs['used_threshold'].eq(0.35).all()

metrics = ['Precision', 'Recall', 'F1', 'MCC', 'PR_AUC', 'ROC_AUC', 'FP', 'FN']
summary = runs.groupby(keys)[metrics].agg(['mean', 'std'])  # sample SD across five seeds
print(summary.loc[('CC', slice(None), 'baseline'), ['F1', 'MCC']].round(4))

baf = runs.loc[runs['dataset'] == 'BAF']
mean_f1 = baf.groupby(['classifier', 'pipeline'])['F1'].mean()
ranks = mean_f1.groupby(level='classifier').rank(ascending=False, method='min')
print(ranks[ranks == 1])
```

The ULB baseline mean F1-scores are 0.8746 for XGBoost, 0.8783 for Random Forest and 0.8396 for the feed-forward neural network (FNN). At the study threshold of 0.35, `smote_enn` ranks first by mean F1-score on BAF for XGBoost and Random Forest. `cluster_smote_gan_enn` ranks first for FNN. Ranks use unrounded means.

### Check the selected ULB precision comparison

The following example continues from `runs` above and requires SciPy (`python -m pip install scipy`). It pairs GAN–SMOTE and SMOTE–GAN by seed and compares their precision within each classifier. The conventional paired test and the repeated-holdout sensitivity calculation each have a three-classifier Holm adjustment. The sensitivity calculation uses the original outer ULB training and test sizes, including for the FNN, which also sets aside an internal training holdout.

```python
from math import sqrt
from scipy.stats import t

pair = runs.loc[runs['dataset'].eq('CC') &
                runs['pipeline'].isin(['gan_smote', 'smote_gan'])]
wide = pair.pivot(index=['classifier', 'seed'], columns='pipeline', values='Precision')
gaps = wide['gan_smote'] - wide['smote_gan']
by_model = gaps.groupby(level='classifier').agg(['count', 'mean', 'std'])
assert by_model['count'].eq(5).all()

for name, factor in [('paired', sqrt(1 / 5)),
                     ('corrected', sqrt(1 / 5 + 56962 / 199364))]:
    statistic = by_model['mean'] / (by_model['std'] * factor)
    p = pd.Series(2 * t.sf(statistic.abs(), df=4), index=statistic.index).sort_values()
    holm = (p * list(range(len(p), 0, -1))).cummax().clip(upper=1)
    by_model[name + '_holm_p'] = holm

print(by_model.loc[['xgb', 'rf', 'nn'], ['mean', 'paired_holm_p', 'corrected_holm_p']].round(4))
```

### Reproduce the threshold and metric rankings (Table S6)

The following code continues from the first example. It reads the archived JSON files directly from the six ZIPs, retains the 35 paper configurations and uses the recorded F1-scores. The grid is descriptive and was not used to tune a new decision threshold on the test sets.

```python
import json
from zipfile import ZipFile

paper_pipelines = set(pipeline_labels['pipeline'])
model_order = ['xgb', 'rf', 'nn']
thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
records = []
allocations = []

for dataset, folder in [('CC', 'CC Dataset'), ('BAF', 'BAF Dataset')]:
    for archive in sorted((root / folder).glob('*.zip')):
        with ZipFile(archive) as z:
            for name in z.namelist():
                if 'individual_runs/' not in name or not name.endswith('.json'):
                    continue
                run = json.loads(z.read(name))
                pipeline = run['pipeline']
                if not run['success'] or pipeline not in paper_pipelines:
                    continue
                key = (dataset, run['classifier'], pipeline, run['seed'])
                for entry in run['threshold_analysis']:
                    threshold = round(entry['threshold'], 2)
                    if threshold in thresholds and threshold != 0.35:
                        records.append((*key, threshold, entry['f1'], entry['precision']))
                metrics = run['metrics']
                records.append((*key, 0.35, metrics['F1_pos'], metrics['Precision_pos']))
                if dataset == 'CC' and pipeline in {'gan_smote', 'smote_gan'}:
                    origins = run['sample_origins']
                    allocations.append((*key, int(origins['GAN_GENERATED']),
                                        int(origins['SMOTE_GENERATED'])))

sweep = pd.DataFrame(records, columns=keys + ['seed', 'threshold', 'F1', 'Precision'])
assert len(sweep) == 10500
assert not sweep.duplicated(keys + ['seed', 'threshold']).any()
assert sweep.groupby(keys + ['threshold'])['seed'].agg(set).map(lambda group: group == seeds).all()
assert sweep.groupby(keys)['threshold'].agg(set).map(lambda group: group == set(thresholds)).all()

# Table S6(h): ULB baseline F1 ranks, across all 35 configurations.
ulb_grid = sweep.loc[sweep['dataset'].eq('CC')]
mean_grid = ulb_grid.groupby(['classifier', 'threshold', 'pipeline'])['F1'].mean()
grid_ranks = mean_grid.groupby(level=['classifier', 'threshold']).rank(ascending=False, method='min')
baseline_ranks = grid_ranks.xs('baseline', level='pipeline').unstack('threshold')
print(baseline_ranks.reindex(index=model_order, columns=thresholds).astype(int))

# Table S6(i): BAF mean F1 and PR-AUC, with ranks among all 35 configurations.
baf_means = runs.loc[runs['dataset'].eq('BAF')].groupby(['classifier', 'pipeline'])[['F1', 'PR_AUC']].mean()
baf_ranks = baf_means.groupby(level='classifier').rank(ascending=False, method='min').astype(int)
baf_table = baf_means.join(baf_ranks.add_suffix('_rank'))
selected = pd.MultiIndex.from_product([model_order, ['baseline', 'smote_enn']],
                                      names=['classifier', 'pipeline'])
print(baf_table.reindex(selected)[['F1', 'F1_rank', 'PR_AUC', 'PR_AUC_rank']].round(4))

# ULB synthetic sample allocation (Implementation details).
counts = pd.DataFrame(allocations, columns=keys + ['seed', 'GAN', 'SMOTE'])
counts['Total'] = counts['GAN'] + counts['SMOTE']
assert len(counts) == 30 and not counts.duplicated(keys + ['seed']).any()
assert counts.groupby('pipeline')[['GAN', 'SMOTE', 'Total']].nunique().eq(1).all().all()
assert counts['Total'].eq(21769).all()
print(counts.groupby('pipeline')[['GAN', 'SMOTE', 'Total']].first())
```

Expected ULB baseline F1 ranks (Table S6(h)):

| Classifier | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 15 | 35 |
| RF | 1 | 1 | 1 | 1 | 6 | 27 | 35 | 35 | 35 | 35 |
| FNN | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 6 | 7 |

Expected BAF results (Table S6(i)):

| Classifier | Configuration | Mean F1-score | F1 rank | Mean PR-AUC | PR-AUC rank |
| --- | --- | ---: | ---: | ---: | ---: |
| XGBoost | Baseline | 0.1224 | 33 | 0.1480 | 28 |
| XGBoost | SMOTE-ENN | 0.2220 | 1 | 0.1536 | 12 |
| RF | Baseline | 0.0127 | 35 | 0.1131 | 27 |
| RF | SMOTE-ENN | 0.1739 | 1 | 0.1023 | 31 |
| FNN | Baseline | 0.0992 | 33 | 0.0630 | 17 |
| FNN | SMOTE-ENN | 0.1425 | 2 | 0.0769 | 1 |

F1 uses the main threshold of 0.35. PR-AUC uses the precision–recall curve across thresholds. Both rankings use unrounded five-seed means, with rank 1 indicating the highest mean. The BAF comparisons remain descriptive because some configurations used different subsamples.

Expected ULB sample counts per training partition:

| Configuration | GAN-generated fraud | SMOTE-generated fraud | Total synthetic fraud |
| --- | ---: | ---: | ---: |
| GAN-SMOTE | 12,359 | 9,410 | 21,769 |
| SMOTE-GAN | 13,821 | 7,948 | 21,769 |

The total number of synthetic records is the same for both sequences. The allocation between augmentation stages differs.

### Check ULB duplicate overlap before augmentation

Place the original `creditcard.csv` in the working directory and install scikit-learn (`python -m pip install scikit-learn`). This example recreates the stratified splits in `data_prep.py`. It counts each test row once when all 30 feature values and its label match at least one original training row. The validation rows are excluded from the training set before checking for matches.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ulb = pd.read_csv('creditcard.csv')
labels = ulb['Class'].to_numpy()
groups = ulb.groupby(list(ulb.columns), sort=False, dropna=False).ngroup().to_numpy()
indices = np.arange(len(ulb))
print('seed, matching test rows, matching fraud test rows')
for seed in [42, 456, 2566, 1011, 9842]:
    pool, test = train_test_split(indices, test_size=0.20, stratify=labels, random_state=seed)
    train, _ = train_test_split(pool, test_size=0.125, stratify=labels[pool], random_state=seed)
    match = np.isin(groups[test], groups[train])
    print(seed, int(match.sum()), int(labels[test][match].sum()))
```

In seed order, 319, 270, 301, 326 and 280 test rows match training rows. Of those, 8, 3, 4, 6 and 5 are fraud rows, respectively. Each test set contains 98 fraud rows. The matches establish equality of the recorded features and label, not the identity of the underlying transactions. Their effect on performance is unknown.

### Files and metric conventions

| Path under `Results/` | Contents |
| --- | --- |
| `paper_pipelines.csv` | Internal identifiers and manuscript labels for the 35 configurations. |
| `CC Dataset/CC_per_seed_long.csv` and `BAF Dataset/BAF_per_seed_long.csv` | 525 runs each. These files are the starting point for the paper’s five-seed tables. |
| `cleaning_counts_per_seed.csv` | Per-run counts of original legitimate and fraud rows removed, with paths to the source records. |
| `cleaning_counts.csv` | Means and sample standard deviations of legitimate-row removals. |
| Three ZIPs per dataset directory | One per classifier. Each contains 190 successful run JSON files under `individual_runs/`, alongside summary and diagnostic outputs. |

- Precision, recall, F1-score, MCC and confusion counts in the per-seed CSVs use the fixed threshold of 0.35. PR-AUC is trapezoidal area under the precision–recall curve (`precision_recall_curve` followed by `auc(recall, precision)`). ROC-AUC and PR-AUC use the scores across decision thresholds.
- Use the five per-seed values for sample standard deviations. Historical summaries also exist in the ZIPs, some using population standard deviations. Each ZIP contains three exploratory configurations beyond the paper set (`smote_plus_gan_5050`, `cluster_enn_gan`, `cluster_gan_enn`). Filtering by `paper_pipelines.csv` gives the 1,050 paper records among 1,140 archived records.
- Each JSON’s top-level `threshold_analysis` holds metrics at 0.10 through 0.90 in 0.10 increments. Its `metrics` object holds the main result at 0.35. Full test score vectors are unavailable. The saved XGBoost arrays contain 2,000 scores per run. Profit fields in historical outputs use fixed illustrative cost assumptions.

## Run the models

Obtain the original dataset separately. The default input filenames are `creditcard.csv` for [ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and `Base.csv` for [BAF](https://github.com/feedzai/bank-account-fraud). ULB contains 284,807 transactions with target `Class`. The BAF loader selects a stratified 250,000-row subsample with target `fraud_bool`.

The archived runs used Google Colab with a T4 GPU. The archive does not include a complete historical package manifest. For a new run, use a Python environment supported by both TensorFlow and PyTorch. The maintained code imports the model libraries when starting the runner, including for a tree-only batch.

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or macOS, or `.venv\Scripts\activate` in the Windows command prompt. Install the training dependencies and inspect the options.

```bash
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost matplotlib tensorflow torch
python Code/run.py --help
```

For a five-seed ULB baseline, replace the path with the local dataset path.

```bash
python Code/run.py --dataset ulb --data-path /path/to/creditcard.csv --classifier rf --pipelines baseline --output-name ulb_rf_baseline
```

For the paired XGBoost comparison, select the pipeline names individually.

```bash
python Code/run.py --dataset ulb --data-path /path/to/creditcard.csv --classifier xgb --pipelines gan_smote smote_gan --output-name ulb_xgb_order
```

`--classifier all --pipelines all` schedules 525 runs for one dataset. For BAF, pass `--dataset baf`, the path to `Base.csv` and a separate output name. Calling `python Code/run.py` without arguments opens the interactive menu. In Colab, use `%run Code/run.py` from the extracted project directory. Use a fresh output name when changing data or configuration. The runner can otherwise reuse successful runs already saved under that name.

## Implementation details

| Module in `Code/` | Role |
| --- | --- |
| `config.py`, `data_prep.py` | Settings, dataset selection, preparation, stratified splitting and feature scaling. |
| `sampling_ops.py`, `cluster_ops.py` | Global and within-cluster SMOTE, majority-class ENN and Tomek Links. |
| `gan_wrappers.py`, `pipelines.py` | Generator implementations, pipeline definitions, stage targets and row-origin tracking. |
| `classifiers.py`, `train_eval.py` | Classifier construction, fitting and scoring. |
| `run.py` | Entry point, seeds, batch selection, metric aggregation and output files. |

- Experimental splits allocate 70% to training, 10% to validation and 20% to testing. The execution seed order is 42, 456, 2566, 1011 and 9842. Standardization uses the training partition. Sampling and cleaning operate on training data. The 0.35 threshold was selected during preliminary ULB experiments and held fixed in the final runs on both datasets. The outer 10% validation partition was not used for fitting or threshold selection in the final runs. FNN training reserves a separate 10% of its shuffled pipeline input for internal monitoring and uses final-epoch weights.
- In the maintained code, the default `smotified` backend uses a guided PyTorch GAN. The discriminator sees original fraud as its real examples. The generator’s guidance pool depends on pipeline position. The `ctgan` package is an optional alternative backend.
- The augmentation target is 10% fraud before any subsequent cleaning. In `smote_gan`, SMOTE first targets 4%. In `gan_smote`, GAN first targets 6%. These two configurations differ in both sequence and intermediate target. `union` is the manuscript’s sequential SMOTE+GAN variant, with a 5% intermediate target. A `cluster_` prefix selects within-cluster SMOTE at the SMOTE stage.
- On BAF, category levels are read from the full Base file. Numeric imputation medians come from the selected subsample before splitting. The loader treats −1 as missing in four count fields. Negative `intended_balcon_amount` and `session_length_in_minutes` of −1 remain numerical inputs, although the dataset authors designate these values as missing. Some archived configurations used different subsamples. Test-index checksums use the first 2,000 positions within a subsample. The original source row IDs are unavailable, so the cross-configuration BAF ranks are descriptive. Augmentation operates on the numerical representation, and synthetic one-hot values can be fractional or violate category constraints. [BAF datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)

## Citation and data sources

When citing this work, identify the manuscript and the repository commit used to obtain the code or results. [Project repository](https://github.com/basantnabil-hub/fraud-detection-pipeline-ordering)

> Nabil, B.; Hassanein, E.E.; Mohy, N.N. A Pipeline Ordering Framework for Hybrid Data Augmentation and Cleaning in Fraud Detection Under Extreme Class Imbalance. Unpublished manuscript. Code and results: commit `4ec828c507ab9d5ba596087877c9d8154d9ec191`.

The ULB and BAF datasets are not redistributed in this archive. Obtain them from the original sources listed above and follow their respective terms of use.
