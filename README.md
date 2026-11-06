# Fraud Detection Pipeline Ordering

Research code and archived results for studying how the sequence of data augmentation and cleaning affects fraud detection under class imbalance. The framework combines SMOTE, a guided generative adversarial network (GAN), Tomek Links, Edited Nearest Neighbours (ENN), and K-means-based SMOTE.

The paper configuration set contains **35 pipelines, including the baseline**, evaluated with **XGBoost, Random Forest, and a feed-forward neural network (FNN)** on the **ULB Credit Card Fraud** and **Bank Account Fraud (BAF)** datasets. Five repetitions per combination give **1,050 paper runs**.

## Repository contents

Place this README at the repository root, alongside `Code/` and `Results/`. Keep the nine Python files together under their original names.

| File | Purpose |
| --- | --- |
| [`Code/config.py`](Code/config.py) | Dataset paths, sampling targets, model settings, and decision threshold. |
| [`Code/data_prep.py`](Code/data_prep.py) | Dataset loading, BAF encoding, stratified splitting, and feature scaling. |
| [`Code/sampling_ops.py`](Code/sampling_ops.py) | SMOTE generation, Tomek Links, and ENN cleaning. |
| [`Code/cluster_ops.py`](Code/cluster_ops.py) | K-means-based SMOTE operations. |
| [`Code/gan_wrappers.py`](Code/gan_wrappers.py) | Guided GAN implementation and an optional CTGAN wrapper. |
| [`Code/pipelines.py`](Code/pipelines.py) | The 35 pipeline definitions, row-origin labels, and cleaning counts. |
| [`Code/classifiers.py`](Code/classifiers.py) | Classifier construction; the paper runner selects `xgb`, `rf`, or `nn`. |
| [`Code/train_eval.py`](Code/train_eval.py) | Model fitting, prediction, evaluation metrics, and training history. |
| [`Code/run.py`](Code/run.py) | Interactive and command-line execution, five repetitions, saved runs, and reports. |

The `Results/` directory contains the paper CSV tables and six ZIP archives. The original input datasets are downloaded separately.

## Datasets

| Dataset | Download and documentation | Required file | Target column | Default selection |
| --- | --- | --- | --- | --- |
| ULB Credit Card Fraud | [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` | `Class` | Full dataset: 284,807 transactions, including 492 frauds. |
| Bank Account Fraud | [Feedzai project and download link](https://github.com/feedzai/bank-account-fraud) | `Base.csv` | `fraud_bool` | Stratified subsample of 250,000 rows from the Base variant. |

Use the original ULB dataset and the **Base** variant of BAF. For Colab, the default locations are `/content/creditcard.csv` and `/content/Base.csv`. A different location can be supplied with `--data-path`.

BAF loading handles the configured categorical columns, missing-value sentinels, and feature exclusions in `data_prep.py`. Feature scaling is fitted on the training partition. Sampling and cleaning operate on the training data; validation and test data retain their original class distributions.

## Installation

Use a separate Python environment for local execution. The setup below targets Python **3.11-3.13** and uses TensorFlow 2.21 with Keras 3. It installs dependencies for a new run; the archived experiments do not include a complete software-version manifest.

From the repository root on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost matplotlib "tensorflow==2.21.0" "keras>=3" torch
```

On Windows, activate the environment with `.venv\Scripts\activate` before running the installation commands. For platform-specific builds, consult the [TensorFlow installation guide](https://www.tensorflow.org/install/pip), [Keras setup guide](https://keras.io/getting_started/), and [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Both TensorFlow and PyTorch are needed by the current module imports, including when selecting a tree classifier. The default GAN backend is `CFG.gan_backend = "smotified"`, implemented in PyTorch. **CTGAN is optional** and is needed only if the backend is deliberately changed to `"ctgan"`.

The runner does not install or uninstall packages. When changing packages in a notebook, restart the session before importing the experiment modules.

## Run in Google Colab

1. Open a notebook. A GPU runtime can accelerate GAN training.
2. Upload all nine files from `Code/` directly to `/content/`, keeping their filenames. Upload the selected dataset there as well.
3. Install the dependencies in a separate cell:

```python
%pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost matplotlib "tensorflow==2.21.0" "keras>=3" torch
```

4. Restart the session if packages changed, then check imports and pipeline registration:

```python
%cd /content
import tensorflow as tf
import keras
import torch
from pipelines import get_all_pipelines

for seed in (42, 456, 42):
    tf.keras.utils.set_random_seed(seed)

assert len(get_all_pipelines()) == 35
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA available:", torch.cuda.is_available())
print("Imports, repeated seed calls, and pipeline registration passed.")
```

5. Start with one pipeline and one classifier:

```python
%run run.py --dataset ulb --data-path /content/creditcard.csv --classifier rf --pipelines baseline --output-name baseline_rf_check
```

This performs five baseline repetitions using the configured model settings. A successful batch reports `5/5` successful runs. It checks the baseline path; GAN and cleaning pipelines are exercised when selected separately.

To use the menu instead:

```python
%run run.py
```

The menu displays the pipeline names under each group. Select **5. Custom** to enter names separated by commas. An unknown pipeline name prompts another attempt; enter `exit` to cancel. Selecting **0. Exit** at the batch menu also cancels execution.

The guided GAN selects CUDA when PyTorch detects an available GPU, otherwise CPU. TensorFlow and PyTorch detect GPU support separately. GAN pipelines take longer than the baseline because the default GAN trains for 100 epochs with 200 steps per epoch.

## Command-line examples

Run these commands from the repository root, replacing the dataset paths with your local paths.

**Selected pipelines with Random Forest:**

```bash
python Code/run.py --dataset ulb --data-path /path/to/creditcard.csv --classifier rf --pipelines baseline smote_only smote_enn smote_tomek --output-name ulb_rf_check
```

**The combined SMOTE+GAN configuration with XGBoost:**

```bash
python Code/run.py --dataset ulb --data-path /path/to/creditcard.csv --classifier xgb --pipelines union --output-name ulb_union_xgb
```

**The ENN group with the FNN on BAF:**

```bash
python Code/run.py --dataset baf --data-path /path/to/Base.csv --classifier nn --batch 3 --output-name baf_enn_fnn
```

**All paper pipelines and all three classifiers on one dataset:**

```bash
python Code/run.py --dataset ulb --data-path /path/to/creditcard.csv --classifier all --pipelines all --output-name ulb_full
```

The final command schedules 525 runs. Repeat it with `--dataset baf`, the BAF path, and a new output name to run the second dataset. The complete two-dataset study contains 1,050 runs and can require substantial execution time.

Command-line pipeline names are separated by **spaces**. Use either `--batch` or `--pipelines`. Classifier identifiers are `xgb`, `rf`, `nn`, and `all`; `nn` means FNN. Run `python Code/run.py --help` to view the available options.

## Pipeline groups

Names generally describe the sequence of operations from left to right. The `cluster_` prefix selects K-means-based SMOTE at the SMOTE stage. The human-readable labels are listed in [`Results/paper_pipelines.csv`](Results/paper_pipelines.csv).

### 1. Basics: 6 pipelines

- `baseline`
- `smote_only`
- `gan_only`
- `union`
- `smote_gan`
- `gan_smote`

`union` is the paper's **SMOTE+GAN** configuration: SMOTE first targets a 5% fraud proportion, then GAN generation fills the remaining gap to the overall 10% target. Use `union` when selecting this configuration.

### 2. Tomek cleaning: 9 pipelines

- `tomek_smote`
- `tomek_gan`
- `smote_tomek`
- `gan_tomek`
- `smote_gan_tomek`
- `gan_smote_tomek`
- `smote_tomek_gan`
- `tomek_smote_gan`
- `tomek_gan_smote`

### 3. ENN cleaning: 10 pipelines

- `enn_smote`
- `enn_gan`
- `smote_enn`
- `gan_enn`
- `smote_gan_enn`
- `gan_smote_enn`
- `gan_enn_smote`
- `smote_enn_gan`
- `enn_smote_gan`
- `enn_gan_smote`

### 4. Clusters: 10 pipelines

- `cluster_smote_gan`
- `cluster_gan_smote`
- `cluster_smote_tomek_gan`
- `cluster_tomek_smote_gan`
- `cluster_smote_enn_gan`
- `cluster_enn_smote_gan`
- `cluster_enn_gan_smote`
- `cluster_smote_gan_enn`
- `cluster_gan_smote_enn`
- `cluster_gan_enn_smote`

## Experiment settings

Most settings are defined in `Code/config.py`. Repetition count and the seed sequence are defined in `Code/run.py`.

| Setting | Current default |
| --- | --- |
| Repetitions | 5 per pipeline and classifier. |
| Seeds, in execution order | `42`, `456`, `2566`, `1011`, `9842`. |
| Data partitions | Stratified 70% training, 10% validation, 20% test. |
| Overall augmentation target | 10% fraud in the augmented training set; the baseline keeps the original data. Cleaning after augmentation can change the final proportion. |
| Sequential intermediate targets | SMOTE first targets 4% in `smote_gan`; GAN first targets 6% in `gan_smote`. The second stage targets 10%. |
| SMOTE neighbours | `CFG.smote_k = 5`; reduced when necessary for small minority groups. |
| K-means clusters | `CFG.kmeans_k = 5`, a separate parameter from SMOTE neighbours. |
| Cleaning | Tomek Links and ENN target the majority class. ENN uses 3 neighbours with `kind_sel="all"`. |
| GAN | Guided PyTorch GAN; 100 epochs, batch size 256, learning rate 0.0002, noise dimension 64, hidden width 128. |
| XGBoost | 400 trees, maximum depth 6, learning rate 0.05, row and column subsampling 0.9. |
| Random Forest | 300 trees and `class_weight="balanced_subsample"`. |
| FNN | Hidden layers of 256 and 128 units; 100 epochs, batch size 128, learning rate 0.0001. |
| Decision threshold | `CFG.default_threshold = 0.35` for the reported classification metrics. |

The FNN also reserves 10% of its shuffled training input for its internal training-validation history. The GAN discriminator uses real fraud rows; the generator receives a guidance pool that can include SMOTE-generated fraud rows, depending on the pipeline.

## Output folders and resuming runs

In Colab, `run.py` attempts to mount Google Drive and writes to:

```text
/content/drive/MyDrive/Colab Notebooks/thesis_experiments/<dataset>_<output-name>/
```

The dataset prefix is `creditcard` for ULB and `baf` for BAF. For example, the Colab baseline command above creates `creditcard_baseline_rf_check` under this directory.

If Drive is unavailable, Colab output goes to `/content/thesis_experiments/`. Outside Colab, output goes to `thesis_experiments/` under the current working directory. Local Colab files need to be downloaded before the runtime is discarded.

Each experiment folder contains:

| Subdirectory | Contents |
| --- | --- |
| `individual_runs/` | Per-run JSON results and additional prediction outputs where produced. |
| `statistics/` | Summaries across successful repetitions. |
| `reports/` | Comparison tables and text/LaTeX reports. |
| `plots/` | Generated comparison plots. |
| `console_outputs/` | Saved run logs. |

Successful saved runs are reused when the same output folder is selected. **Use a new output name whenever data, settings, or code change.** Resume checks identify saved pipeline/classifier runs but do not compare the full experiment configuration. The interactive menu can append a timestamp to create a separate folder.

## Included results

| File or folder | Contents |
| --- | --- |
| [`Results/CC Dataset/CC_per_seed_long.csv`](Results/CC%20Dataset/CC_per_seed_long.csv) | 525 ULB runs: 35 pipelines x 3 classifiers x 5 seeds. Dataset identifier: `CC`. |
| [`Results/BAF Dataset/BAF_per_seed_long.csv`](Results/BAF%20Dataset/BAF_per_seed_long.csv) | 525 BAF runs with the same configuration coverage. Dataset identifier: `BAF`. |
| [`Results/paper_pipelines.csv`](Results/paper_pipelines.csv) | Mapping between the 35 pipeline identifiers and paper labels. |
| [`Results/cleaning_counts_per_seed.csv`](Results/cleaning_counts_per_seed.csv) | 1,050 records of removed legitimate and fraud rows, with source archive and record paths. |
| [`Results/cleaning_counts.csv`](Results/cleaning_counts.csv) | Mean removal counts and sample standard deviations for the 35 paper configurations. |
| [`Results/CC Dataset/`](Results/CC%20Dataset/) | Three ZIP archives for ULB: Random Forest (`RF`), FNN (`nn`), and XGBoost (`xgboost`). |
| [`Results/BAF Dataset/`](Results/BAF%20Dataset/) | Three corresponding ZIP archives for BAF. |

Each historical ZIP contains 190 successful run JSON records: 38 configurations x 5 seeds for one dataset/classifier combination. These archives retain three additional configurations: `smote_plus_gan_5050`, `cluster_enn_gan`, and `cluster_gan_enn`. They are excluded from the 35-pipeline paper set and from the current runner. Across the six ZIPs there are 1,140 historical records; the two paper result CSVs select 1,050 of them.

Use the per-seed paper CSVs and `paper_pipelines.csv` when reconstructing paper tables. The historical ZIP reports can include the additional configurations.

### Metric names and standard deviations

- `Precision`, `Recall`, `F1`, and `MCC` are evaluated at the stored `used_threshold`, which is 0.35 in all 1,050 paper records.
- `PR_AUC` in the CSVs corresponds to `AUC_PR` in the run JSON files and is computed with scikit-learn's **average precision** function. `ROC_AUC` corresponds to `AUC_ROC`.
- `TP`, `FP`, `TN`, and `FN` are confusion-matrix counts. Scores are stored on a 0-1 scale, except MCC, which can range from -1 to 1.
- The maintained runner reports **sample standard deviation (`ddof=1`)**. Older summaries inside the ZIPs use population standard deviation (`ddof=0`). Recalculate from the per-seed values when producing a consistent mean-and-SD table.
- The saved threshold sweeps are additional analyses. They do not change the 0.35 threshold used for the main reported metrics.

New runs count removals at each cleaning step using the real/SMOTE/GAN row-origin labels and sum them across the pipeline. The supplied cleaning CSVs contain original-row removal counts; synthetic-removal fields are available in individual run records.

### Recalculate summaries without training

Run the following Python code from the repository root. It reads the supplied CSVs and prints mean values and sample standard deviations; no dataset download or model training is needed.

```python
from pathlib import Path
import pandas as pd

root = Path(".")
paper = pd.read_csv(root / "Results/paper_pipelines.csv")
results = pd.concat([
    pd.read_csv(root / "Results/CC Dataset/CC_per_seed_long.csv"),
    pd.read_csv(root / "Results/BAF Dataset/BAF_per_seed_long.csv"),
], ignore_index=True)

keys = ["dataset", "classifier", "pipeline"]
assert len(results) == 1050
assert not results.duplicated(keys + ["seed"]).any()
assert set(results["pipeline"]) == set(paper["pipeline"])
assert results.groupby(keys)["seed"].nunique().eq(5).all()
assert results["used_threshold"].eq(0.35).all()

metrics = ["Precision", "Recall", "F1", "PR_AUC", "ROC_AUC", "MCC"]
summary = results.groupby(keys)[metrics].agg(["mean", "std"])
print(summary.round(4).to_string())  # pandas std uses ddof=1
```

For new experiments, retain the installed package versions, hardware information, settings, and code commit alongside the outputs. The archived per-run values provide the reference for the supplied paper summaries; new training runs can vary with software versions and hardware.
