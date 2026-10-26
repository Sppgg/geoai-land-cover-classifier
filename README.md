# GeoAI Land Cover Classifier

**Satellite image classification using Python, Machine Learning and Sentinel-2 imagery.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-2563EB?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Overview

**GeoAI Land Cover Classifier** is an applied machine learning project exploring the classification of satellite imagery into different land-cover and land-use categories.

Using the **EuroSAT RGB dataset**, derived from Sentinel-2 satellite imagery, the project investigates how image representations, preprocessing techniques and model regularization affect classification performance.

The work follows an experimental approach: exploring the data, establishing a baseline, evaluating model behavior, analyzing errors and comparing alternative feature representations.

The objective is not simply to maximize accuracy, but to understand **what models learn, where they fail and how to improve their generalization**.

## Project Objectives

- Explore and understand a real satellite imagery dataset.
- Build reproducible training, validation and testing workflows.
- Implement baseline classification models using Python and scikit-learn.
- Compare raw RGB pixel features with Histogram of Oriented Gradients (HOG).
- Investigate the effects of feature scaling and regularization.
- Evaluate models using accuracy, macro-F1 and confusion matrices.
- Document experimental decisions, limitations and future improvements.

## Dataset

The project uses [EuroSAT](https://github.com/phelber/EuroSAT), a publicly available dataset based on Sentinel-2 satellite imagery.

| Property | Description |
|---|---|
| Dataset | EuroSAT RGB |
| Source | Sentinel-2 |
| Images | 27,000 |
| Categories | 10 |
| Image resolution | 64 × 64 pixels |
| Channels | RGB |
| Task | Multiclass image classification |

### Land-Cover Categories

The dataset contains the following classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

The dataset is moderately imbalanced, with between 2,000 and 3,000 images per category.

### Dataset Split

The complete dataset was divided using stratified sampling:

| Subset | Images | Proportion |
|---|---:|---:|
| Training | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Test | 4,050 | 15% |
| **Total** | **27,000** | **100%** |

Initial experiments use a stratified subset of 2,000 training images and 1,000 validation images to reduce computational costs and accelerate iteration.

The test set remains reserved for final evaluation.

**Important:** The split is stratified by class but not geographically independent. Consequently, validation results should not be interpreted as evidence of generalization to entirely new geographic regions.

## Methodology

### 1. Exploratory Data Analysis

The initial exploration includes:

- Inspecting image dimensions and RGB channels.
- Visualizing examples from all ten categories.
- Examining class distributions.
- Verifying stratified data splitting.
- Identifying potential sources of class confusion.

### 2. RGB Baseline

The first approach converts each RGB image into a normalized feature vector.

Each image contains:

`64 × 64 × 3 = 12,288 features`

A linear classifier is trained using `SGDClassifier` with logistic loss.

This establishes an initial performance reference while highlighting the limitations of directly using flattened pixel values.

### 3. Feature Scaling and Regularization

Further experiments introduce:

- `StandardScaler` for feature standardization.
- `Pipeline` to prevent preprocessing leakage.
- Different `alpha` values to investigate regularization strength.
- Model selection using validation macro-F1.

### 4. HOG Feature Extraction

A second approach extracts **Histogram of Oriented Gradients (HOG)** descriptors.

HOG represents local image gradients and edge orientations, producing a more compact feature representation.

| Representation | Features per Image |
|---|---:|
| Flattened RGB | 12,288 |
| HOG | 1,764 |

This corresponds to approximately **85.6% fewer input features**.

The HOG configuration uses:

- 9 gradient orientations.
- 8 × 8 pixels per cell.
- 2 × 2 cells per block.
- Grayscale image input.

HOG captures local gradient structure but discards color information in this implementation.

## Experimental Results

The following results were obtained using the same stratified validation sample of 1,000 images.

All models use `SGDClassifier` with logistic loss.

| Representation | Scaling | Alpha | Validation Accuracy | Validation Macro-F1 |
|---|---|---:|---:|---:|
| RGB baseline | No | 0.0001 | 28.1% | 0.2615 |
| RGB | StandardScaler | 0.001 | 41.0% | 0.3921 |
| HOG | StandardScaler | 0.001 | 47.3% | 0.4625 |
| **HOG** | **StandardScaler** | **0.01** | **49.0%** | **0.4732** |
| HOG | StandardScaler | 0.1 | 50.3% | 0.4710 |
| HOG | StandardScaler | 1.0 | 47.5% | 0.4024 |

### Best Experimental Configuration

The highest observed validation macro-F1 was obtained with:

- **Feature representation:** HOG
- **Classifier:** SGDClassifier
- **Feature scaling:** StandardScaler
- **Regularization:** `alpha=0.01`
- **Validation accuracy:** 49.0%
- **Validation macro-F1:** 0.4732

Compared with the initial RGB baseline, this configuration improved validation accuracy by **20.9 percentage points** and macro-F1 by approximately **0.212**.

These are exploratory validation results, not final test-set metrics.

### Generalization and Overfitting

The experiments also revealed considerable differences between training and validation performance.

For example, the HOG classifier with `alpha=0.001` achieved:

| Metric | Training | Validation |
|---|---:|---:|
| Accuracy | 94.1% | 47.3% |
| Macro-F1 | 0.9386 | 0.4625 |

This large generalization gap suggests substantial overfitting.

Increasing regularization reduced the gap in several configurations, but stronger regularization did not always improve macro-F1.

These findings reinforce the importance of evaluating unseen data rather than relying exclusively on training accuracy.

## Key Findings

**Feature representation matters.**

HOG outperformed flattened RGB features in the evaluated configurations, suggesting that local gradient information can provide useful discriminatory features for this task.

**More regularization is not always better.**

Increasing `alpha` improved generalization up to a point, but excessive regularization reduced macro-F1.

**Accuracy alone is insufficient.**

Some categories exhibited very different precision and recall values. Macro-F1 and confusion matrices helped identify weaknesses that overall accuracy could conceal.

**High training accuracy does not guarantee generalization.**

Large train-validation performance gaps highlighted the limitations of the current models and the need for further experimentation.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | scikit-learn |
| Image Processing | Pillow, scikit-image |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Dataset Loading | torchvision |
| Experimentation | Google Colab, Jupyter Notebook |
| Version Control | Git, GitHub |

## Repository Structure

```text
geoai-land-cover-classifier/
├── notebooks/
│   ├── 00_original_experiments.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── results/
│   └── experiments.csv
├── .gitignore
├── LICENSE
└── README.md
```

## Running the Notebooks

The project is organized into four stages:

1. **Data Exploration:** Inspect the EuroSAT dataset, visualize satellite imagery and analyze class distributions.
2. **Preprocessing:** Generate reproducible data splits, prepare RGB features and extract HOG descriptors.
3. **Model Training:** Train linear classifiers and compare preprocessing and regularization strategies.
4. **Model Evaluation:** Review recorded validation results and analyze model performance and limitations.

Each notebook is designed to run independently in Google Colab. Some data preparation operations are repeated to avoid dependencies on variables stored in other notebook sessions.

To reproduce an experiment, open the corresponding notebook in Google Colab and select **Runtime → Run all**.

Results reported in this repository are exploratory validation results. The held-out test set has not yet been used for final evaluation.


**Results**

[experiments.csv](results/experiments.csv) contains a summary of the evaluated model configurations and validation metrics.

## Reproducing the Experiments

The experiments were developed using **Google Colab**.

The simplest way to explore or reproduce them is to open the notebook:

[Open the notebook in Google Colab](https://colab.research.google.com/github/Sppgg/geoai-land-cover-classifier/blob/main/notebooks/01_data_exploration.ipynb)

The notebook includes the code used to download the EuroSAT dataset through `torchvision`, prepare the data and execute the experiments.

A Python environment with the required dependencies is necessary. The original experiments used Python 3.13 and scikit-learn 1.6.1.

**Reproducibility note:** The notebook relies on data downloaded at runtime. The repository does not contain the complete EuroSAT dataset. Dependency pinning and automated reproduction checks are planned improvements.

## Limitations

This project is currently experimental and has several limitations:

- Initial models use only 2,000 training images.
- Validation metrics are calculated using a subset of 1,000 images.
- The split is not geographically independent.
- The RGB baseline does not explicitly model spatial relationships.
- The grayscale HOG implementation discards color information.
- Linear classifiers may struggle to represent complex image patterns.
- Repeated comparisons on the same validation subset can influence model selection.
- Final evaluation on the reserved test set has not yet been completed.

The reported results should not be interpreted as production-ready classification performance.

## Roadmap

Future development is planned around improving predictive performance, reproducibility and engineering quality.

- [x] Explore and visualize EuroSAT RGB.
- [x] Analyze class distribution.
- [x] Create stratified train, validation and test splits.
- [x] Train an RGB baseline classifier.
- [x] Evaluate accuracy, macro-F1 and class-level errors.
- [x] Implement HOG feature extraction.
- [x] Compare feature representations and regularization values.
- [x] Export an experiment summary.
- [ ] Evaluate alternative feature representations and models.
- [ ] Investigate larger training subsets.
- [ ] Improve experiment tracking and reproducibility.
- [ ] Evaluate the selected model on the reserved test set.
- [ ] Develop an interactive inference application.
- [ ] Add automated tests and continuous integration.

## Data Attribution

This project uses the **EuroSAT dataset**.

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

[EuroSAT repository](https://github.com/phelber/EuroSAT)

Dataset ownership, licensing and attribution remain with their respective authors and providers.

## Author

Developed by [Sara P.](https://github.com/Sppgg) as an applied machine learning and geospatial AI project.

The project documents a practical learning and engineering process: from data exploration and baseline development to model evaluation, experimentation and future deployment.

---

**Status:** Active development — experimental results, not production-ready.
