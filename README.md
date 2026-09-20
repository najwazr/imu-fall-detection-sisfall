![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-blue)
![Dataset](https://img.shields.io/badge/Dataset-SisFall-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

# IMU-Based Fall Detection Using Machine Learning

A machine learning pipeline for detecting falls from wearable Inertial Measurement Unit (IMU) data. The pipeline combines **Madgwick filter** sensor fusion (for body orientation estimation) with statistical/kinematic feature extraction and seven classical ML models, evaluated on the public **SisFall** dataset using a leakage-free, **subject-wise cross-validation** scheme.

## Highlights

- **116,497 windows** extracted from 3,537 recordings across 23 subjects (SisFall young-adult group)
- **24 features** across six groups: SVM statistics, per-axis acceleration, peak detection, orientation (roll/pitch from Madgwick filter), gyroscope, and advanced signal descriptors
- **Subject-wise 10-fold cross-validation** — windows from the same subject never appear in both train and test sets, preventing subject leakage and giving a realistic estimate of generalization to unseen individuals
- **Best model: Random Forest (√p = 5 predictors)** — 83.25% accuracy, 77.01% sensitivity, AUC 0.8840
- **Ablation study** quantifying the contribution of Madgwick-derived orientation features (up to +5.12 percentage points in accuracy)

## Why Subject-Wise Cross-Validation?

A naive window-based k-fold split can place windows from the *same subject* in both the training and test sets. Since consecutive windows overlap and share biomechanical traits of that individual, this leaks subject-specific information into the test set and produces overly optimistic performance estimates. This pipeline partitions folds **by subject**, not by window, to avoid that pitfall — see `training.m` for the partition and an automatic leakage-check that verifies no subject appears on both sides of any fold.


## Repository Structure

```
.
├── feature_extraction_final.m   # Preprocessing: unit conversion, Madgwick filter,
│                                 # sliding window segmentation, 24-feature extraction
├── before_after_madgwick.m      # Visualization: raw gyro integration vs. Madgwick
│                                 # filter orientation estimates (drift comparison)
├── training.m                   # Subject-wise 10-fold CV training of 7 models
│                                 # (DT, 2x RF, 2x KNN, 2x SVM) with leakage verification
├── ablation.m                   # Ablation study: retrains all 7 models without the
│                                 # 4 Madgwick-derived orientation features
├── generate_figures_report.m    # Generates confusion matrices (count + row-normalized)
│                                 # and ROC curves for all 7 models from the last fold
└── requirements.md              # Environment and toolbox requirements
```

## How to Run

Scripts are designed to run independently in this order:

1. **`feature_extraction_final.m`** — extracts features from the raw SisFall dataset.
   Produces `features_dataset_v3.mat` / `.csv`.
2. **`before_after_madgwick.m`** — visualizes orientation drift
   for a single recording. Not required for the training pipeline.
3. **`training.m`** — trains and evaluates all 7 models with subject-wise 10-fold CV.
   Produces `hasil_subjectwise_cv_summary.csv`, per-fold detail CSVs, and
   `trained_models_lastfold.mat` (used by step 5).
4. **`ablation.m`** — repeats training without the 4 orientation features, for
   comparison against step 3's results. Produces `hasil_ablation_20fitur_summary.csv`.
5. **`generate_figures_report.m`** — run after step 3. Generates confusion matrix and
   ROC curve figures for all 7 models into `figures_report/`.

## Results Summary

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| **RF (√p = 5)** | **83.25%** | **77.01%** | 87.92% | **0.8840** |
| Bagged Trees | 81.40% | 76.79% | 84.86% | 0.8377 |
| SVM Quadratic | 81.89% | 72.30% | 89.07% | 0.8745 |
| Weighted KNN | 80.58% | 73.47% | 85.90% | 0.8513 |
| Decision Tree | 80.21% | 67.55% | 89.69% | 0.7898 |
| Fine KNN | 76.80% | 73.70% | 79.12% | 0.7430 |

*SVM Cubic is excluded from this summary due to solver non-convergence at this dataset scale (documented as a finding).*

## Ablation Study: Contribution of Madgwick Filter Orientation Features

| Model | 24 Features | 20 Features (no orientation) | Δ (points) |
|---|---|---|---|
| **RF (√p = 5)** | **83.25%** | **80.28%** | **+2.97** |
| SVM Quadratic | 81.89% | 76.77% | +5.12 |
| Bagged Trees | 81.40% | 78.70% | +2.70 |
| Weighted KNN | 80.58% | 78.31% | +2.27 |
| Decision Tree | 80.21% | 79.92% | +0.29 |
| Fine KNN | 76.80% | 73.96% | +2.84 |

Removing the four Madgwick-derived orientation features (`Roll_Std`, `Pitch_Std`, `Roll_Range`, `Pitch_Range`) consistently degrades performance across all well-converged models, confirming their contribution to classification accuracy.

## Dataset

This project uses the public **SisFall** dataset (young adult group, SA01–SA23), developed by Sucerquia et al. (2017). The dataset is **not included** in this repository — download it from the [official SisFall repository](http://sistemic.udea.edu.co/en/research/projects/english-falls/) and place it as:

```
SisFall_data/
├── SA01/
├── SA02/
├── ...
└── SA23/
```

## Requirements

See [`requirements.md`](requirements.md).

## Citation

If referencing this work, please cite the dataset:

> Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A Fall and Movement Dataset. *Sensors*, 17(1), 198.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The SisFall dataset is not included and is subject to its own terms.
