# Challenge-01

## Instructions

1. **Task**: Train a multiclass classifier (`0`-`9`) on `DBs/PenDigits/pendigits_tr.txt` and predict the unlabeled samples in `DBs/PenDigits/pendigits_te.txt`.
2. **Target**: Accuracy $\ge 85\%$ on the test set.

## Rules & Constraints
- **Allowed Models**: Only `sklearn.neighbors.KNeighborsClassifier` or `sklearn.svm.SVC`.
- **Forbidden**: Ensembles (no voting, bagging, boosting), external data, data augmentation, adding or changing features.
- **Allowed Preprocessing**: Scaling (`StandardScaler`, `MinMaxScaler`) and `PCA`.
- **Data Leakage Rule**: Fit scalers/PCA strictly on the training set using `.fit_transform()`. Use `.transform()` on the test set.

## Datasets
- Train: `dataset/PenDigits/pendigits_tr.txt` (442 samples, 16 features, 1 target label).
- Test: `dataset/PenDigits/pendigits_te.txt` (5,000 unlabeled samples, 16 features).

## Output File Requirements
- **Predictions**: `Es3Predictions.txt`
  - Must contain exactly 5,000 lines.
  - One predicted integer (`0` to `9`) per line.
  - No header, no index, no commas.
- **Submission ZIP**: `<filename>.zip` with the following structure:
  ```text
  <filename>.zip
  ├── Codice/
  │   └── <all code/scripts used for training and inference>
  └── Es3Predictions.txt