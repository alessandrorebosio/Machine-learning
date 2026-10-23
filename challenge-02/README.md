# Challenge-02

## Instructions

1. **Task**: Maximize the classification accuracy (cats vs dogs images) using **HOG features only**, evaluating the choices with *Cross-Validation* (`cross_val_score`, `GridSearchCV`). Explore:
   - different classifiers (e.g. k-NN, SVM, Random Forest, AdaBoost, or others);
   - hyperparameter optimization of the classifiers;
   - optimization of the HOG extraction parameters;
   - combinations of multiple classifiers.
2. **Target**: Accuracy $\ge 75\%$ on the test set (threshold to obtain 2 points in the ranking).
3. **Suggestion**: Proceed iteratively, starting from a limited set of combinations and then refining the search: `cross_val_score(...)` and `GridSearchCV(...)` can become prohibitively slow with many hyperparameter combinations and folds.

## Rules & Constraints
- **Allowed Models**: Any classifier, **except** those based on neural networks or transformers.
- **Allowed**:
  - Ensembles of classifiers, and free variation of classifier/ensemble hyperparameters.
  - Changing the image resolution.
  - **Image preprocessing** (e.g. contrast, brightness) before HOG extraction, provided it is identical for train and test and does **not** increase the number of patterns in the dataset.
  - **HOG feature preprocessing** (e.g. rescaling, standardization).
  - Varying HOG hyperparameters (orientations, pixels per block, etc.), including extracting HOG at multiple resolutions for concatenation or for training different classifiers to combine.
- **Forbidden**:
  - Any feature in addition to or in place of HOG.
  - Expanding the dataset with external cat/dog photos.
  - Data augmentation (random contrast, rotations, translations, scaling, etc.).
- **Data Leakage Rule**: Fit any preprocessing strictly on the training set using `.fit_transform()`. Use `.transform()` on the test set.

## Datasets
- Train: `<DBs/CaniGatti_ML18/TrainingSet>` (900 labeled images: 509 cats, 391 dogs; 2304 HOG features per image with the baseline extraction parameters).
- Test: `<DBs/CaniGatti_ML18/TestSet>`.
- Class balance: cats 56.6%, dogs 43.4%, so a constant "cat" predictor already scores about 56.6%.

## Output File Requirements
- **Predictions**: `Es4Predictions.txt`
  - One predicted label per line, one line per test sample.
  - No header, no index, no commas.
- **Submission ZIP**: `<filename>.zip` with the following structure:
  ```text
  <filename>.zip
  ├── Codice/
  │   └── <all code (notebooks and scripts) used for training and testing the classifier>
  └── Es4Predictions.txt
  ```
  **NB**: `Es4Predictions.txt` must be at the root of the zip, **not** inside a folder.