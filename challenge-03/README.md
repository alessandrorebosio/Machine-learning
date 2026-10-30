# Challenge-03

## Instructions

1. **Task**: *Predict ahead* scenario. Predict the power produced by the photovoltaic plant during the **next hour** (t+1), minimizing the **RMSE** on the test set. Explore:
   - which data/features to include in the dataset (re-running loading and *preprocessing* if needed) and how to prepare it for the *predict ahead* setting;
   - different regressors and hyperparameter optimization;
   - cyclical encoding of date features (hour of day, day of year, month), see [Cyclical features encoding](https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca);
   - combinations of two or more regressors (ensembles);
   - cleaning of anomalous data.
2. **Target**: RMSE $\le 75$ on the test set (threshold to obtain 2 points in the ranking).
3. **Suggestion**: Evaluate choices with *Cross-Validation*, preferably with folds made of **ordered temporal periods** (e.g. `TimeSeriesSplit` or contiguous blocks) rather than random patterns, to avoid optimistic estimates caused by temporally adjacent samples. If you skip Cross-Validation, split the original dataset into a training set and a validation set.

## Scoring with RMSE in scikit-learn

`GridSearchCV(...)` and `cross_val_score(...)` use the regressor's `.score(...)` method by default, which returns $R^2$. To use MSE, set `scoring='neg_mean_squared_error'` (string). To recover the RMSE from the grid search results:

```python
import numpy as np

best_rmse = np.sqrt(-gscv.best_score_)
all_rmse  = np.sqrt(-gscv.cv_results_['mean_test_score'])
```

## Rules & Constraints
- **Allowed Models**: Any regressor, **except** those based on neural networks or transformers.
- **Allowed**:
  - Ensembles of regressors, and free variation of regressor/ensemble hyperparameters.
  - **Preprocessing** of the data (e.g. rescaling, standardization).
  - **Reducing** the dataset by removing anomalous data (*cleaning*).
  - **Deriving new features** from existing ones (e.g. cyclical encoding of dates).
- **Forbidden**:
  - Extending the dataset (*augmentation*) by adding new patterns or creating derived ones from existing patterns.
  - Using features from times earlier than $t$: **only the features at time $t$** may be used to predict the energy produced at time $t+1$.
  - Using the energy produced at time $t$ to estimate the one at $t+1$: the current value of `y` must **not** be included in the input vector `x` (it would also be impossible to predict on `test_x`, since `test_y` is not provided).
- **Data Leakage Rule**: Fit any preprocessing strictly on the training set using `.fit_transform()`. Use `.transform()` on the test set.

## Datasets
- Train: `<DBs/SolarPark/train>`
- Test: `<DBs/SolarPark/test>` (`test_x` only, `test_y` is not provided)

## Output File Requirements
- **Predictions**: `Es5Predictions.txt`
  - One predicted value per line, one line per test sample.
  - No header, no index, no commas.
- **Submission ZIP**: `<filename>.zip` with the following structure:
  ```text
  <filename>.zip
  ├── Codice/
  │   └── <all code (notebooks and scripts) used for training and testing the regressor>
  └── Es5Predictions.txt
  ```
  **NB**: `Es5Predictions.txt` must be at the root of the zip, **not** inside a folder.