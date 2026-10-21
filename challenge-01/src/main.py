import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import utilities

# Configuration and Data Loading
train_path = 'DBs/PenDigits/pendigits_tr.txt'
test_path = 'DBs/PenDigits/pendigits_te.txt'
result_path = 'Es3Predictions.txt'

feature_count = 16

# Load training data (features and labels) and testing data (features only)
X_tr, y_tr = utilities.load_labeled_dataset_from_txt(train_path, feature_count)
test_x = utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)
X_tr, y_tr, test_x = np.asarray(X_tr), np.asarray(y_tr), np.asarray(test_x)

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)

GAMMAS_RAW = [float.fromhex(h) for h in (
    '0x1.0p-14',  # 6.10e-05
    '0x1.4p-14',  # 7.63e-05
    '0x1.8p-14',  # 9.16e-05
    '0x1.cp-14',  # 1.07e-04
    '0x1.0p-13',  # 1.22e-04
    '0x1.4p-13',  # 1.53e-04
    '0x1.8p-13',  # 1.83e-04
    '0x1.cp-13',  # 2.14e-04
    '0x1.0p-12',  # 2.44e-04
)]

raw_search = GridSearchCV(
    Pipeline([
        ('pca', PCA(random_state=42)),
        ('svc', SVC(kernel='rbf'))
    ]),
    param_grid={
        'pca__n_components': list(range(10, 17)),
        'svc__C': [1.5, 2.0, 2.8, 3.5, 5.0, 7.0],
        'svc__gamma': GAMMAS_RAW
    },
    cv=inner,
    n_jobs=-1
)

scaled_search = GridSearchCV(
    Pipeline([
        ('sc', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('svc', SVC(kernel='rbf'))
    ]),
    param_grid={
        'pca__n_components': [12, 14, 15, 16],
        'svc__C': list(np.geomspace(0.5, 20, 8)),
        'svc__gamma': list(np.geomspace(0.01, 0.3, 8))
    },
    cv=inner,
    n_jobs=-1
)

print(f'Salvataggio delle prediction riuscito')