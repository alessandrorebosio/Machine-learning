import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import utilities

# Configuration and Data Loading
train_path = "DBs/PenDigits/pendigits_tr.txt"
test_path = "DBs/PenDigits/pendigits_te.txt"
result_path = "pred_sol2.txt"

feature_count = 16

X_tr, y_tr = utilities.load_labeled_dataset_from_txt(train_path, feature_count)
test_x = utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)
X_tr, y_tr, test_x = np.asarray(X_tr), np.asarray(y_tr), np.asarray(test_x)


# Solution 2: raw PCA-SVC with inner Grid Search
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gammas_raw = [float.fromhex(h) for h in (
    "0x1.0p-14",  # 6.10e-05
    "0x1.4p-14",  # 7.63e-05
    "0x1.8p-14",  # 9.16e-05
    "0x1.cp-14",  # 1.07e-04
    "0x1.0p-13",  # 1.22e-04
    "0x1.4p-13",  # 1.53e-04
    "0x1.8p-13",  # 1.83e-04
    "0x1.cp-13",  # 2.14e-04
    "0x1.0p-12",  # 2.44e-04
)]

solution2_search = GridSearchCV(
    Pipeline([
        ("pca", PCA(random_state=42)),
        ("svc", SVC(kernel="rbf")),
    ]),
    param_grid={
        "pca__n_components": list(range(10, 17)),
        "svc__C": [1.5, 2.0, 2.8, 3.5, 5.0, 7.0],
        "svc__gamma": gammas_raw,
    },
    cv=inner,
    scoring="accuracy",
    n_jobs=-1,
)

solution2_search.fit(X_tr, y_tr)
predictions_solution2 = solution2_search.predict(test_x)

with open(result_path, "w") as f:
    for prediction in predictions_solution2:
        f.write(f"{int(prediction)}\n")

print("Solution 2 selected parameters:", solution2_search.best_params_)
print(f"Solution 2 cross-validation accuracy: {solution2_search.best_score_ * 100:.2f}%")
print(f"Solution 2 predictions successfully saved to {result_path}")
