import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import utilities

# Configuration and Data Loading
train_path = "DBs/PenDigits/pendigits_tr.txt"
test_path = "DBs/PenDigits/pendigits_te.txt"
result_path = "pred_sol1.txt"

feature_count = 16

X_tr, y_tr = utilities.load_labeled_dataset_from_txt(train_path, feature_count)
test_x = utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)
X_tr, y_tr, test_x = np.asarray(X_tr), np.asarray(y_tr), np.asarray(test_x)


# Solution 1: PCA and SVC with repeated cross-validation
solution1_search = GridSearchCV(
    Pipeline([
        ("pca", PCA(random_state=42)),
        ("svc", SVC(kernel="rbf")),
    ]),
    param_grid={
        "pca__n_components": [10, 12, 14, 16],
        "svc__C": [1, 3, 10, 30],
        "svc__gamma": [5e-5, 1e-4, 1.5e-4, 3e-4],
    },
    cv=RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=10,
        random_state=42,
    ),
    scoring="accuracy",
    n_jobs=-1,
)

solution1_search.fit(X_tr, y_tr)
predictions_solution1 = solution1_search.predict(test_x)

with open(result_path, "w") as f:
    for prediction in predictions_solution1:
        f.write(f"{int(prediction)}\n")

print(f"Solution 1 cross-validation accuracy: {solution1_search.best_score_ * 100:.2f}%")
print(f"Solution 1 predictions successfully saved to {result_path}")
