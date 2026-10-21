import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import utilities


# Configuration and Data Loading
train_path = "DBs/PenDigits/pendigits_tr.txt"
test_path = "DBs/PenDigits/pendigits_te.txt"
result_path = "pred_sol3.txt"

feature_count = 16

X_tr, y_tr = utilities.load_labeled_dataset_from_txt(train_path, feature_count)
test_x = utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)
X_tr, y_tr, test_x = np.asarray(X_tr), np.asarray(y_tr), np.asarray(test_x)


# Solution 3: fixed optimized raw PCA-SVC configuration
solution3_cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=10,
    random_state=42,
)

solution3_model = Pipeline([
    ("pca", PCA(n_components=16, random_state=42)),
    ("svc", SVC(kernel="rbf", C=1.5, gamma=2e-4)),
])

solution3_scores = cross_val_score(
    solution3_model,
    X_tr,
    y_tr,
    cv=solution3_cv,
    scoring="accuracy",
    n_jobs=-1,
)

solution3_model.fit(X_tr, y_tr)
predictions_solution3 = solution3_model.predict(test_x)

with open(result_path, "w") as f:
    for prediction in predictions_solution3:
        f.write(f"{int(prediction)}\n")

print(
    "Solution 3 cross-validation accuracy: "
    f"{solution3_scores.mean() * 100:.2f}%"
)
print(f"Solution 3 predictions successfully saved to {result_path}")
