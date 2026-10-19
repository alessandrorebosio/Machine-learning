from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np

import utilities

# Configuration and Data Loading
train_path = 'DBs/PenDigits/pendigits_tr.txt'
test_path = 'DBs/PenDigits/pendigits_te.txt'
result_path = 'Es3Predictions.txt'

feature_count = 16

# Load training data (features and labels) and testing data (features only)
X_tr, y_tr = utilities.load_labeled_dataset_from_txt(train_path, feature_count)
test_x = utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)

# Pipeline Initialization
# Note on Model Selection: SVC vs. KNN
# SVC (RBF kernel) is preferred over KNN because it guarantees faster inference 
# and lower memory usage by relying only on a subset of support vectors, whereas 
# KNN computes distances against the entire training set. Additionally, SVC handles 
# complex, non-linear boundaries better for handwriting recognition tasks.
pipe = Pipeline([
    ('pca', PCA(random_state=42)),
    ('svc', SVC())
])

# Define the hyperparameter grid for optimization.
param_grid = [{
    'pca__n_components': [10, 12, 14, 16],
    'svc__kernel': ['rbf'],
    'svc__C': [1, 3, 10, 30],
    'svc__gamma': [5e-5, 1e-4, 1.5e-4, 3e-4],
}]

# Hyperparameter Tuning via Grid Search
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs.fit(X_tr, y_tr)

# Extract and display the top 10 configurations based on mean cross-validation accuracy
# r = gs.cv_results_
# for i in np.argsort(-r['mean_test_score'])[:10]:
#     print(f"{r['mean_test_score'][i] * 100:.2f}% (± {r['std_test_score'][i] * 100:.2f}%)  {r['params'][i]}")
# print('Best parameters:', gs.best_params_, f'- CV: {gs.best_score_ * 100:.2f}%')

# Final Prediction and Export
predictions = gs.predict(test_x)

with open(result_path, 'w') as f:
    for prediction in predictions:
        f.write(str(int(prediction)) + '\n')
print('Salvataggio delle prediction riuscito')
