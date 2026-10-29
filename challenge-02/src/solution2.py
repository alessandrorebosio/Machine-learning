import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
from joblib import Memory
from skimage import color, feature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

import utilities

# HOG and pipeline parameters
image_sides_solution2 = [176, 208]
orientations = 12
cell_side = 16
block_side = 3
alpha = 0.5

db_path = 'DBs/CaniGatti_ML18'
exp_path = 'Experiments'
train_filelist = 'BinaryTrainingSet.txt'
test_filelist = 'Unlabeled_BinaryTestSet.txt'
result_path = 'Es4Predictions.txt'
memory = Memory(exp_path, verbose=0)


def extract_hog_features(images, side):
    """Resize to side x side, convert to grayscale, and extract HOG features."""
    resized = utilities.resize_images(images, side, side, cache=memory)
    hog_features = []
    for img in resized:
        hog = feature.hog(color.rgb2gray(img),
                          orientations=orientations,
                          pixels_per_cell=(cell_side, cell_side),
                          cells_per_block=(block_side, block_side),
                          block_norm='L2-Hys')
        hog_features.append(hog)
    return np.array(hog_features)


def power_transform(X, alpha=0.5):
    """Element-wise x**alpha (HOG L2-Hys is >= 0; clipping guards against -0.0)."""
    return np.power(np.clip(X, 0, None), alpha)


class PrecomputedRBFSVC(SVC):
    """SVC(kernel='rbf', gamma='scale') with the kernel matrix computed once by numpy: same predictions, much faster."""

    def fit(self, X, y):
        self.X_fit_, self.gamma_ = X, 1 / (X.shape[1] * X.var())
        return super().fit(rbf_kernel(X, gamma=self.gamma_), y)

    def predict(self, X):
        return super().predict(rbf_kernel(X, self.X_fit_, gamma=self.gamma_))

    def predict_proba(self, X):
        return super().predict_proba(rbf_kernel(X, self.X_fit_, gamma=self.gamma_))


# Load training and test patterns
train_orig_x, train_y = utilities.load_labeled_dataset(train_filelist, db_path, cache=memory)
test_orig_x = utilities.load_unlabeled_dataset(test_filelist, db_path, cache=memory)
utilities.shuffle_in_unison([train_orig_x, train_y], seed=1234)
print('Loaded %d training images (cats: %d, dogs: %d) and %d test images.'
      % (len(train_orig_x), np.count_nonzero(train_y == 0), np.count_nonzero(train_y == 1), len(test_orig_x)))

# HOG at two resolutions, identical for training and test
train_parts = [extract_hog_features(train_orig_x, side) for side in image_sides_solution2]
test_feature_x2 = np.hstack([extract_hog_features(test_orig_x, side) for side in image_sides_solution2])
train_feature_x2 = np.hstack(train_parts)
print('HOG train/test shape:', train_feature_x2.shape, test_feature_x2.shape)

# One pipeline per resolution (each sees only its own columns), combined by soft voting
bounds = np.cumsum([0] + [part.shape[1] for part in train_parts]).tolist()
clf_solution2 = VotingClassifier([('svc%d' % side, Pipeline([
    ('select', ColumnTransformer([('hog', 'passthrough', slice(start, stop))])),
    ('pow', FunctionTransformer(power_transform, kw_args={'alpha': alpha})),
    ('scaler', StandardScaler()),
    ('svc', PrecomputedRBFSVC(kernel='precomputed', C=3, probability=True, random_state=0)),
])) for side, start, stop in zip(image_sides_solution2, bounds, bounds[1:])], voting='soft')

# Evaluate the ensemble with repeated stratified cross-validation
cv_solution2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2024)
scores_solution2 = cross_val_score(clf_solution2, train_feature_x2, train_y, cv=cv_solution2, scoring='accuracy', n_jobs=4)
print('CV accuracy: %.4f +/- %.4f' % (scores_solution2.mean(), scores_solution2.std()))

# Train on the whole training set and save predictions
clf_solution2.fit(train_feature_x2, train_y)
predictions_solution2 = clf_solution2.predict(test_feature_x2)

with open(result_path, "w") as f:
    for prediction in predictions_solution2:
        f.write(str(int(prediction)) + '\n')
print('Predictions: %d cats / %d dogs' % (np.count_nonzero(predictions_solution2 == 0),
                                          np.count_nonzero(predictions_solution2 == 1)))
print('Predictions successfully saved to %s' % result_path)
