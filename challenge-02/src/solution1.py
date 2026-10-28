import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
from joblib import Memory
from skimage import color, feature
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

import utilities

# HOG and pipeline parameters
image_side = 128
orientations = 12
cell_side = 16
alpha = 0.5
n_pca = 256

db_path = 'DBs/CaniGatti_ML18'
exp_path = 'Experiments'
train_filelist = 'BinaryTrainingSet.txt'
test_filelist = 'Unlabeled_BinaryTestSet.txt'
result_path = 'Es4Predictions.txt'
memory = Memory(exp_path, verbose=0)


def extract_hog_features(images):
    """Resize to image_side x image_side, convert to grayscale, and extract HOG features."""
    resized = utilities.resize_images(images, image_side, image_side, cache=memory)
    hog_features = []
    for img in resized:
        hog = feature.hog(color.rgb2gray(img),
                          orientations=orientations,
                          pixels_per_cell=(cell_side, cell_side),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys')
        hog_features.append(hog)
    return np.array(hog_features)


def power_transform(X, alpha=0.5):
    """Element-wise x**alpha (HOG L2-Hys is >= 0; clipping guards against -0.0)."""
    return np.power(np.clip(X, 0, None), alpha)


# Load training and test patterns
train_raw_x, train_y = utilities.load_labeled_dataset(train_filelist, db_path, cache=memory)
test_raw_x = utilities.load_unlabeled_dataset(test_filelist, db_path, cache=memory)
utilities.shuffle_in_unison([train_raw_x, train_y], seed=1234)
print('Loaded %d training images (cats: %d, dogs: %d) and %d test images.'
      % (len(train_raw_x), np.count_nonzero(train_y == 0), np.count_nonzero(train_y == 1), len(test_raw_x)))

# Same preprocessing and HOG extraction for training and test images
train_feature_x = extract_hog_features(train_raw_x)
test_feature_x = extract_hog_features(test_raw_x)
print('HOG train shape:', train_feature_x.shape)
print('HOG test shape:', test_feature_x.shape)
assert test_feature_x.shape[1] == train_feature_x.shape[1]

# Build the classifier: every step is fitted only on the training data it receives
clf = Pipeline([
    ('pow', FunctionTransformer(power_transform, kw_args={'alpha': alpha})),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_pca, random_state=1234)),
    ('svc', SVC(kernel='rbf', C=3, gamma=6e-4)),
])

# Evaluate the pipeline with repeated stratified cross-validation
cv_solution1 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2024)
scores_solution1 = cross_val_score(clf, train_feature_x, train_y, cv=cv_solution1, scoring='accuracy', n_jobs=4)
print('CV accuracy: %.4f +/- %.4f' % (scores_solution1.mean(), scores_solution1.std()))

# Train on the whole training set and save predictions
clf.fit(train_feature_x, train_y)
predictions = clf.predict(test_feature_x)

with open(result_path, "w") as f:
    for prediction in predictions:
        f.write(str(int(prediction)) + '\n')
print('Predictions: %d cats / %d dogs' % (np.count_nonzero(predictions == 0),
                                          np.count_nonzero(predictions == 1)))
print('Predictions successfully saved to %s' % result_path)
