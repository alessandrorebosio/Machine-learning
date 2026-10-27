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

# Path containing test patterns
image_side = 128
orientations = 12
cell_side = 16
alpha = 0.5
n_pca = 256

db_path = 'DBs/CaniGatti_ML18'
exp_path = 'Experiments'
train_filelist = 'BinaryTrainingSet.txt'
test_filelist = 'Unlabeled_BinaryTestSet.txt'
result_path = 'pred_sol2.txt'  # Later copied as Es4Predictions.txt
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


# Load training patterns
train_raw_x, train_y = utilities.load_labeled_dataset(train_filelist, db_path, cache=memory)
utilities.shuffle_in_unison([train_raw_x, train_y], seed=1234)
print('Loaded %d training images (cats: %d, dogs: %d).'
      % (len(train_raw_x), np.count_nonzero(train_y == 0), np.count_nonzero(train_y == 1)))

# Preprocessing and HOG extraction (training)
train_feature_x = extract_hog_features(train_raw_x)
print('HOG train shape:', train_feature_x.shape)

# Build the classifier
clf = Pipeline([
    ('pow', FunctionTransformer(power_transform, kw_args={'alpha': alpha})),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_pca, random_state=1234)),
    ('svc', SVC(kernel='rbf', C=3, gamma=6e-4)),
])

# Train the classifier
clf.fit(train_feature_x, train_y)

# Load test patterns
test_raw_x = utilities.load_unlabeled_dataset(test_filelist, db_path, cache=memory)

# Preprocessing and HOG extraction (test)
test_feature_x = extract_hog_features(test_raw_x)
print('HOG test shape:', test_feature_x.shape)
assert test_feature_x.shape[1] == train_feature_x.shape[1]

cv_solution2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2024)
scores2 = cross_val_score(clf, train_feature_x, train_y, cv=cv_solution2, scoring='accuracy', n_jobs=4)
print('CV accuracy: %.4f +/- %.4f' % (scores2.mean(), scores2.std()))

# Save predictions
predictions = clf.predict(test_feature_x)

with open(result_path, "w") as f:
    for prediction in predictions:
        f.write(str(int(prediction)) + '\n')
print('Predictions: %d cats / %d dogs' % (np.count_nonzero(predictions == 0),
                                          np.count_nonzero(predictions == 1)))
print('Ok')