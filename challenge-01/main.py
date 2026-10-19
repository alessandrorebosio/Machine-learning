from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid, train_test_split

import utilities

result_path = 'Es3Predictions.txt'

feature_count = 2
dataset_path = 'DBs/PenDigits/pendigits_tr_Pca_K2.txt'

dataset_patterns, dataset_labels = utilities.load_labeled_dataset_from_txt(dataset_path, feature_count)
print('Shape dataset:', dataset_patterns.shape)
print('Shape labels:', dataset_labels.shape)

feature_count = 16
test_path = 'DBs/PenDigits/pendigits_te.txt'

test_x = utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)

clf = ...

clf.fit(dataset_patterns, dataset_labels)

predictions = clf.predict(test_x)

with open(result_path, "w") as f:
    for prediction in predictions:
        f.write(str(int(prediction)) + '\n')
print('Salvataggio delle prediction riuscito')