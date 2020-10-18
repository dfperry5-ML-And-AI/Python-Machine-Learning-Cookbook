import sys
import csv

import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from my_utils import plot_feature_importances


## Step 1: Load the dataset. So provide the function.
def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rt'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        # X.append(row[2:13])
        # X.append(row[2:15])
        X.append(row[2:14])
        y.append(row[-1])

    # Extract feature names
    feature_names = np.array(X[0])

    # Remove the first row because they are feature names
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

# For dynamic commandline loading
# X, y, feature_names = load_dataset(sys.argv[1])
print("Loading DataSet")
X, y, feature_names = load_dataset("Chapter 1/bike_hour.csv")
X, y  = shuffle(X, y, random_state=7)

## Separate data into train and test data. 90% Train
num_training = int(0.9 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

## Build Random Forest Regressor
# n_estimators = number of trees
# depth = max depth of tree
print("Building Regressor")
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10)
rf_regressor.fit(X_train, y_train)

## Evaluate Random Forest
y_pred = rf_regressor.predict(X_test)

## Find the value
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred) 
print("\n#### Random Forest regressor performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print(rf_regressor.feature_importances_)

plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)

