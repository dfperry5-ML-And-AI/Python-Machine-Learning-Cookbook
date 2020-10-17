import sys
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pickle
## Ordinary least squares considers every single datapoint when 
# it's building the model. Hence, the actual model ends up looking 
# like the dotted line as shown in the preceding figure. 
# We can clearly see that this model is suboptimal. 
# To avoid this, we use regularization where a penalty is imposed on 
# the size of the coefficients. This method is called Ridge Regression.

# Read In Data
filename = "Chapter 1/data_multivar.txt"
X = []
Y = []
with open(filename, "r") as f:
    for line in f.readlines():
        # print([float(i) for i in line.split(',')])
        ## Split the File
        row = [float(i) for i in line.split(',')]
        xt, yt = row[:-1], row[-1]
        X.append(xt)
        Y.append(yt)
print(X)
print(Y)

# Need to create two data sets:
# One for training, the other for testing.
# Rule of thumb -- 80% for Training, 20% for testing
num_training_points = int(0.8 * len(X))
num_test = len(X) - num_training_points

# Create Training the Data
X_train = np.array(X[:num_training_points])
Y_train = np.array(Y[:num_training_points])

# Create Test Data
X_test = np.array(X[num_training_points:])
Y_test = np.array(Y[num_training_points:])

# Create Ridge Regression
ridge_regressor = linear_model.Ridge()
linear_regressor = linear_model.LinearRegression()

## Now Train Ridge Regressor
ridge_regressor.fit(X_train, Y_train)
linear_regressor.fit(X_train, Y_train)

y_ridge_test_pred = ridge_regressor.predict(X_test)
y_linear_test_pred = linear_regressor.predict(X_test)

## Check the Metrics
# A good practice is to make sure that the mean squared error is low 
# and the explained variance score is high.
print("========== RIDGE RESULTS ========")
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_ridge_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_ridge_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, y_ridge_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(Y_test, y_ridge_test_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, y_ridge_test_pred), 2))

print("\n ========== LINEAR RESULTS ========")
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_linear_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_linear_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, y_linear_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(Y_test, y_linear_test_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, y_linear_test_pred), 2))

