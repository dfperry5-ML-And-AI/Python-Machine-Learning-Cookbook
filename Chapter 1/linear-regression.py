import sys
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pickle

# Linear regression refers to estimating the underlying function using a linear combination of input variable

# The goal of linear regression is to extract the underlying linear model 
# that relates the input variable to the output variable. 
# This aims to minimize the sum of squares of differences between the actual output 
# and the predicted output using a linear function. This method is called Ordinary least squares.

# Main Advantage is that its not complex.

## Read in the data
filename = "Chapter 1/data_singlevar.txt"
X = []
Y = []
with open(filename, "r") as f:
    for line in f.readlines():
        # print([float(i) for i in line.split(',')])
        ## Split the File
        xt, yt = [float(i) for i in line.split(',')]
        ## Populate X and Y
        X.append(xt)
        Y.append(yt)
# print(X)
# print(Y)

# Need to create two data sets:
# One for training, the other for testing.
# Rule of thumb -- 80% for Training, 20% for testing
num_training_points = int(0.8 * len(X))
num_test = len(X) - num_training_points

# Create Training the Data
X_train = np.array(X[:num_training_points]).reshape((num_training_points, 1))
Y_train = np.array(Y[:num_training_points])

# Create Test Data
X_test = np.array(X[num_training_points:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training_points:])

# Train the Model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, Y_train)

# Test the Model
y_train_pred = linear_regression.predict(X_train)
# Build the Plot
plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, y_train_pred, color='black')
plt.title("Training Data")
plt.show()

# Test
y_test_pred = linear_regression.predict(X_test)
plt.scatter(X_test, Y_test, color="green")
plt.plot(X_test, y_test_pred, color="black")
plt.title('Test Data')
plt.show()

## Check the Metrics
# A good practice is to make sure that the mean squared error is low 
# and the explained variance score is high.

print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(Y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, y_test_pred), 2))

## Save the model
output_model_file = 'Chapter 1/saved_model.pkl'
with open(output_model_file, 'wb') as output_file:
    pickle.dump(linear_regression, output_file)

## Import saved Model:
with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print ("\nNew mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred_new), 2))

    