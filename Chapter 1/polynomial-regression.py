from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import linear_model
## One of the main constraints of a linear regression model is 
# the fact that it tries to fit a linear function to the input data. 
# The polynomial regression model overcomes this issue by allowing 
# the function to be a polynomial, thereby increasing the accuracy of the model.

polynomial = PolynomialFeatures(degree=10)

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
# print(X)
# print(Y)

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

X_train_transformed = polynomial.fit_transform(X_train)
datapoint = np.array([[0.39, 2.78, 7.11]])
poly_datapoint = polynomial.fit_transform(datapoint)

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, Y_train)


poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, Y_train)
print("\nLinear regression:", linear_regressor.predict(datapoint)[0])
print("\nPolynomial regression:", poly_linear_model.predict(poly_datapoint)[0])