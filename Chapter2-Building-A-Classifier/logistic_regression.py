import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import utils
## Create Data Points
X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

## Initialize Logistic Classifier
## As we increase C, we increase the penalty for misclassification
## getting closer to optimal.
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

## Train the classifier
classifier.fit(X, y)

## Draw DataPoints
utils.plot_classifier(classifier, X, y)

