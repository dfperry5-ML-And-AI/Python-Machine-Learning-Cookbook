import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import utils

## Create Sample Data
sample_data = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])
## Assign labels to the points
## We only have 2 classes - 0, and 1.
## If you have more classes, range will go from 0 to N-1
labels = [0, 1, 1, 0, 0, 1, 1, 0]

## Separate data into classes based on labels.
class_0 = np.array([sample_data[i] for i in range(len(sample_data)) if labels[i]==0])
class_1 = np.array([sample_data[i] for i in range(len(sample_data)) if labels[i]==1])

## Plot Classes
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')

## Draw dividing line
line_x = range(10)
line_y = line_x
plt.plot(line_x, line_y, color='black', linewidth=3)
plt.show()

## Simple Classifier:
## Input Point: (x, y)
## if X >= Y, class = 1
## if X < Y, class = 0
## This is a linear classifier.





