from sklearn.naive_bayes import GaussianNB
from utils import plot_classifier
import numpy as np
from sklearn import model_selection

# Define the input file
input_file = "Chapter2-Building-A-Classifier/data_multivar.txt"

## Read in file
## Load vars in X, labels into Y
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1]) 

X = np.array(X)
y = np.array(y)

## Create and Train Gaussian Naive Bayes Classifier
classifier_gaussian_nb = GaussianNB()
classifier_gaussian_nb.fit(X, y)

## Predict Y
y_pred = classifier_gaussian_nb.predict(X)

## Compute Accuracy of classifier
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")
plot_classifier(classifier_gaussian_nb, X, y)

# Create Training / Test Data
# 75% for Training, 25% for Testing
# Model_Selection = Train/Test Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussian_nb_new = GaussianNB()
classifier_gaussian_nb_new.fit(X_train, y_train)

## Test Accuracy of Model
y_test_pred = classifier_gaussian_nb_new.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")
plot_classifier(classifier_gaussian_nb, X_test, y_test)

# Test Cross-Validation Accuracy
num_validations = 5
accuracy = model_selection.cross_val_score(classifier_gaussian_nb, 
        X, y, scoring='accuracy', cv=num_validations)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

## Test F1 Score (Precision and Recall together)
f1 = model_selection.cross_val_score(classifier_gaussian_nb, 
        X, y, scoring='f1_weighted', cv=num_validations)
print("F1: " + str(round(100*f1.mean(), 2)) + "%")

## Test Precision - only values OUTSIDE our model
precision = model_selection.cross_val_score(classifier_gaussian_nb, 
        X, y, scoring='precision_weighted', cv=num_validations)
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")

## Test Recall - ON TRAINED VALUES
recall = model_selection.cross_val_score(classifier_gaussian_nb, 
        X, y, scoring='recall_weighted', cv=num_validations)
print("Recall: " + str(round(100*recall.mean(), 2)) + "%")

