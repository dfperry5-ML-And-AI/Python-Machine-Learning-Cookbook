import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

# Load the Data
input_file = "Chapter2-Building-A-Classifier/adult.data.txt"
# Reading the data
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 10000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k = count_lessthan50k + 1

        elif data[-1] == '>50K' and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k = count_morethan50k + 1

        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break

X = np.array(X)

## Convert String Attributes to Numerical Data, while leaving out original numerical data
label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else: 
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Build a classifier
classifier_gaussian_nb = GaussianNB()
classifier_gaussian_nb.fit(X, y)

# Split Data to Test and Training Sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussian_nb = GaussianNB()
classifier_gaussian_nb.fit(X_train, y_train)
y_test_pred = classifier_gaussian_nb.predict(X_test)

f1 = model_selection.cross_val_score(classifier_gaussian_nb, X, y, scoring='f1_weighted', cv=5)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Predict and print output for a particular datapoint
output_class = classifier_gaussian_nb.predict([X[5]])
print(label_encoder[-1].inverse_transform(output_class)[0])