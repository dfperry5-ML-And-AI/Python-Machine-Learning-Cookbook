import numpy as np
from sklearn import preprocessing

## Create some sample data
data = np.array([
    [3, -1.5, 2, -5.4],
    [0, 4, -0.3, 2.1],
    [1, 3.3, -1.9, -4.3]
])

# Data Preprocessing -- Removing the Mean 
# It is usually valuable to remove the mean from each feature so that it is centered on zero.
# This helps remove bias
data_standardized = preprocessing.scale(data)
print("\n Mean= ", data_standardized.mean(axis=0))
print("\n Standard Deviation= ", data_standardized.std(axis=0))
## Mean is almost 0.
## Standard deviation is 1

# Data Preprocessing -- Scaling
# Values of each feature can vary between data points, so it can be important to
# Scale them to ensure a level playing field.
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("Data Scaled with MinMax Scaler= ", data_scaled)

# Data Preprocessing - Normalization
# Data normalization is used when you want to adjust the values in the feature vector so
# that they can be measured on a common scale. 
# One of the most common forms of normalization that is used in machine learning 
# adjusts the values of a feature vector so that they sum up to 1.
data_l1_normalized = preprocessing.normalize(data, norm='l1')
print("Data L1 Normalized: ", data_l1_normalized)
data_l2_normalized = preprocessing.normalize(data, norm='l2')
print("Data L2 Normalized: ", data_l2_normalized)

# Data Preprocessing - Binarization
# Used to conert numerical features into boolean vectors.
## Can define a threshold here that will return 1 for values > it, and 0 for values less than it.
data_binarizer = preprocessing.Binarizer(threshold=1.4)
data_binarized = data_binarizer.transform(data)
print("Binarized Data (Threshold 1.4) = ", data_binarized)

# Data Preprocessing - One Hot Encoding
# Sometimes we deal with numerical values that are sparse and scattered all over the place.
# One Hot Encoding is a tool to tighten the feature vector.
encoder = preprocessing.OneHotEncoder()
new_data_set = np.array([
    [0, 2, 1, 12],
    [1, 3, 5, 3],
    [2, 3, 2, 12],
    [1, 2, 4, 3]
])
encoder.fit(new_data_set)
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("Encoded Vector= ", encoded_vector)



 