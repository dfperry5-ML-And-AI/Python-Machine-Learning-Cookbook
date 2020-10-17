from sklearn import preprocessing

# Create a Label Encoder
label_encoder = preprocessing.LabelEncoder()

# Define Labels
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

# Encode these labels.
label_encoder.fit(input_classes)
print("Class Mapping: ")
## classes_ is a tuple
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)

# The words have now been transformed into 0-indexed numbers.
# Now, when we encounter the labels again, they are easy to transform.
new_labels = ['toyota', 'ford', 'audi']
encoded_new_labels = label_encoder.transform(new_labels)
print("\n Labels: ", new_labels)
print("\n Encoded Labels: ", encoded_new_labels)

# Main advantage here is we no longer need to maintain a map between words and numers
# the encoder does it for us.
# Now, transform numbers back to words.
test_encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(test_encoded_labels)
print("\n Encoded Labels: ", test_encoded_labels)
print("\n Decoded Labels: ", decoded_labels)


