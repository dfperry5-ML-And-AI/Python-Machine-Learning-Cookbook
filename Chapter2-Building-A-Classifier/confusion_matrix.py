from sklearn.metrics import confusion_matrix, classification_report
import utils

## Create Data
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]

## Generate Confusion Matrix
confusion_mat = confusion_matrix(y_true, y_pred)

## Plot Confusion Matrix
utils.plot_confusion_matrix(confusion_mat)
## Target Names
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
## print it out
print(classification_report(y_true, y_pred, target_names=target_names))