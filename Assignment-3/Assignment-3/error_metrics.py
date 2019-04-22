from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Prints standard regression error metrics given predicted and actual values.
def print_regression_error_metrics(preds, y_test):
    print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds),
           median_absolute_error(y_test, preds),
           r2_score(y_test, preds),
           explained_variance_score(y_test, preds)]))
		   
# Print out common error metrics for binary classifications.
def print_binary_classif_error_report(preds, y_test):
	print('Accuracy: ' + str(accuracy_score(preds, y_test)))
	print('Precision: ' + str(precision_score(preds, y_test)))
	print('Recall: ' + str(recall_score(preds, y_test)))
	print('F1: ' + str(f1_score(preds, y_test)))
	print('ROC AUC: ' + str(roc_auc_score(preds, y_test)))
	print('Confusion Matrix: \n' + str(confusion_matrix(preds, y_test)))
	
# Print out common error metrics for multinomial classifications.
def print_multiclass_classif_error_report(preds, y_test):
	print('Accuracy: ' + str(accuracy_score(preds, y_test)))
	print('Avg. F1 (Micro): ' + str(f1_score(preds, y_test, average='micro')))
	print('Avg. F1 (Macro): ' + str(f1_score(preds, y_test, average='macro')))
	print('Avg. F1 (Weighted): ' + str(f1_score(preds, y_test, average='weighted')))
	print(classification_report(y_test, preds))
	print('Confusion Matrix: \n' + str(confusion_matrix(preds, y_test)))