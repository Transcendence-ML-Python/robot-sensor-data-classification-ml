from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import svm
from xgboost import XGBClassifier, plot_tree
from confusion_matrix_plot import confusion_matrix_plot
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from yellowbrick.classifier import ROCAUC

import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

def naive_bayes(X, Y, labels, dataset_name):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = GaussianNB()
	model_name = "Naive Bayes"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def decision_tree(X, Y, labels, dataset_name, random_state, criterion, ccp_alpha, min_samples_split):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = DecisionTreeClassifier(random_state=random_state, criterion=criterion, ccp_alpha=ccp_alpha, min_samples_split=min_samples_split)
	model_name = "Decision Tree"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels) 
	st.subheader(model_name)
	dot_data  = export_graphviz(model, filled=True, rounded=True, out_file=None)
	st.graphviz_chart(dot_data)
	return model

def single_layer_neural_network(X, Y, labels, dataset_name, eta0, random_state, max_iter):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = Perceptron(eta0=eta0, random_state=random_state, max_iter=max_iter)
	model_name = "Perceptron"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def mlp(X, Y, labels, dataset_name, random_state, activation, hidden_layer_sizes, max_iter, learning_rate):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = MLPClassifier(random_state=random_state, activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, learning_rate_init=learning_rate)
	model_name = "MLP"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def svm_model(X, Y, labels, dataset_name, gamma):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = svm.SVC(gamma=gamma)
	model_name = "SVM"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def xgboost_model(X, Y, labels, dataset_name):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	model = XGBClassifier(objective="binary:logistic", random_state=42)
	model_name = "XGBoost"
	scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels)
	return model

def scikit_validation_and_testing(X, Y, X_train, y_train, X_test, y_test, model, dataset_name, model_name, labels):
	model.fit(X_train, y_train)
	kfold_cross_validation_scikit(X_train, y_train, model)
	y_test_pred=model.predict(X_test)
	test_scikit_models(y_test, y_test_pred)
	if dataset_name == "Robot Dataset":
		plotting_ROC_curve_yellowbrick(model, labels, X_train, y_train, X_test, y_test)
	else:
		precision_recall_curve(X, Y, model, model_name)
	confusion_matrix_plot(model, X_test, y_test, dataset_name, model_name, labels)
	return

def kfold_cross_validation_scikit(X_train, y_train, model):
	result = cross_validation_functions(model, X_train, y_train)
	st.subheader("Validation Result - KFold")
	st.write("Accuracy: %.2f" % (result['mean']*100), "%")
	st.write("Standard Deviation: %.2f" % (result['std']*100))
	st.write("Confusion Matrix:\n", result['conf_mat'])
	return

def cross_validation_functions(model, input, output):
   kfold = StratifiedKFold(n_splits=10, random_state=1)
   cv_results = cross_val_score(model, input, output, cv=kfold, scoring='accuracy')
   y_pred = cross_val_predict(model, input, output, cv=10)
   conf_mat = confusion_matrix(output, y_pred)
   mean = cv_results.mean()
   std = cv_results.std()
   return ({
      'cv_results': cv_results,
      'conf_mat': conf_mat,
      'mean': mean,
      'std': std
   })

def test_scikit_models(y_test, y_test_pred):
	st.subheader("Test Result")
	accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)
	st.write("Accuracy = %.2f" % (accuracy*100), "%")      
	st.write("Confusion Matrix :\n", confusion_matrix(y_true=y_test, y_pred=y_test_pred))
	return

def plotting_ROC_curve_yellowbrick(model, labels, X_train, y_train, X_test, y_test):
    st.subheader('ROC Curve')
    visualizer = ROCAUC(model, classes=labels)
    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()
    return