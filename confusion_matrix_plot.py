from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import streamlit as st

def confusion_matrix_plot(model, X_test, y_test, dataset_name, model_name, disp_labels):
	st.subheader("Confusion Matrix")
	title = "Confusion Matrix of %s using %s" % (dataset_name, model_name) 
	disp = plot_confusion_matrix(model, X_test, y_test,
								display_labels=disp_labels,
								cmap=plt.cm.Blues,
								normalize='true')
	disp.ax_.set_title(title)
	st.pyplot()
	return