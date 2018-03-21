###################################################################################
# Version 1 of the main neural network implementation
# this will attempt to functionalize the code drawn up in prep2
# and include a way for external programs to interface with this module
# note: this module will ONLY include customization relevant to the data analysis 
# portion, not the data formatting portion
# from there every part of this will be expanded and customized
#
# things that are passed into the function: 
# - dataset, array of names?, predidction set, prediction column index
###################################################################################

valve = ""

# IMPORTS #########################################################################

# add filepaths to sys path import
import sys
sys.path.insert(0, '../interface/')
sys.path.insert(0, '../prep/')
sys.path.insert(0, '../training_sets/')

# rest of the regular imports
import 	numpy
import 	pandas
import 	matplotlib.pyplot as plt
from 	pandas.plotting 				import scatter_matrix
from 	sklearn 						import model_selection
from 	sklearn.metrics 				import classification_report
from 	sklearn.metrics 				import confusion_matrix
from 	sklearn.metrics 				import accuracy_score
from 	sklearn.linear_model 			import LogisticRegression
from 	sklearn.tree 					import DecisionTreeClassifier
from 	sklearn.neighbors 				import KNeighborsClassifier
from 	sklearn.discriminant_analysis 	import LinearDiscriminantAnalysis
from 	sklearn.naive_bayes 			import GaussianNB
from 	sklearn.svm 					import SVC
from 	sklearn 						import linear_model
from 	sklearn 						import svm

# METHODS ########################################################################

def getValidation(dset, prediction_column_index, v_size):
	'''	extract a validation dataset from the full one
	'''
	print type(dset)
	reduced_dataset = dset.values

	# split the array into two arrays--one containing the contemplation information and the other containing the information we want to predict

	prediction_column 		= reduced_dataset[:, prediction_column_index]
	contemplation_columns 	= numpy.delete(	reduced_dataset, 
											prediction_column_index, 
											axis = 1 )
	# print prediction_column 	# PC
	# print contemplation_columns	# CC

	# percentage of data to be used for validation
	validation_size = v_size

	# set random seed for initial matrices
	seed = 7

	# get the training and validation models (to be passed on)
	CC_train, CC_validation, PC_train, PC_validation = model_selection.train_test_split(contemplation_columns, prediction_column, test_size=validation_size, random_state=seed)

	return CC_train, CC_validation, PC_train, PC_validation


def spotCheckAlgorithms(CC_train, CC_validation, PC_train, PC_validation, scoring = 'accuracy'):
	''' checks which algorithm is most effective on the dataset '''
	models = []
	models.append(('LR', 	LogisticRegression()))
	models.append(('LDA', 	LinearDiscriminantAnalysis()))
	models.append(('KNN', 	KNeighborsClassifier()))
	models.append(('CART', 	DecisionTreeClassifier()))
	models.append(('NB', 	GaussianNB()))
	models.append(('SVM', 	SVC()))

	seed = 7

	# evaluate each model in turn
	# after evaluation, dynamically select the correct model to use
	# formula for adjusted/normalized success: 1-(1/(mean/std))
	success = []
	results = []
	names 	= []
	for name, model in models:
		# test the models on a subset of the dataset
		kfold 		= model_selection.KFold(n_splits=10, random_state=seed)
		cv_results 	= model_selection.cross_val_score(model, CC_train, PC_train, cv=kfold, scoring=scoring)
		# calculate the success of the model (weighted success)
		success.append((1-float((1/(cv_results.mean()/cv_results.std()))), model, name))
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		# print(msg)

	(rate, best_model, name) = (max(success)[0], max(success)[1], max(success)[2])
	print "{0} model is right {1} percent of the time.".format(name, rate*100)
	return rate, best_model, name

def spawnNeuralNet(rate, best_model, prediction_set, CC_train, PC_train):
	''' generates the neural net and thinks about the data '''

	# generates the neural network based on the best mdoel
	neurNet = best_model

	# trains the neural network
	neurNet.fit(CC_train, PC_train)

	# formats the prediction set into a numpy array and reshapes it
	prediction_set = numpy.array(prediction_set).reshape(1,-1)

	# think
	predictions = neurNet.predict(prediction_set)

	return predictions


def Predict(dset, prediction_column_index, prediction_set, val_size=0.2):
	'''	coalesces all of the other functions into one thing; 
		will return prediction
	'''
	# split the datasets between validation and prediction
	CC_train, CC_validation, PC_train, PC_validation = getValidation(dset, prediction_column_index, val_size)

	# spot check the various algorithms to determine the best model for use in this case
	(rate, best_model, name) = spotCheckAlgorithms(CC_train, CC_validation, PC_train, PC_validation)

	# think 
	result = spawnNeuralNet(rate, best_model, prediction_set, CC_train, PC_train)
	return result
	
