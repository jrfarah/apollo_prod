###################################################################################
# Version 1 of the main neural network implementation, with specific prediction
# this will attempt to functionalize the code drawn up in prep3
# and include a way for external programs to interface with this module
# note: this module will ONLY include customization relevant to the data analysis 
# portion, not the data formatting portion
# from there every part of this will be expanded and customized
#
# things that are passed into the function: 
# - dataset, list of columns, test vector, starting v, secondary v, number of
# iterations, show graph?
###################################################################################

valve = ""

# IMPORTS #########################################################################

# add filepaths to sys path import
import sys
sys.path.insert(0, '../interface/core')
sys.path.insert(0, '../prep/')
sys.path.insert(0, '../training_sets/')

# rest of the regular imports
import DataFormatterv1
import ClassificationNetv1
import pandas
import numpy
import random
import math
import matplotlib.pyplot as plt

# METHODS ########################################################################
def getLowerBound(hist):

	desirability = []
	for point in hist:
		print point
		desirability.append(1/(((point[1]-point[0]))*((hist.index(point)-3.5)**10))) # old 7.5

	probable_bounds = hist[desirability.index(max(desirability[1:]))]

	print "It is most likely on this range: ", probable_bounds

	return min(min(probable_bounds), max(probable_bounds), key=abs)


def IterativeApproximation(dset, test_vector, column_list, start_v, second_v, num_iter, show_graph=False):
	''' coalesces the other functions into a single one, handles all program
		inputs 
	'''

	# save the important column from the list of columns
	backup_column_list = []
	for element in column_list[-1]:
		backup_column_list.append(element)
	use_column_list = column_list
	backup_dset = dset
	print backup_column_list
	prediction_column = use_column_list[-1]

	# delete it from the list of columns
	del use_column_list[-1]

	# define the boundaries and an empty bound-history list
	boundaries = [-1*numpy.inf, 1*numpy.inf]
	history_of_boundaries = [boundaries]

	# is the value greater than or less than zero?
	# first let's check: are all the values positive/negative?
	is_positive = all(i >= 0 for i in prediction_column) 

	# if they are all positive, then we know what to do with the boundaries
	if is_positive == True:
		boundaries = [0, 1*numpy.inf]
		history_of_boundaries.append(boundaries)


	# define v to be start_v and do the first iteration to begin the 
	# bootstrap; define an empty editable column
	v = start_v
	newclass_column = []

	# fill the new column with the new data--are the known values 
	# bigger or smaller than start_v
	for res in prediction_column:
		if res >= v:
			newclass_column.append(1)
		else:
			newclass_column.append(0)

	# add the new column to the dataset and turn it back into a DataFrame
	use_column_list.append(newclass_column)
	new_dataframe = pandas.DataFrame(use_column_list)

	# transpose the dataframe
	rotated = pandas.DataFrame.transpose(new_dataframe)

	# test the prediction vector against the dataset
	output = ClassificationNetv1.Predict(rotated, len(test_vector), test_vector)

	# does the net think the output will be larger or smaller than the bound?
	# if the output is larger (1) make the V value the new lower bound
	if output == 1:
		boundaries = [v, boundaries[1]]
		sign = 1

	# if it is smaller (0) make it the new upper bound
	elif output == 0:
		boundaries = [boundaries[0], v]

		# if we are testing for positivity/negativity, we know that it has to
		# be negative, since zero is an upper bound
		if v == 0:
			sign = -1

	# add the new boundaries
	history_of_boundaries.append(boundaries)

	# still working on this! pick a v (for now, the secondary V)
	# TODO: automatically decide on a secondary V
	v = second_v

	# iterate for as many times as the user specifies
	for iteration in range(num_iter):

		# delete the newly added list
		del use_column_list[-1]

		# define an empty editable column
		newclass_column = []

		# are the boundaries still infinity? in that case we have to narrow it
		# down
		# TODO: FIGURE OUT A WAY OF BETTER PICKING V
		if boundaries[0] == -1*numpy.inf or boundaries[1] == numpy.inf:
			v = 10*abs(v)
		else:
			# if the boundaries are fine, find the average and use that as v
			v = abs(boundaries[0]) + abs(boundaries[1])
			v = float(abs(v/2))

		# build the new column based on the new v
		for res in prediction_column:
			if res >= sign*v:
				newclass_column.append(1)
			else:
				newclass_column.append(0)

		# append the new column to the dataset for analysis and make
		# a new dataframe
		use_column_list.append(newclass_column)
		new_dataframe = pandas.DataFrame(use_column_list)
		rotated = pandas.DataFrame.transpose(new_dataframe)

		# check the output, is it greater than or less than v?
		output = ClassificationNetv1.Predict(rotated, len(test_vector), test_vector)

		# does the net think the output will be larger or smaller than the bound?
		# if the output is larger (1) make the V value the new lower bound
		if output == 1 and sign*v >= boundaries[0]:
			boundaries = [sign*v, boundaries[1]]
		elif output == 0 and sign*v <= boundaries[1]:
			boundaries = [boundaries[0], sign*v]

		# append the new boundaries
		history_of_boundaries.append(boundaries)

	lower_bound = getLowerBound(history_of_boundaries)

	print lower_bound

	print backup_column_list

	del column_list[-1]

	column_list.append(backup_column_list)

	(boundaries, history_of_boundaries) = IterativeApproximationFinite(dset, test_vector, column_list, lower_bound, 20, sign, show_graph=True)

	# if show_graph == True:
	# 	plt.plot(history_of_boundaries)
	# 	plt.show()

	# return the history and the final estimate
	return boundaries, history_of_boundaries

def IterativeApproximationFinite(dset, test_vector, column_list, lower_bound, num_iter, sign, show_graph=False):
	''' coalesces the other functions into a single one, handles all program
		inputs 
	'''
	print dset
	# save the important column from the list of columns
	prediction_column = column_list[-1]
	print column_list
	print prediction_column

	# delete it from the list of columns
	del column_list[-1]

	# define the boundaries and an empty bound-history list
	boundaries = [-1*numpy.inf, 1*numpy.inf]
	history_of_boundaries = []

	# is the value greater than or less than zero?
	# first let's check: are all the values positive/negative?
	# is_positive = all(i >= 0 for i in prediction_column)
	is_positive = True
	if sign == -1:
		is_positive = False

	# if they are all positive, then we know what to do with the boundaries
	if is_positive == True:
		boundaries = [lower_bound, 1*numpy.inf]
		history_of_boundaries.append(boundaries)
	else:
		boundaries = [-1*numpy.inf, lower_bound]
		history_of_boundaries.append(boundaries)


	# define v to be start_v and do the first iteration to begin the 
	# bootstrap; define an empty editable column
	v = lower_bound
	newclass_column = []

	# fill the new column with the new data--are the known values 
	# bigger or smaller than start_v
	for res in prediction_column:
		if res >= v:
			newclass_column.append(1)
		else:
			newclass_column.append(0)

	# add the new column to the dataset and turn it back into a DataFrame
	column_list.append(newclass_column)
	new_dataframe = pandas.DataFrame(column_list)

	# transpose the dataframe
	rotated = pandas.DataFrame.transpose(new_dataframe)

	# test the prediction vector against the dataset
	output = ClassificationNetv1.Predict(rotated, len(test_vector), test_vector)

	# add the new boundaries
	history_of_boundaries.append(boundaries)

	# still working on this! pick a v (for now, the secondary V)
	# TODO: automatically decide on a secondary V
	# v = second_v

	# iterate for as many times as the user specifies
	for iteration in range(num_iter):

		# delete the newly added list
		del column_list[-1]

		# define an empty editable column
		newclass_column = []

		# are the boundaries still infinity? in that case we have to narrow it
		# down
		# TODO: FIGURE OUT A WAY OF BETTER PICKING V
		if boundaries[0] == -1*numpy.inf or boundaries[1] == numpy.inf:
			v = 10*abs(v)
		else:
			# if the boundaries are fine, find the average and use that as v
			v = abs(boundaries[0]) + abs(boundaries[1])
			v = float(abs(v/2))

		# build the new column based on the new v
		for res in prediction_column:
			if res >= sign*v:
				newclass_column.append(1)
			else:
				newclass_column.append(0)

		# append the new column to the dataset for analysis and make
		# a new dataframe
		column_list.append(newclass_column)
		new_dataframe = pandas.DataFrame(column_list)
		rotated = pandas.DataFrame.transpose(new_dataframe)

		# check the output, is it greater than or less than v?
		output = ClassificationNetv1.Predict(rotated, len(test_vector), test_vector, val_size=0.4)

		# does the net think the output will be larger or smaller than the bound?
		# if the output is larger (1) make the V value the new lower bound
		if output == 1 and sign*v >= boundaries[0]:
			boundaries = [sign*v, boundaries[1]]
		elif output == 0 and sign*v <= boundaries[1]:
			boundaries = [boundaries[0], sign*v]

		# append the new boundaries
		history_of_boundaries.append(boundaries)

	if show_graph == True:
		plt.plot(history_of_boundaries)
		plt.show()

	# return the history and the final estimate
	return boundaries, history_of_boundaries