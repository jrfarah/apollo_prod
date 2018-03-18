###################################################################################
# Version 1 of the main data formatter
# this will attempt to reduce the data formatting to a single line
# and include a way for external programs to interface with this module
# note: this module will ONLY include customization relevant to the data formatting 
# portion, not the data analysis portion
# from there every part of this will be expanded and customized
#
# things that are passed into the function: 
# - dataset, array of names?, predidction set, prediction column index
###################################################################################

# add filepaths to sys path import
import sys
sys.path.insert(0, '../interface/')
sys.path.insert(0, '../prep/')
sys.path.insert(0, '../training_sets/')

# rest of the regular imports
import 	os
import 	numpy
import 	pandas
import 	matplotlib.pyplot as plt

# METHODS ########################################################################

def getAbsolutePath(f):
	return os.path.abspath(f)


def getNumberOfColumns(f):
	''' gets the number of columns the csv '''
	ds = pandas.read_csv(f)
	(r, c) = ds.shape

	return c

def generateNames(n):
	''' generates a name vector based on the number of columns '''
	names = []
	for i in range(1, n+1):
		 names.append(str(i))

	return names


def Format(filepath):
	''' coalesces all of the other functions into one thing, will return
		the dataset object
	'''
	filepath = getAbsolutePath(filepath)
	num_column = getNumberOfColumns(filepath)
	names = generateNames(num_column)
	print names, filepath
	return pandas.read_csv(filepath, names=names)