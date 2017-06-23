from numpy import *

def stumpClassify(data_matrix, dimen, thresh_val, thresh_ineq):
	ret_array = ones((shape(data_matrix)[0],1))

	if thresh_ineq == 'lt':
		ret_array[data_matrix[:,dimen] <= thresh_val] = -1.0
	else:
		ret_array[data_matrix[:,dimen] > thresh_val] = -1.0

	return ret_array