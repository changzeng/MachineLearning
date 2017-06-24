from numpy import *

def stumpContinuesClassify(data_matrix, dimen, thresh_val, thresh_ineq):
	ret_array = ones((shape(data_matrix)[0],1))

	if thresh_ineq == 'lt':
		ret_array[data_matrix[:,dimen] <= thresh_val] = -1.0
	else:
		ret_array[data_matrix[:,dimen] > thresh_val] = -1.0

	return ret_array

def stumpDiscreteClassify(data_matrix, dimen, val):
	ret_array = zeros((shape(data_matrix)[0],1))

	ret_array[data_matrix[:,dimen] == val] = 1

	return ret_array

def buildContinuesStump(data_arr, class_labels, D):
	data_matrix = mat(data_arr)
	label_mat = mat(class_labels).T

	m,n = shape(data_matrix)
	num_steps = 10.0
	best_stump = {}
	best_clas_est = mat(zeros((m,1)))
	min_error = inf

	for i in range(n):
		range_min = data_matrix[:,i].min()
		range_max = data_matrix[:,i].max()

		step_size = (range_max-range_min)/num_steps
		for j in range(-1,int(num_steps)+1):
			for inequal in ['lt','gt']:
				thresh_val = (range_min + float(j)*step_size)
				predicted_vals = stumpClassify(data_matrix,i,thresh_val,inequal)

				err_arr = mat(ones((m,1)))
				# array filtrate
				err_arr[predicted_vals == label_mat] = 0
				weighted_error = D.T * err_arr
				# print("split: dim %d, thresh %.2f,thresh inequal: %s, the weighted error is %.3f" % (i, thresh_val, inequal, weighted_error))

				if weighted_error < min_error:
					min_error = weighted_error
					best_clas_est = predicted_vals.copy()
					best_stump['dim'] = i
					best_stump['thresh'] = thresh_val
					best_stump['ineq'] = inequal

	return best_stump,min_error,best_clas_est

def buildDiscreteStump(data_arr, class_labels, D):
	data_matrix = mat(data_arr)
	label_mat = mat(class_labels).T

	m,n = shape(data_matrix)
	best_stump = {}
	best_clas_est = mat(zeros((m,1)))
	min_error = inf

	for i in range(n):
		for feature in (1.0,0.0):
			predicted_vals = stumpDiscreteClassify(data_matrix,i,feature)

			err_arr = mat(ones((m,1)))
			# array filtrate
			err_arr[predicted_vals == label_mat] = 0
			weighted_error = D.T * err_arr

			if weighted_error < min_error:
				min_error = weighted_error
				best_clas_est = predicted_vals.copy()
				best_stump['dim'] = i
				best_stump['thresh'] = feature

	return best_stump,min_error,best_clas_est

# from adaboost import loadSimpData
# data_mat,class_labels = loadSimpData()
# D = mat(ones((5,1))/5)
# buildStump(data_mat,class_labels,D)