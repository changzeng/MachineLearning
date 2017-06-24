from numpy import *
from boost import *

def loadSimpData():
    data_mat = matrix([[1.  ,  2.1],
        [2.  ,  1.1],
        [1.3 ,  1. ],
        [1.  ,  1. ],
        [2.  , 1.  ]])
    class_label = [1.0, 1.0, -1.0, -1.0, 1.0]

    return data_mat,class_label

def adaBoostTrainDS(data_arr,class_labels,num_it=40):
	weak_class_arr = []
	m = shape(data_arr)[0]
	D = mat(ones((m,1))/m)
	agg_class_est = mat(zeros((m,1)))

	for i in range(num_it):
		best_stump,error,class_est = buildStump(data_arr,class_labels,D)
		# print("D:",D.T)
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
		best_stump['alpha'] = alpha
		weak_class_arr.append(best_stump)
		# print("classEst: ",class_est.T)
		expon = multiply(-1*alpha*mat(class_labels).T,class_est)
		D = multiply(D,exp(expon))
		D = D/D.sum()
		agg_class_est += alpha*class_est
		# print("aggClassEst: ",agg_class_est.T)
		agg_errors = multiply(sign(agg_class_est) != mat(class_labels).T,ones((m,1)))
		error_rate = agg_errors.sum()/m

		print("total error: ",error_rate)

		if error_rate == 0.0:
			break

	return weak_class_arr

def adaClassify(data,classifier_arr):
	data_mat = mat(data)
	m = shape(data_mat)[0]
	agg_class_est = mat(zeros((m,1)))
	for i in range(len(classifier_arr)):
		class_est = stumpClassify(data_mat,classifier_arr[i]['dim'],classifier_arr[i]['thresh'],classifier_arr[i]['ineq'])
		agg_class_est += classifier_arr[i]['alpha']*class_est

		print(agg_class_est)

	return sign(agg_class_est)

data_mat,class_labels = loadSimpData()
weak_class_arr = adaBoostTrainDS(data_mat,class_labels)
predict = adaClassify(data_mat,weak_class_arr)

print(class_labels)
print(predict)