'''
'''
import os, sys 
import numpy as np 
from ctypes import c_double
from ssl import * 

def ssl_train_with_data(X, y, verbose, **kwargs):
	# check y
	if not isinstance(y, np.ndarray):
		if not isinstance(y, list):
			raise ValueError('y must be an iterable type (list, numpy.ndarray)')
		else:
			y = np.array(y)
	else:
		if np.prod(y.shape) != y.shape[0]:
			raise ValueError('y must be a column or row vector')

	# check y
	labels = set(y)
	if not (labels == set([1.0,-1.0,0.0])) and not (labels == set([1.0,-1.0])):
		raise ValueError('label array must contain positive(+1) and negative (-1) samples, and optionally unlabeled ones (0).')

	# check x vs. y
	if not isinstance(X, np.ndarray):
		raise ValueError('X and y must be numpy.ndarray')
	elif X.shape[0] != y.shape[0]:
		raise ValueError('X and y must have  the same number of samples')

	ssl_data = data()
	ssl_weights = vector_double()
	ssl_options = options(**kwargs)
	ssl_data.from_data(X,y, ssl_options.Cp, ssl_options.Cn)
	ssl_outputs = vector_double()
	libssl.ssl_train(ssl_data, ssl_options, ssl_weights, ssl_outputs, verbose)
	# libssl.L2_SVM_MFN(ssl_data, ssl_options, ssl_weights, ssl_outputs, verbose)
	
	clf = np.array(np.fromiter(ssl_weights.vec, dtype=np.float64, count=ssl_weights.d))

	#libssl.clear_data(ssl_data)
	libssl.clear_vec_double(ssl_outputs)
	libssl.clear_vec_double(ssl_weights)

	return clf
