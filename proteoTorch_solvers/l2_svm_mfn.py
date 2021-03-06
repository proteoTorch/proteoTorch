'''
'''
from os import path
import numpy as np 
from ctypes import *

try:
	dirname = path.dirname(path.abspath(__file__))
	libssl = CDLL(path.join(dirname, 'libssl.so'))
except:
	raise Exception("Could not find libssl.so")

def genFields(names, types):
	return list(zip(names, types))

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes

# construct constants
class data(Structure):
	'''
	m: number of examples
	n: number of features
	X: flattened 2D feature matrix
	Y: labels
	'''
	_names = ['m', 'n', 'X', 'Y']
	_types = [c_int, c_int, POINTER(c_double), POINTER(c_double) ]
	_fields_ = genFields(_names, _types)

	def __str__(self):
		s = ''
		attrs = data._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs)
		for attr, val in zip(attrs, values):
			s += ('%s: %s\n' % (attr, val))
		s.strip()

		return s

	def from_data(self, X, y):
		self.__frombuffer__ = False
		# set constants
		self.m = len(y)
		self.n = X.shape[1]+1 # include bias term

		# Copy data over
		self.Y = np.ctypeslib.as_ctypes(y.astype(np.float64))
		self.X = np.ctypeslib.as_ctypes(X.reshape(-1))
	def __init__(self):
		self.__createfrom__ = 'python'
		self.__frombuffer__ = True

class vector_double(Structure):
	_names = ['d', 'vec']
	_types = [c_int, POINTER(c_double)]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'


class vector_int(Structure):
	_names = ['d', 'vec']			
	_types = [c_int, POINTER(c_int)]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'


class options(Structure):
	_names = ['lambda_l', 'Cp', 'Cn', 'epsilon', 'cgitermax', 'mfnitermax']
	_types = [c_double, c_double, c_double, c_double, c_int, c_int]
	_fields_ = genFields(_names, _types)

	def __init__(self, **kwargs):
		self.set_defaults()

		if kwargs:
			if 'lambda_l' in kwargs.keys():
				self.lambda_l = kwargs['lambda_l']

			if 'Cp' in kwargs.keys():
				self.Cp = kwargs['Cp']

			if 'Cn' in kwargs.keys():
				self.Cn = kwargs['Cn']

			if 'epsilon' in kwargs.keys():
				self.epsilon = kwargs['epsilon']

			if 'cgitermax' in kwargs.keys():
				self.cgitermax = kwargs['cgitermax']

			if 'mfnitermax' in kwargs.keys():
				self.mfnitermax = kwargs['mfnitermax']

	def set_defaults(self):
		self.lambda_l = 1.0
		self.Cp = 1.0 
		self.Cn = 1.0 
		self.epsilon = 1e-7
		self.cgitermax = 10000
		self.mfnitermax = 50

	def __str__(self):
		s = ''
		attrs = options._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs)
		for attr, val in zip(attrs, values):
			s += ('%s: %s\n'%(attr, val))
		s.strip()
		return s


fillprototype(libssl.call_L2_SVM_MFN, None, [POINTER(data), POINTER(options), POINTER(vector_double), POINTER(vector_double), c_int, c_double, c_double])
fillprototype(libssl.init_vec_double, None, [POINTER(vector_double), c_int, c_double])
fillprototype(libssl.init_vec_int, None, [POINTER(vector_int), c_int])
fillprototype(libssl.clear_vec_double, None, [POINTER(vector_double)])
fillprototype(libssl.clear_vec_int, None, [POINTER(vector_int)])

def solver(X, y, verbose, **kwargs):
	""" Set up data structures and call optimized L2-SVM-MFN function.  Note that to make the data 
	transfer of the numpy feature matrix to a flat ctype array as fast as possible, the L2-SVM-MFN 
	source assumes the bias is not represented as a column of ones in the passed-in feature matrix, 
	i.e., the bias term is handled separately whenever the passed in feature matrix X (called set 
	the C++ L2_SVM_MFN function) is directly accessed.
	"""
	# check y
	if not isinstance(y, np.ndarray):
		if not isinstance(y, list):
			raise ValueError('y must be an iterable type (list, numpy.ndarray)')
		else:
			y = np.array(y)
	else:
		if np.prod(y.shape) != y.shape[0]:
			raise ValueError('y must be a column or row vector')

	# Note: disable check below, assume this is handled in the main PIN parser
	# # check y
	# labels = set(y)
	# assert set(y) == set([1.0,-1.0]), 'label array must contain positive(+1) and negative (-1) samples'

	# check x vs. y
	if not isinstance(X, np.ndarray):
		raise ValueError('X and y must be numpy.ndarray')
	elif X.shape[0] != y.shape[0]:
		raise ValueError('X and y must have  the same number of samples')

	ssl_data = data()
	ssl_weights = vector_double()
	ssl_options = options(**kwargs)
	ssl_data.from_data(X,y)
	ssl_outputs = vector_double()
	libssl.call_L2_SVM_MFN(ssl_data, ssl_options, ssl_weights, ssl_outputs, verbose, ssl_options.Cp, ssl_options.Cn)
	
	clf = np.array(np.fromiter(ssl_weights.vec, dtype=np.float64, count=ssl_weights.d))

	libssl.clear_vec_double(ssl_outputs)
	libssl.clear_vec_double(ssl_weights)

	return clf
