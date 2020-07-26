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
	l: number of labeled examples
	u: number of unlabeled examples
	n: number of features
	X: flattened 2D feature matrix
	Y: labels
	C: cost associated with each examples
	'''
	_names = ['m', 'l', 'u', 'n', 'X', 'Y', 'C']
	_types = [c_int, c_int, c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double) ]
	_fields_ = genFields(_names, _types)

	def __str__(self):
		s = ''
		attrs = data._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs)
		for attr, val in zip(attrs, values):
			s += ('%s: %s\n' % (attr, val))
		s.strip()

		return s

	def from_data(self, X, y, cp = 1., cn = 1.):
		self.__frombuffer__ = False
		# set constants
		self.m = len(y)
		self.l = sum(y != 0)
		self.u = self.m - self.l
		self.n = X.shape[1]+1 # include bias term

		# allocate memory
		self.X = (c_double * (self.n*self.m))()
		self.Y = (c_double * self.m)()
		# self.C = (c_double * self.m)()

		idx = 0
		# copying data
		for i in range(self.m):
			for j in range(self.n-1):
				self.X[idx] = X[i,j]
				idx += 1
			self.X[idx] = 1.
			idx += 1

		# set labels
		for i,v in enumerate(y):
			self.Y[i] = v
			# if v == 1:
			# 	self.C[i] = cp
			# elif v== -1:
			# 	self.C[i] = cn
			# else:
			# 	self.C[i] = 1.0
	def __init__(self):
		self.__createfrom__ = 'python'
		self.__frombuffer__ = True

	def dump(self, filename):
		with open(filename, 'wt') as fout:
			for j in range(self.m):
				# write label
				fout.write('%d\t' % (self.Y[j]))

				# write non-zero indices
				start_ix = self.rowptr[j]
				stop_ix = self.rowptr[j+1]
				for i in range(start_ix, stop_ix-1):
					fout.write('%d:%2.4f ' % (self.colind[i]+1, self.val[i]))
					fout.write('\n')

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
	_names = ['algo', 'lambda_l', 'lambda_u', 'S', 'R', 'Cp', 'Cn', 'epsilon', 'cgitermax', 'mfnitermax']
	_types = [c_int, c_double, c_double, c_int, c_double, c_double, c_double, c_double, c_int, c_int]
	_fields_ = genFields(_names, _types)

	def __init__(self, **kwargs):
		self.set_defaults()

		if kwargs:
			if 'algo' in kwargs.keys():
				self.algo = kwargs['algo']

			if 'lambda_l' in kwargs.keys():
				self.lambda_l = kwargs['lambda_l']

			if 'lambda_u' in kwargs.keys():
				self.lambda_u = kwargs['lambda_u']

			if 'S' in kwargs.keys():
				self.S = kwargs['S']

			if 'R' in kwargs.keys():
				self.R = kwargs['R']

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
		self.algo = 1
		self.lambda_l = 1.0
		self.lambda_u = 1.0
		self.S = 10000
		self.R = 0.5 
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
	libssl.call_L2_SVM_MFN(ssl_data, ssl_options, ssl_weights, ssl_outputs, verbose, ssl_options.Cp, ssl_options.Cn)
	
	clf = np.array(np.fromiter(ssl_weights.vec, dtype=np.float64, count=ssl_weights.d))

	libssl.clear_vec_double(ssl_outputs)
	libssl.clear_vec_double(ssl_weights)

	return clf
