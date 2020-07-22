import ctypes
import os
import numpy as np
dirname = os.path.dirname(os.path.abspath(__file__))
# libssl = ctypes.CDLL(os.path.join(os.getcwd(), 'libssl.so'))
libssl = ctypes.CDLL(os.path.join(dirname, 'libssl.so'))
# try: 
#         dirname = path.dirname(path.abspath(__file__))
#         libssl = ctypes.CDLL(path.join(dirname, 'libssl.so'))
# except:
#         raise Exception('libssl.so not found')
def L2_SVM_MFN(features, labels, cpos, cneg, verbose = 0):
    l = features.shape
    lambda_l = 1.
    m = l[0] # num rows
    n = l[1] # num columns
    X = (ctypes.c_double * (m*(n+1)))()
    Y = (ctypes.c_double * m)()
    w = (ctypes.c_double * (n+1))()

    libssl.call_L2_SVM_MFN.argtypes = [ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_double, ctypes.c_double,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_int]
    idx = 0
    for i in range(m):
        Y[i] = labels[i]
        for j in range(n):
            X[idx] = features[i,j]
            idx += 1
        X[idx] = 1.
        idx += 1
        libssl.call_L2_SVM_MFN
    libssl.call_L2_SVM_MFN(X, Y, w, cpos, cneg, n+1, m, lambda_l, verbose)
    wout = np.array([w[i] for i in range(n+1)])
    # free(X)
    # free(Y)
    # free(w)
    return wout
