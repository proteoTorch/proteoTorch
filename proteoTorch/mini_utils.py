"""
Written by Gregor Urban <gur9000@outlook.com> (and John Halloran <jthalloran@ucdavis.edu>)

Copyright (C) 2020 Gregor Urban and John Halloran
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0
"""
import time
import numpy as np
from os import makedirs as _makedirs
from os.path import exists as _exists


from proteoTorch_qvalues import calcQAndNumIdentified, numIdentifiedAtQ
# try:
#    from proteoTorch_qvalues import calcQAndNumIdentified, numIdentifiedAtQ
# except:
#    from pyfiles.qvalsBase import calcQAndNumIdentified, numIdentifiedAtQ
#####################
### Generic Functions
#####################

def mkdir(path):
    if len(path)>1 and _exists(path)==0:
        _makedirs(path)


def softmax(x):
    """Compute softmax values for each sets of scores in x. (numpy)
    """
    return np.exp(x) / (np.sum(np.exp(x), axis=1)[:, None] + 1e-10)


def binary_search(sorted_data, target):
    '''
    Returns index of match, if no perfect match is found then the index of the closest match is returned.
    '''
    lower = 0
    upper = len(sorted_data)
    while lower < upper:
        x = lower + (upper - lower) // 2
        val = sorted_data[x]
        if target == val:
            return x
        elif target > val:
            if lower == x:
                break
            lower = x
        elif target < val:
            upper = x
    if upper == len(sorted_data):
        return lower
    return lower if abs(sorted_data[lower] - target) <= abs(sorted_data[upper] - target) else upper



def TimeStamp():
    """can be used inside of file names"""
    return time.strftime("%m-%d-%Y__%Hh_%Mm_%Ss_", time.localtime(time.time()))+str(time.time()%1)[10:]


#########################
### MS-Specific Functions
#########################


def calcQCompetition_v2(predictions, labels):
    """Calculates P vs q xy points from arrays"""
    if labels.ndim==2:
        labels = np.argmax(labels, axis=1)
    if predictions.ndim==2:
        predictions = predictions[:,1] #softmax() already applied #[:, 1] - predictions[:, 0]

    qs, ps = calcQAndNumIdentified(predictions, labels)
    return np.asarray(qs, 'float32'), np.asarray(ps, 'float32')

def numIdentifiedAtQ_v2(predictions, labels, thresh = 0.002):
    """Calculates P vs q xy points from arrays"""
    if labels.ndim==2:
        labels = np.argmax(labels, axis=1)
    if predictions.ndim==2:
        predictions = predictions[:,1] #softmax() already applied #[:, 1] - predictions[:, 0]

    ps = numIdentifiedAtQ(predictions, labels, thresh)
    return np.asarray(ps, 'float32'), len(labels)


def AccuracyAtTol(predictions, labels, qTol=0.01):
    ps, numPsms = numIdentifiedAtQ_v2(predictions, labels, qTol)
    return ps[-1] / float(numPsms) * 100
    # qs, ps = calcQCompetition_v2(predictions, labels)
    # idx = binary_search(qs, qTol)
    # return float(ps[idx]) / float(len(qs)) * 100


def AUC_up_to_tol(predictions, labels, qTol=0.005, qCurveCheck = 0.001):
    """
    Re-weighted AUC towards lower q values. Not normalized to 1.
    """
    if labels.ndim==2:
        labels = np.argmax(labels, axis=1)
    qs, ps = calcQCompetition_v2(predictions, labels)
    idx1 = binary_search(qs, qTol)
    idx2 = binary_search(qs, qCurveCheck)
#    den = float(np.sum(labels>=1))
    #print('AUC_upto_Tol: den =',den)
    auc = np.trapz(ps[:idx1])#/den/idx1
    if qTol > qCurveCheck:
        auc = 0.3 * auc + 0.7 * np.trapz(ps[:idx2])#/den/idx2
    return auc


def save_text(fname, string, append = False):
    f=open(fname,'a' if append else 'w')
    f.write(string)
    f.close()
    

def AUC_up_to_tol_singleQ(qTol=0.002):
    """
    Re-weighted AUC towards lower q values. Not normalized to 1.
    
    Returns:
        
        function with inputs: <predictions>, <labels>
    """
    def fn(predictions, labels):
        if labels.ndim==2:
            labels = np.argmax(labels, axis=1)
        ps, _ = numIdentifiedAtQ_v2(predictions, labels, qTol)
        auc = np.trapz(ps)
        return auc
    return fn



