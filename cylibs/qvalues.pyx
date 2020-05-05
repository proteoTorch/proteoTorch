# distutils: language=c++

# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2020 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free, qsort
import operator

_includeNegativesInResult=True
_scoreInd=0
_labelInd=1


cdef extern from "stdlib.h":
    void qsort(void *base, int nmemb, int size,
                int(*compar)(const void *, const void *)) nogil

cdef struct psm:
        double score
        int label
        unsigned int index

# cdef int compare(const_void * a, const_void * b):
#     # sort in reverse order
#     cdef double v = ((a)).score - ((b)).score
#     if v < 0:
#         return 1
#     elif v > 0:
#         return -1
#     else:
#         return 0

cdef int compare(const void * pa, const void * pb):
    cdef double a, b 
    a = (<psm *>pa)[0].score
    b = (<psm *>pb)[0].score
    # sort in reverse order
    if a < b:
        return 1
    elif a > b:
        return -1
    else:
        return 0

#########################################################
#########################################################
################### CV-bin score normalization
#########################################################
#########################################################
def qMedianDecoyScore(scores, labels, thresh = 0.01, skipDecoysPlusOne = False):
    """ Returns the minimal score which achieves the specified threshold and the
        median decoy score from the set
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    cdef int numPsms
    numPsms = len(scores)
    # # allScores: list of triples consisting of score, label, and index
    # allScores = zip(scores,labels, range(len(scores)))
    # #--- sort descending
    # allScores.sort(reverse=True)
    cdef psm *allScores  = <psm *> malloc(numPsms * sizeof(psm))
    cdef unsigned int idx
    for idx in range(numPsms):
        allScores[idx].score = scores[idx]
        allScores[idx].label = labels[idx]
        allScores[idx].index = idx
    # psmsort(allScores, numPsms)
    qsort(allScores, numPsms, sizeof(psm), compare)
    cdef double pi0 = 1.
    cdef int sdpo = 0
    if skipDecoysPlusOne:
        sdpo = 1
    qvals = getQValues(pi0, allScores, numPsms,
                       sdpo)

    # Calculate minimum score which achieves q-value thresh
    u = allScores[0].score
    for idx in range(numPsms):
        q = qvals[idx]
    # for idx, q in enumerate(qvals):
        if q > thresh:
            break
        u = allScores[idx].score

    # find median decoy score
    d = allScores[0].score + 1.
    dScores = sorted([score for score,l in zip(scores,labels) if l != 1])
    if len(dScores):
        d = dScores[max(0,len(dScores) / 2)]
    free(allScores)
    return u, d

#########################################################
#########################################################
################### Q-value estimation functions
#########################################################
#########################################################
# From itertools, https://docs.python.org/3/library/itertools.html#itertools.accumulate
def accumulate(iterable, func=operator.add, initial=None):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total = func(total, element)
        yield total

###### TODO: add calculation of pi0 for q-value re-estimation after PSM rescoring
# def findpi0():
cdef getQValues(double pi0, psm* combined, 
                unsigned int numPsms,
                int skipDecoysPlusOne = 0, int verb = -1):
    """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
    """
    cdef vector[double] qvals
    qvals.reserve(numPsms)
    cdef unsigned int idx

    cdef vector[int] h_w_le_z # N_{w<=z} and N_{z<=z}
    cdef vector[int] h_z_le_z
    cdef unsigned int countTotal = 0
    cdef unsigned int n_z_ge_w = 0
    cdef unsigned int n_w_ge_w = 0
    cdef unsigned int queue = 0

    if pi0 < 1.0:
        for idx in range(numPsms-1, -1, -1):
            if(combined[idx].label==1):
                n_w_ge_w += 1
            else:
                n_z_ge_w += 1
                queue += 1
            if idx == 0 or combined[idx].scoe != combined[idx-1].score:
                for i in range(queue):
                    h_w_le_z.push_back(n_w_ge_w)
                    h_z_le_z.push_back(n_z_ge_w)
                    countTotal += 1
                queue = 0

    cdef double estPx_lt_zj = 0.
    cdef double E_f1_mod_run_tot = 0.0
    cdef double fdr = 0.0
    cdef double cnt_z = 0
    cdef double cnt_w = 0
    cdef int j = 0
    n_z_ge_w = 1
    n_w_ge_w = 0 # N_{z>=w} and N_{w>=w}
    if skipDecoysPlusOne:
        n_z_ge_w = 0

    cdef unsigned int decoyQueue = 0 # handles ties
    cdef unsigned int targetQueue = 0
    for idx in range(numPsms):
        if combined[idx].label == 1:
            n_w_ge_w += 1
            targetQueue += 1
        else:
            n_z_ge_w += 1
            decoyQueue += 1

        if idx==numPsms-1 or combined[idx].score != combined[idx+1].score:
            if pi0 < 1.0 and decoyQueue > 0:
                j = countTotal - (n_z_ge_w - 1)
                cnt_w = float(h_w_le_z[j])
                cnt_z = float(h_z_le_z[j])
                estPx_lt_zj = (cnt_w - pi0*cnt_z) / ((1.0 - pi0)*cnt_z)

                if estPx_lt_zj > 1.:
                    estPx_lt_zj = 1.
                if estPx_lt_zj < 0.:
                    estPx_lt_zj = 0.
                E_f1_mod_run_tot += float(decoyQueue) * estPx_lt_zj * (1.0 - pi0)
                if verb >= 3:
                    print "Mix-max num negatives correction: %f vs. %f" % ((1.0 - pi0)*float(n_z_ge_w), E_f1_mod_run_tot)

            if _includeNegativesInResult:
                targetQueue += decoyQueue

            fdr = (n_z_ge_w * pi0 + E_f1_mod_run_tot) / float(max(1, n_w_ge_w))
            for i in range(targetQueue):
                qvals.push_back(min(fdr,1.))
            decoyQueue = 0
            targetQueue = 0

    # Convert the FDRs into q-values.
    # Below is equivalent to: partial_sum(qvals.rbegin(), qvals.rend(), qvals.rbegin(), min);
    return list(accumulate(qvals[::-1], min))[::-1]
    
def calcQ(scores, labels, thresh = 0.01, skipDecoysPlusOne = False,
          verb = -1):
    """Returns q-values and the indices of the positive class such that q <= thresh
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    cdef int numPsms
    numPsms = len(scores)
    # # allScores: list of triples consisting of score, label, and index
    # allScores = zip(scores,labels, range(len(scores)))
    # #--- sort descending
    # allScores.sort(reverse=True)
    # replace allScores 
    cdef psm *allScores  = <psm *> malloc(numPsms * sizeof(psm))
    cdef unsigned int idx
    for idx in range(numPsms):
        allScores[idx].score = scores[idx]
        allScores[idx].label = labels[idx]
        allScores[idx].index = idx
    # psmsort(allScores, numPsms)
    qsort(allScores, numPsms, sizeof(psm), compare)
    cdef double pi0 = 1.
    cdef int sdpo = 0
    if skipDecoysPlusOne:
        sdpo = 1
    qvals = getQValues(pi0, allScores, numPsms,
                       sdpo, verb)
    
    taq = []
    daq = []
    for idx in range(numPsms):
        q = qvals[idx]
        if q > thresh:
            break
        else:
            curr_label = allScores[idx].label
            curr_og_idx = allScores[idx].index
            if curr_label == 1:
                taq.append(curr_og_idx)
            else:
                daq.append(curr_og_idx)
    qvals = [qvals[allScores[idx].index] for i in range(numPsms)]
    free(allScores)
    return taq,daq, qvals

def calcQAndNumIdentified(scores, labels, thresh = 0.01, skipDecoysPlusOne = False, verb = -1):
    """Returns q-values and the number of identified spectra at each q-value
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    cdef int numPsms
    numPsms = len(scores)
    # # allScores: list of triples consisting of score, label, and index
    # allScores = zip(scores,labels, range(len(scores)))
    # #--- sort descending
    # allScores.sort(reverse=True)
    # pi0 = 1.
    # qvals = getQValues(pi0, allScores, skipDecoysPlusOne)
    cdef psm *allScores  = <psm *> malloc(numPsms * sizeof(psm))
    cdef unsigned int idx
    for idx in range(numPsms):
        allScores[idx].score = scores[idx]
        allScores[idx].label = labels[idx]
        allScores[idx].index = idx
    # psmsort(allScores, numPsms)
    qsort(allScores, numPsms, sizeof(psm), compare)
    cdef double pi0 = 1.
    qvals = getQValues(pi0, allScores, numPsms,
                       skipDecoysPlusOne, verb)
    
    posTot = 0
    ps = []
    for idx in range(numPsms):
        q = qvals[idx]
        curr_label = allScores[idx].label
        curr_og_idx = allScores[idx].index
        if curr_label == 1:
            posTot += 1
        ps.append(posTot)
    free(allScores)
    return qvals, ps
