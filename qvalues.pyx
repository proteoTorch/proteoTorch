# distutils: language=c++

from libcpp.vector cimport vector

import operator

_includeNegativesInResult=True
_scoreInd=0
_labelInd=1

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

## This is a reimplementation of Uri Keich's code written in R.
#
# Assumes that scores are sorted in descending order
#
# If pi0 == 1.0 this is equal to the "traditional" q-value calculation
##
# Assumes that scores are sorted in descending order
def getMixMaxCounts(combined, h_w_le_z, h_z_le_z):
    """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
    """
    cnt_z = 0
    cnt_w = 0
    queue = 0
    for idx in range(len(combined)-1, -1, -1):
        if(combined[idx][1]==1):
            cnt_w += 1
        else:
            cnt_z += 1
            queue += 1
        if idx == 0 or combined[idx][0] != combined[idx-1][0]:
            for i in range(queue):
                h_w_le_z.append(float(cnt_w))
                h_z_le_z.append(float(cnt_z))
            queue = 0

def getQValues(double pi0, combined, 
    unsigned int numPsms,
    skipDecoysPlusOne = False, int verb = -1):
    """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
    """
    cdef vector[double] qvals
    qvals.reserve(numPsms)
    cdef unsigned int scoreInd = _scoreInd
    cdef unsigned int labelInd = _labelInd

    # h_w_le_z = [] # N_{w<=z} and N_{z<=z}
    # h_z_le_z = []
    cdef vector[int] h_w_le_z # N_{w<=z} and N_{z<=z}
    cdef vector[int] h_z_le_z
    cdef unsigned int countTotal = 0
    cdef unsigned int n_z_ge_w = 0
    cdef unsigned int n_w_ge_w = 0
    cdef unsigned int queue = 0

    if pi0 < 1.0:
        # getMixMaxCounts(combined, h_w_le_z, h_z_le_z)
        for idx in range(len(combined)-1, -1, -1):
            if(combined[idx][1]==1):
                n_w_ge_w += 1
            else:
                n_z_ge_w += 1
                queue += 1
            if idx == 0 or combined[idx][0] != combined[idx-1][0]:
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
    n_z_ge_w = 1
    n_w_ge_w = 0 # N_{z>=w} and N_{w>=w}
    if skipDecoysPlusOne:
        n_z_ge_w = 0

    cdef unsigned int decoyQueue = 0 # handles ties
    cdef unsigned int targetQueue = 0
    for idx in range(len(combined)):
        if combined[idx][labelInd] == 1:
            n_w_ge_w += 1
            targetQueue += 1
        else:
            n_z_ge_w += 1
            decoyQueue += 1

        if idx==len(combined)-1 or combined[idx][scoreInd] != combined[idx+1][scoreInd]:
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
    # allScores: list of triples consisting of score, label, and index
    allScores = zip(scores,labels, range(len(scores)))
    #--- sort descending
    allScores.sort(reverse=True)
    pi0 = 1.
    qvals = getQValues(pi0, allScores, len(allScores),
    skipDecoysPlusOne, verb)
    
    taq = []
    daq = []
    for idx, q in enumerate(qvals):
        if q > thresh:
            break
        else:
            curr_label = allScores[idx][1]
            curr_og_idx = allScores[idx][2]
            if curr_label == 1:
                taq.append(curr_og_idx)
            else:
                daq.append(curr_og_idx)
    return taq,daq, [qvals[i] for _,_,i in allScores]

def calcQAndNumIdentified(scores, labels, thresh = 0.01, skipDecoysPlusOne = False):
    """Returns q-values and the number of identified spectra at each q-value
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    # allScores: list of triples consisting of score, label, and index
    allScores = zip(scores,labels, range(len(scores)))
    #--- sort descending
    allScores.sort(reverse=True)
    pi0 = 1.
    qvals = getQValues(pi0, allScores, skipDecoysPlusOne)
    
    posTot = 0
    ps = []
    for idx, q in enumerate(qvals):
        curr_label = allScores[idx][1]
        curr_og_idx = allScores[idx][2]
        if curr_label == 1:
            posTot += 1
        ps.append(posTot)
    return qvals, ps
