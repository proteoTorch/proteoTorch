#!/usr/bin/env python
#
# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2020 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0

from __future__ import with_statement

import collections
import csv
import itertools
import math
import optparse
import os
import random
import sys
import cPickle as pickle

import operator
from sklearn.utils import check_random_state
from copy import deepcopy
from sklearn.svm import LinearSVC as svc
from sklearn import preprocessing
from pprint import pprint
import util.args
import util.iterables
import struct
import array

from scipy import linalg, stats
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import mixture
from svmlin import svmlin

#########################################################
################### Global variables
#########################################################
_debug=True
_verb=0
_mergescore=True
_includeNegativesInResult=True
_standardNorm=True
_topPsm=False
# General assumed iterators for lists of score tuples
_scoreInd=0
_labelInd=1
_indInd=2 # Used to keep track of feature matrix rows when sorting based on score

def qMedianDecoyScore(scores, labels, thresh = 0.01, skipDecoysPlusOne = False):
    """ Returns the minimal score which achieves the specified threshold and the
        median decoy score from the set
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    scoreInd = _scoreInd
    labelInd = _labelInd
    # allScores: list of triples consisting of score, label, and index
    allScores = zip(scores,labels, range(len(scores)))
    #--- sort descending
    allScores.sort(reverse=True)
    pi0 = 1.
    qvals = getQValues(pi0, allScores, skipDecoysPlusOne)

    # Calculate minimum score which achieves q-value thresh
    u = allScores[0][scoreInd]
    for idx, q in enumerate(qvals):
        if q > thresh:
            break
        u = allScores[idx][scoreInd]

    # find median decoy score
    d = allScores[0][scoreInd] + 1.
    dScores = sorted([score for score,l in zip(scores,labels) if l != 1])
    if len(dScores):
        d = dScores[max(0,len(dScores) / 2)]
    return u, d

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

# def findpi0():
    

## This is a reimplementation of
#   Crux/src/app/AssignConfidenceApplication.cpp::compute_decoy_qvalues_mixmax
# Which itself was a reimplementation of Uri Keich's code written in R.
#
# Assumes that scores are sorted in descending order
#
# If pi0 == 1.0 this is equal to the "traditional" q-value calculation
##
# Assumes that scores are sorted in descending order
def getMixMaxCount(combined, h_w_le_z, h_z_le_z):
    """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
    """
    cnt_z = 0
    cnt_w = 0
    queue = 0
    for idx in range(len(combined)-1, -1, -1):
        if(combiined[idx][1]==1):
            cnt_w += 1
        else:
            cnt_z += 1
            queue += 1
        if idx == 0 or combined[idx][0] != combined[idx-1][0]:
            for i in range(queue):
                h_w_le_z.append(float(cnt_w))
                h_z_le_z.append(float(cnt_z))
            queue = 0

def getQValues(pi0, combined, skipDecoysPlusOne = False):
    """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
    """
    qvals = []
    scoreInd = _scoreInd
    labelInd = _labelInd

    h_w_le_z = [] # N_{w<=z} and N_{z<=z}
    h_z_le_z = []
    if pi0 < 1.0:
        getMixMaxCounts(combined, h_w_le_z, h_z_le_z)
    estPx_lt_zj = 0.
    E_f1_mod_run_tot = 0.0
    fdr = 0.0
    n_z_ge_w = 1
    n_w_ge_w = 0 # N_{z>=w} and N_{w>=w}
    if skipDecoysPlusOne:
        n_z_ge_w = 0

    decoyQueue = 0 # handles ties
    targetQueue = 0
    for idx in range(len(combined)):
        if combined[idx][labelInd] == 1:
            n_w_ge_w += 1
            targetQueue += 1
        else:
            n_z_ge_w += 1
            decoyQueue += 1

        if idx==len(combined)-1 or combined[idx][scoreInd] != combined[idx+1][scoreInd]:
            if pi0 < 1.0 and decoyQueue > 0:
                j = len(h_w_le_z) - (n_z_ge_w - 1)
                cnt_w = float(h_w_le_z[j])
                cnt_z = float(h_z_le_z[j])
                estPx_lt_zj = (cnt_w - pi0*cnt_z) / ((1.0 - pi0)*cnt_z)

                if estPx_lt_zj > 1.:
                    estPx_lt_zj = 1.
                if estPx_lt_zj < 0.:
                    estPx_lt_zj = 0.
                E_f1_mod_run_tot += float(decoyQueue) * estPx_lt_zj * (1.0 - pi0)
                if _debug and _verb >= 3:
                    print "Mix-max num negatives correction: %f vs. %f" % ((1.0 - pi0)*float(n_z_ge_w), E_f1_mod_run_tot)

            if _includeNegativesInResult:
                targetQueue += decoyQueue

            fdr = (n_z_ge_w * pi0 + E_f1_mod_run_tot) / float(max(1, n_w_ge_w))
            for i in range(targetQueue):
                qvals.append(min(fdr,1.))
            decoyQueue = 0
            targetQueue = 0
    # Convert the FDRs into q-values.
    # Below is equivalent to: partial_sum(qvals.rbegin(), qvals.rend(), qvals.rbegin(), min);
    return list(accumulate(qvals[::-1], min))[::-1]
    
def calcQ(scores, labels, thresh = 0.01, skipDecoysPlusOne = True):
    """Returns q-values and the indices of the positive class such that q <= thresh
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    # allScores: list of triples consisting of score, label, and index
    allScores = zip(scores,labels, range(len(scores)))
    #--- sort descending
    allScores.sort(reverse=True)
    pi0 = 1.
    qvals = getQValues(pi0, allScores, skipDecoysPlusOne)
    
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

def load_pin_return_featureMatrix(filename, oneHotChargeVector = True):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number

        Normal tide features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	Peptide	Proteins
    """

    with open(filename, 'r') as f:
        r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
        headerInOrder = r.fieldnames
        reader = [l for l in r]
    l = headerInOrder
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    sids = []
    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    maxCharge = 1
    chargeKeys = [] # used as a fast hash check when determining charge integer
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.append(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i.lower() == 'xcorr':
                scoreKey = i            

    constKeys = ["SpecId", "Label", sidKey, "Peptide", "Proteins", "CalcMass", "ExpMass"]
    if not oneHotChargeVector:
        constKeys = set(constKeys + chargeKeys) # exclude these when reserializing data
    else:
        constKeys = set(constKeys) # exclude these when reserializing data

    keys = []
    for h in headerInOrder: # keep order of keys intact
        if h not in constKeys:
            keys.append(h)
    
    featureNames = []
    if not oneHotChargeVector:
        featureNames.append("Charge")
    for k in keys:
        featureNames.append(k)
            
    targets = {}  # mapping between sids and indices in the feature matrix
    decoys = {}

    X = [] # Feature matrix
    Y = [] # labels
    pepstrings = []
    scoreIndex = _scoreInd # column index of the ranking score used by the search algorithm 
    numRows = 0
    
    for i, l in enumerate(reader):
        try:
            sid = int(l[sidKey])
        except ValueError:
            print "Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1)

        charge = 0
        # look for current PSM, encoded as a one-hot vector
        for c in chargeKeys:
            try:
                charge = int(l[c])
            except ValueError:
                print "Could not convert charge %s on line %d to int, exitting" % (l[c], i+1)

            if charge:
                charge = int(c[-1])
                break

        assert charge > 0, "No charge denoted with value 1 or greater for PSM on line %d, exitting" % (i+1)

        try:
            y = int(l["Label"])
        except ValueError:
            print "Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1)
        if y != 1 and y != -1:
            print "Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1)
            exit(-1)

        el = []
        if not oneHotChargeVector:
            el.append(charge)
        for k in keys:
            el.append(float(l[k]))
        
        if not _topPsm:
            X.append(el)
            Y.append(y)
            pepstrings.append(l["Peptide"][2:-2])
            sids.append(sid)
            numRows += 1
        else:
            if y == 1:
                if sid in targets:
                    featScore = X[targets[sid]][scoreIndex]
                    if el[scoreIndex] > featScore:
                        X[targets[sid]] = el
                        pepstrings[targets[sid]] = l["Peptide"][2:-2]
                else:
                    targets[sid] = numRows
                    X.append(el)
                    Y.append(1)
                    pepstrings.append(l["Peptide"][2:-2])
                    sids.append(sid)
                    numRows += 1
            elif y == -1:
                if sid in decoys:
                    featScore = X[decoys[sid]][scoreIndex]
                    if el[scoreIndex] > featScore:
                        X[decoys[sid]] = el
                        pepstrings[decoys[sid]] = l["Peptide"][2:-2]
                else:
                    decoys[sid] = numRows
                    X.append(el)
                    Y.append(-1)
                    pepstrings.append(l["Peptide"][2:-2])
                    sids.append(sid)
                    numRows += 1

    # # Standard-normalize the feature matrix
    # m= np.mean(X, axis=0)
    # s= np.std(X, axis=0)
    # for i,sd in enumerate(s):
    #     if sd <= 0.:
    #         s[i] = 1.
    # if _debug and _verb >= 2:
    #     print "Feature\tmean\tstd"
    #     for i,j,k in zip(featureNames, m, s):
    #         print "%s\t%.2e\t%.2e" % (i,j,k)
    # X = (np.array(X) - m) / s
    # return pepstrings, X, np.array(Y), featureNames, sids

    if _standardNorm:
        return pepstrings, preprocessing.scale(np.array(X)), np.array(Y), featureNames, sids
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        return pepstrings, min_max_scaler.fit_transform(np.array(X)), np.array(Y), featureNames, sids

def sortRowIndicesBySid(sids):
    """ Sort Scan Identification (SID) keys and retain original row indices of feature matrix X
    """
    keySids = sorted(zip(sids, range(len(sids))))
    xRowIndices = [j for (_,j) in keySids]
    sids = [i for (i,_) in  keySids]
    return sids, xRowIndices

def findInitDirection(X, Y, thresh, featureNames):
    l = X.shape
    m = l[1] # number of columns/features
    initDirection = -1
    numIdentified = -1
    # TODO: add check verifying best direction idetnfies more than -1 spectra, otherwise something
    # went wrong
    negBest = False
    for i in range(m):
        scores = X[:,i]
        # Check scores multiplied by both 1 and positive -1
        for checkNegBest in range(2):
            if checkNegBest==1:
                taq, _, _ = calcQ(-1. * scores, Y, thresh, True)
            else:
                taq, _, _ = calcQ(scores, Y, thresh, True)
            if len(taq) > numIdentified:
                initDirection = i
                numIdentified = len(taq)
                negBest = checkNegBest==1
            if _debug and _verb >= 2:
                if checkNegBest==1:
                    print "Direction -%d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq))
                else:
                    print "Direction %d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq))
    return initDirection, numIdentified, negBest

def getDecoyIdx(labels, ids):
    return [i for i in ids if labels[i] != 1]

def searchForInitialDirection(keys, X, Y, q, featureNames):
    """ Iterate through cross validation training sets and find initial search directions
        Returns the scores for the disjoint bins
    """
    initTaq = 0.
    scores = np.zeros(Y.shape)
    # split dataset into thirds for testing/training
    m = len(keys)/3
    for kFold in range(3):
        if kFold < 2:
            testSids = keys[kFold * m : (kFold+1) * m]
        else:
            testSids = keys[kFold * m : ]

        trainSids = list(set(keys) - set(testSids))

        # Find initial direction
        initDir, numIdentified, negBest = findInitDirection(X[trainSids], Y[trainSids], q, featureNames)
        initTaq += numIdentified
        print "CV fold %d: could separate %d PSMs in initial direction %d, %s" % (kFold, numIdentified, initDir, featureNames[initDir])
        scores[trainSids] = -1. * X[trainSids,initDir]
    return scores, initTaq

def doLda(thresh, keys, scores, X, Y):
    """ Perform LDA on CV bins
    """
    totalTaq = 0 # total number of estimated true positives at q-value threshold
    # split dataset into thirds for testing/training
    m = len(keys)/3
    # record new scores as we go
    newScores = np.zeros(scores.shape)
    for kFold in range(3):
        if kFold < 2:
            testSids = keys[kFold * m : (kFold+1) * m]
        else:
            testSids = keys[kFold * m : ]

        trainSids = list(set(keys) - set(testSids))

        taq, daq, _ = calcQ(scores[trainSids], Y[trainSids], thresh, True)
        # Debugging check
        if _debug and _verb >= 1:
            gd = getDecoyIdx(Y, trainSids)
            print "CV fold %d: |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d" % (kFold, len(trainSids) - len(gd), len(gd), len(taq), len(daq))

        trainSids = list(set(taq) | set(getDecoyIdx(Y, trainSids)))

        features = X[trainSids]
        labels = Y[trainSids]
        clf = lda()
        clf.fit(features, labels)

        iter_scores = clf.decision_function(X[testSids])
        # Calculate true positives
        tp, _, _ = calcQ(iter_scores, Y[testSids], thresh, False)
        totalTaq += len(tp)

        # if _mergescore:
        #     u, d = qMedianDecoyScore(iter_scores, Y[testSids], thresh = 0.01)
        #     iter_scores = (iter_scores - u) / (u-d)

        for i, score in zip(testSids, iter_scores):
            newScores[i] = score

    return newScores, totalTaq

def givenInitialDirection_split(keys, X, Y, q, featureNames, initDir):
    """ Iterate through cross validation training sets and find initial search directions
        Returns the scores for the disjoint bins
    """
    initTaq = 0.
    scores = []
    # Add check for whether scores multiplied by +1 or -1 is best
    # split dataset into thirds for testing/training
    for kFold, trainSids in enumerate(keys):
        currScores = X[trainSids,initDir]
        numIdentified = -1
        negBest = False
        # Check scores multiplied by both 1 and positive -1
        for checkNegBest in range(2):
            if checkNegBest==1:
                taq, _, _ = calcQ(-1. * currScores, Y[trainSids], q, True)
            else:
                taq, _, _ = calcQ(currScores, Y[trainSids], q, True)
            if len(taq) > numIdentified:
                numIdentified = len(taq)
                negBest = checkNegBest==1

        initTaq += numIdentified
        if negBest:
            print "CV fold %d: could separate %d PSMs in supplied initial direction -%d, %s" % (kFold, numIdentified, initDir, featureNames[initDir])
            scores.append(-1. * currScores)
        else:
            print "CV fold %d: could separate %d PSMs in supplied initial direction %d, %s" % (kFold, numIdentified, initDir, featureNames[initDir])
            scores.append(currScores)
    return scores, initTaq

def searchForInitialDirection_split(keys, X, Y, q, featureNames):
    """ Iterate through cross validation training sets and find initial search directions
        Returns the scores for the disjoint bins
    """
    initTaq = 0.
    scores = []
    kFold = 0
    for trainSids in keys:
        # Find initial direction
        initDir, numIdentified, negBest = findInitDirection(X[trainSids], Y[trainSids], q, featureNames)

        initTaq += numIdentified
        if negBest:
            print "CV fold %d: could separate %d PSMs in initial direction -%d, %s" % (kFold, numIdentified, initDir, featureNames[initDir])
            scores.append(-1. * X[trainSids,initDir])
        else:
            print "CV fold %d: could separate %d PSMs in initial direction %d, %s" % (kFold, numIdentified, initDir, featureNames[initDir])
            scores.append(X[trainSids,initDir])
        kFold += 1
    return scores, initTaq

def doMergeScores(thresh, testSets, scores, Y):
    # record new scores as we go
    newScores = np.zeros(scores.shape)
    for testSids in testSets:
        u, d = qMedianDecoyScore(scores[testSids], Y[testSids], thresh)
        diff = u - d
        if diff <= 0.:
            diff = 1.
        for ts in testSids:
            newScores[ts] = (scores[ts] - u) / (u-d)
    return newScores

def writeOutput(output, scores, Y, pepstrings,sids):
    n = len(Y)
    fid = open(output, 'w')
    fid.write("Kind\tSid\tPeptide\tScore\n")
    counter = 0
    for i in range(n):
        sid = sids[i]
        p = pepstrings[i]
        score = scores[i]
        if Y[i] == 1:
            fid.write("t\t%d\t%s\t%f\n"
                      % (sid,p,score))
            counter += 1
        else:
            fid.write("d\t%d\t%s\t%f\n"
                      % (sid,p,score))
            counter += 1
    fid.close()
    print "Wrote %d PSMs" % counter

def doUnsupervisedGmm_1d(thresh, scores, Y, t_scores, d_scores, 
                         target_rowsToSids, decoy_rowsToSids):
    """ Assuming LDA was performed prior to this, or training an unsupervised GMM based on the scores provided in the scores list
    """
    # Train Unsupervised GMM
    clf = mixture.GaussianMixture(n_components=2, covariance_type='diag', init_params = 'random', verbose=2)
    scores = scores.reshape(len(scores), 1)
    newScores = np.zeros(shape(scores))
    clf.fit(scores)
    posterior_scores = clf.predict_proba(scores)
    # have to figure out which class corresponds to targets
    tp, _, _ = calcQ(posterior_scores[:,0], Y, thresh, False)
    taqa = len(tp)
    tp, _, _ = calcQ(posterior_scores[:,1], Y, thresh, False)
    taqb = len(tp)
    target_class_ind = 1
    print taqa, taqb, posterior_scores.shape
    if taqa > taqb:
        target_class_ind = 0
    for i, score in enumerate(posterior_scores):
        newScores[i] = score[target_class_ind]
        if Y[i] == 1:
            sid = target_rowsToSids[i]
            t_scores[sid] = score[target_class_ind]
        else:
            sid = decoy_rowsToSids[i]
            d_scores[sid] = score[target_class_ind]
    # Calculate true positives
    tp, _, _ = calcQ(scores, Y, thresh, False)
    totalTaq = len(tp)
    return newScores, totalTaq

def doUnsupervisedGmm(thresh, scores, X, Y, t_scores, d_scores, 
                      target_rowsToSids, decoy_rowsToSids):
    """ Assuming LDA was performed prior to this, or training an unsupervised GMM based on the scores provided in the scores list
    """
    newScores = np.zeros(shape(scores))
    # Train Unsupervised GMM
    clf = mixture.GaussianMixture(n_components=2, covariance_type='diag', init_params = 'random', verbose=2)
    clf.fit(X)
    posterior_scores = clf.predict_proba(X)
    # have to figure out which class corresponds to targets
    tp, _, _ = calcQ(posterior_scores[:,0], Y, thresh, False)
    taqa = len(tp)
    tp, _, _ = calcQ(posterior_scores[:,1], Y, thresh, False)
    taqb = len(tp)
    target_class_ind = 1
    print taqa, taqb, posterior_scores.shape
    if taqa > taqb:
        target_class_ind = 0
    for i, score in enumerate(posterior_scores):
        newScores[i] = score[target_class_ind]
        if Y[i] == 1:
            sid = target_rowsToSids[i]
            t_scores[sid] = score[target_class_ind]
        else:
            sid = decoy_rowsToSids[i]
            d_scores[sid] = score[target_class_ind]
    # Calculate true positives
    tp, _, _ = calcQ(scores, Y, thresh, False)
    totalTaq = len(tp)
    return newScores, totalTaq

def ldaGmm(options, output):
    """ Perform LDA followed rescoring using GMM posteriors
    """
    q = 0.01
    f = options.pin
    # target_rows: dictionary mapping target sids to rows in the feature matrix
    # decoy_rows: dictionary mapping decoy sids to rows in the feature matrix
    # X: standard-normalized feature matrix
    # Y: binary labels, true denoting a target PSM
    target_rows, decoy_rows, pepstrings, X, Y, featureNames = load_pin_return_featureMatrix(f)
    # get mapping from rows to spectrum ids
    target_rowsToSids = {}
    decoy_rowsToSids = {}
    for tr in target_rows:
        target_rowsToSids[target_rows[tr]] = tr
    for dr in decoy_rows:
        decoy_rowsToSids[decoy_rows[dr]] = dr
    l = X.shape
    n = l[0] # number of instances
    m = l[1] # number of features

    print "Loaded %d target and %d decoy PSMS with %d features" % (len(target_rows), len(decoy_rows), l[1])
    keys = range(n)

    random.shuffle(keys)
    t_scores = {}
    d_scores = {}

    initDir = options.initDirection
    if initDir > -1 and initDir < m:
        print "Using specified initial direction %d" % (initDir)
        # Gather scores
        scores = X[:,initDir]
        taq, _, _ = calcQ(scores, Y, q, False)
        print "Direction %d, %s: Could separate %d identifications" % (initDir, featureNames[initDir], len(taq))
    else:
        scores, initTaq = searchForInitialDirection(keys, X, Y, q, featureNames)

    # Perform LDA
    scores, numIdentified = doLda(q, keys, scores, X, Y)
    print "LDA: %d targets identified" % (numIdentified)
    if _mergescore:
        scores = doMergeScores(q, keys, scores, Y)

    # Perform Unsupervised GMM learning over LDA scores, rescore PSMs using 
    # resulting posterior
    scores, numIdentified = doUnsupervisedGmm_1d(q, scores, Y, 
                                                 t_scores, d_scores, 
                                                 target_rowsToSids, decoy_rowsToSids)
    print "GMM: %d targets identified" % (numIdentified)

    writeOutput(output, Y, pepstrings,target_rowsToSids, t_scores,
                decoy_rowsToSids, d_scores)

def gmm(options, output):
    """ Train unsupervised GMM and rescore PSMs using posterior
    """
    q = 0.01
    f = options.pin
    # target_rows: dictionary mapping target sids to rows in the feature matrix
    # decoy_rows: dictionary mapping decoy sids to rows in the feature matrix
    # X: standard-normalized feature matrix
    # Y: binary labels, true denoting a target PSM
    target_rows, decoy_rows, pepstrings, X, Y, featureNames = load_pin_return_featureMatrix(f)
    # get mapping from rows to spectrum ids
    target_rowsToSids = {}
    decoy_rowsToSids = {}
    for tr in target_rows:
        target_rowsToSids[target_rows[tr]] = tr
    for dr in decoy_rows:
        decoy_rowsToSids[decoy_rows[dr]] = dr
    l = X.shape
    n = l[0] # number of instances
    m = l[1] # number of features

    print "Loaded %d target and %d decoy PSMS with %d features" % (len(target_rows), len(decoy_rows), l[1])
    t_scores = {}
    d_scores = {}
    scores = np.zeros(Y.shape)
    # Perform Unsupervised GMM learning over features, rescore PSMs using 
    # resulting posterior
    scores, numIdentified = doUnsupervisedGmm(q, scores, X, Y, 
                                              t_scores, d_scores, 
                                              target_rowsToSids, decoy_rowsToSids)
    print "GMM: %d targets identified" % (numIdentified)

    writeOutput(output, Y, pepstrings,target_rowsToSids, t_scores,
                decoy_rowsToSids, d_scores)

def calculateTargetDecoyRatio(Y):
    # calculate target-decoy ratio for the given training/testing set with labels Y
    numPos = 0
    numNeg = 0
    for y in Y:
        if y==1:
            numPos+=1
        else:
            numNeg+=1

    return float(numPos) / max(1., float(numNeg)), numPos, numNeg

def doTestSvm(thresh, keys, X, Y, ws):
    m = len(keys)/3
    testScores = np.zeros(Y.shape)
    totalTaq = 0
    for kFold in range(3):
        if kFold < 2:
            testSids = keys[kFold * m : (kFold+1) * m]
        else:
            testSids = keys[kFold * m : ]
        w = ws[kFold]
        testScores[testSids] = np.dot(X[testSids], w[:-1]) + w[-1]
        
        # Calculate true positives
        tp, _, _ = calcQ(testScores[testSids], Y[testSids], thresh, False)
        totalTaq += len(tp)
    return testScores, totalTaq

def doTest(thresh, keys, X, Y, ws, svmlin = False):
    m = len(keys)/3
    testScores = np.zeros(Y.shape)
    totalTaq = 0
    kFold = 0
    for testSids in keys:
        w = ws[kFold]
        if svmlin:
            testScores[testSids] = np.dot(X[testSids], w[:-1]) + w[-1]
        else:
            testScores[testSids] = w.decision_function(X[testSids])
        
        # Calculate true positives
        tp, _, _ = calcQ(testScores[testSids], Y[testSids], thresh, False)
        totalTaq += len(tp)
        kFold += 1
    return testScores, totalTaq

def doLdaSingleFold(thresh, kFold, features, labels, validateFeatures, validateLabels):
    """ Perform LDA on a CV bin
    """
    clf = lda()
    clf.fit(features, labels)
    validation_scores = clf.decision_function(validateFeatures)
    tp, _, _ = calcQ(validation_scores, validateLabels, thresh, True)
    print "CV finished for fold %d: %d targets identified" % (kFold, len(tp))
    return validation_scores, len(tp), clf

def doSvmGridSearch(thresh, kFold, features, labels, validateFeatures, validateLabels, 
                    cposes, cfracs, alpha, tron = True, currIter=1):
    bestTaq = -1.
    bestCp = 1.
    bestCn = 1.
    bestClf = []
    # Find cpos and cneg
    for cpos in cposes:
        for cfrac in cfracs:
            cneg = cfrac*cpos
            if tron:
                classWeight = {1: alpha * cpos, -1: alpha * cneg}
                clf = svc(dual = False, fit_intercept = True, class_weight = classWeight, tol = 1e-7)
                clf.fit(features, labels)
                validation_scores = clf.decision_function(validateFeatures)
            else:
                clf = svmlin.ssl_train_with_data(features, labels, 0, Cn = alpha * cneg, Cp = alpha * cpos)
                # if currIter ==0:
                #     if kFold==0:
                #         clf = np.array([-0.0259891,0,0.122862,1.99297,-0.205812,0.131589,-0.240002,0.0422103,0.159012,-0.172794,0,0,0,-0.204852,0.00944616,-0.00376582,-3.02847])
                #     elif kFold == 1:
                #         clf = np.array([0.0218895,0,0.161915,2.26042,-0.29484,0.21187,-0.0500683,0.0707332,0.258911,-0.282006,0,0,0,-0.0980667,0.0239779,-0.0278971,-3.75857])
                #     else:
                #         clf = np.array([0.017683,0,0.152919,2.23658,-0.288384,0.248982,-0.350645,0.0369419,0.195835,-0.207899,0,0,0,-0.358095,-0.0027158,-0.0151766,-3.66699])
                # else:
                #     if kFold==0:
                #         clf = np.array([-0.0159667,0,0.258948,2.35891,-0.427149,0.294673,-0.508974,0.110175,0.301756,-0.337725,0,0,0,-0.458718,0.019466,-0.0285029,-4.06692])
                #     elif kFold == 1:
                #         clf = np.array([0.0472272,0,0.284314,2.71909,-0.523136,0.397579,-0.120766,0.139299,0.425898,-0.471377,0,0,0,-0.211426,0.0274639,-0.0304848,-4.05817])
                #     else:
                #         clf = np.array([0.0387359,0,0.314553,2.8278,-0.51302,0.434318,-0.65393,0.086333,0.365803,-0.393993,0,0,0,-0.644954,-0.00481508,-0.0170554,-4.6971])
                validation_scores = np.dot(validateFeatures, clf[:-1]) + clf[-1]
            tp, _, _ = calcQ(validation_scores, validateLabels, thresh, True)
            currentTaq = len(tp)
            if _debug and _verb > 1:
                if not tron:
                    print clf
                print "CV fold %d: cpos = %f, cneg = %f separated %d validation targets" % (kFold, alpha * cpos, alpha * cneg, currentTaq)
            if currentTaq > bestTaq:
                topScores = validation_scores[:]
                bestTaq = currentTaq
                bestCp = cpos * alpha
                bestCn = cneg * alpha
                bestClf = deepcopy(clf)
    print "CV finished for fold %d: best cpos = %f, best cneg = %f, %d targets identified" % (kFold, bestCp, bestCn, bestTaq)
    return topScores, bestTaq, bestClf

def doIter(thresh, keys, scores, X, Y,
           targetDecoyRatio, method = 0, currIter=1):
    """ Train a classifier on CV bins.
        Method 0: LDA
        Method 1: SVM, solver TRON
        Method 2: SVM, solver SVMLIN
    """
    totalTaq = 0 # total number of estimated true positives at q-value threshold
    # split dataset into thirds for testing/training
    m = len(keys)/3
    # record new scores as we go
    # newScores = np.zeros(scores.shape)
    newScores = []
    clfs = [] # classifiers
    kFold = 0
    # C for positive and negative classes
    cposes = [10., 1., 0.1]
    cfracs = [targetDecoyRatio, 3. * targetDecoyRatio, 10. * targetDecoyRatio]
    estTaq = 0
    tron = False
    alpha = 1.
    if method==1:
        tron = True
        alpha = 0.5
    for cvBinSids in keys:
        validateSids = cvBinSids
        # Find training set using q-value analysis
        taq, daq, _ = calcQ(scores[kFold], Y[cvBinSids], thresh, True)
        gd = getDecoyIdx(Y, cvBinSids)
        # Debugging check
        if _debug: #  and _verb >= 1:
            print "CV fold %d: |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d" % (kFold, len(cvBinSids) - len(gd), len(gd), len(taq), len(daq))

        # trainSids = list(set(taq) | set(gd))
        trainSids = gd + taq

        features = X[trainSids]
        labels = Y[trainSids]
        validateFeatures = X[validateSids]
        validateLabels = Y[validateSids]
        if method == 0:
            topScores, bestTaq, bestClf = doLdaSingleFold(thresh, kFold, features, labels, validateFeatures, validateLabels)
        else:
            topScores, bestTaq, bestClf = doSvmGridSearch(thresh, kFold, features, labels,validateFeatures, validateLabels,
                                                          cposes, cfracs, alpha, tron, currIter)
        newScores.append(topScores)
        clfs.append(bestClf)
        estTaq += bestTaq
        kFold += 1
    estTaq /= 2
    return newScores, estTaq, clfs

def doRand():
    global _seed
    _seed=(_seed * 279470273) % 4294967291
    return _seed

def partitionCvBins(featureMatRowIndices, sids, folds = 3, seed = 1):
    trainKeys = []
    testKeys = []
    for i in range(folds):
        trainKeys.append([])
        testKeys.append([])
    remain = []
    r = len(sids) / folds
    remain.append(len(sids) - (folds-1) * r)
    for i in range(folds-1):
        remain.append(len(sids) / folds)
    prevSid = sids[0]
    # seed = doRand(seed)
    randIdx = doRand() % folds
    it = 0
    for k,sid in zip(featureMatRowIndices, sids):
        if (sid!=prevSid):
            # seed = doRand(seed)
            randIdx = doRand() % folds
            while remain[randIdx] <= 0:
                # seed = doRand(seed)
                randIdx = doRand() % folds
        for i in range(folds):
            if i==randIdx:
                testKeys[i].append(k)
            else:
                trainKeys[i].append(k)

        remain[randIdx] -= 1
        prevSid = sid
        it += 1
    return trainKeys, testKeys

def funcIter(options, output):
    q = options.q
    f = options.pin
    # target_rows: dictionary mapping target sids to rows in the feature matrix
    # decoy_rows: dictionary mapping decoy sids to rows in the feature matrix
    # X: standard-normalized feature matrix
    # Y: binary labels, true denoting a target PSM
    oneHotChargeVector = True
    pepstrings, X, Y, featureNames, sids0 = load_pin_return_featureMatrix(f, oneHotChargeVector)
    sids, sidSortedRowIndices = sortRowIndicesBySid(sids0)
    l = X.shape
    n = l[0] # number of instances
    m = l[1] # number of features

    targetDecoyRatio, numT, numD = calculateTargetDecoyRatio(Y)
    print "Loaded %d target and %d decoy PSMS with %d features, ratio = %f" % (numT, numD, l[1], targetDecoyRatio)

    if _debug and _verb >= 3:
        print featureNames
    trainKeys, testKeys = partitionCvBins(sidSortedRowIndices, sids)

    t_scores = {}
    d_scores = {}

    initTaq = 0.
    initDir = options.initDirection
    if initDir > -1 and initDir < m:
        print "Using specified initial direction %d" % (initDir)
        scores, initTaq = givenInitialDirection_split(trainKeys, X, Y, q, featureNames, initDir)
    else:
        scores, initTaq = searchForInitialDirection_split(trainKeys, X, Y, q, featureNames)

    print "Could initially separate %d identifications" % ( initTaq / 2 )
    for i in range(options.maxIters):
        scores, numIdentified, ws = doIter(q, trainKeys, scores, X, Y,
                                           targetDecoyRatio, options.method, i)
        print "iter %d: estimated %d targets <= %f" % (i, numIdentified, q)
    
    isSvmlin = (options.method==2)
    testScores, numIdentified = doTest(q, testKeys, X, Y, ws, isSvmlin)
    print "Identified %d targets <= %f pre-merge." % (numIdentified, q)
    if _mergescore:
        scores = doMergeScores(q, testKeys, testScores, Y)

    taq, _, _ = calcQ(scores, Y, q, False)
    print "Could identify %d targets" % (len(taq))
    writeOutput(output, scores, Y, pepstrings, sids0)

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--q', type = 'float', action= 'store', default = 0.5)
    parser.add_option('--tol', type = 'float', action= 'store', default = 0.01)
    parser.add_option('--initDirection', type = 'int', action= 'store', default = -1)
    parser.add_option('--verb', type = 'int', action= 'store', default = -1)
    parser.add_option('--method', type = 'int', action= 'store', default = 2)
    parser.add_option('--maxIters', type = 'int', action= 'store', default = 10)
    parser.add_option('--pin', type = 'string', action= 'store')
    parser.add_option('--filebase', type = 'string', action= 'store')
    parser.add_option('--seed', type = 'int', action= 'store', default = 1)

    (options, args) = parser.parse_args()

    # Seed random number generator.  To make shuffling nondeterministic, input seed <= -1
    # if options.seed <= -1:
    #     random.seed()
    # else:
    #     random.seed(options.seed)
    _seed=options.seed
    _verb=options.verb
    discOutput = '%s.txt' % (options.filebase)
    funcIter(options, discOutput)
