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
import pyximport; pyximport.install()
import qvalues

#########################################################
#########################################################
################### Global variables
#########################################################
#########################################################
_identOutput=False
_debug=True
_verb=0
_mergescore=True
_includeNegativesInResult=True
_standardNorm=True
# Take max wrt sid (TODO: change key from sid to sid+exp_mass)
_topPsm=False
# Check if training has converged over past two iterations
_convergeCheck=False
_reqIncOver2Iters=0.01
# General assumed iterators for lists of score tuples
_scoreInd=0
_labelInd=1
_indInd=2 # Used to keep track of feature matrix rows when sorting based on score

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
    scoreInd = _scoreInd
    labelInd = _labelInd
    # allScores: list of triples consisting of score, label, and index
    allScores = zip(scores,labels, range(len(scores)))
    #--- sort descending
    allScores.sort(reverse=True)
    pi0 = 1.
    qvals = qvalues.getQValues(pi0, allScores, skipDecoysPlusOne)

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

# #########################################################
# #########################################################
# ################### Q-value estimation functions
# #########################################################
# #########################################################
# # From itertools, https://docs.python.org/3/library/itertools.html#itertools.accumulate
# def accumulate(iterable, func=operator.add, initial=None):
#     'Return running totals'
#     # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
#     # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
#     # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
#     it = iter(iterable)
#     total = initial
#     if initial is None:
#         try:
#             total = next(it)
#         except StopIteration:
#             return
#     yield total
#     for element in it:
#         total = func(total, element)
#         yield total

# ###### TODO: add calculation of pi0 for q-value re-estimation after PSM rescoring
# # def findpi0():

# ## This is a reimplementation of Uri Keich's code written in R.
# #
# # Assumes that scores are sorted in descending order
# #
# # If pi0 == 1.0 this is equal to the "traditional" q-value calculation
# ##
# # Assumes that scores are sorted in descending order
# def getMixMaxCounts(combined, h_w_le_z, h_z_le_z):
#     """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
#     """
#     cnt_z = 0
#     cnt_w = 0
#     queue = 0
#     for idx in range(len(combined)-1, -1, -1):
#         if(combiined[idx][1]==1):
#             cnt_w += 1
#         else:
#             cnt_z += 1
#             queue += 1
#         if idx == 0 or combined[idx][0] != combined[idx-1][0]:
#             for i in range(queue):
#                 h_w_le_z.append(float(cnt_w))
#                 h_z_le_z.append(float(cnt_z))
#             queue = 0

# def getQValues(pi0, combined, skipDecoysPlusOne = False):
#     """ Combined is a list of tuples consisting of: score, label, and feature matrix row index
#     """
#     qvals = []
#     scoreInd = _scoreInd
#     labelInd = _labelInd

#     h_w_le_z = [] # N_{w<=z} and N_{z<=z}
#     h_z_le_z = []
#     if pi0 < 1.0:
#         getMixMaxCounts(combined, h_w_le_z, h_z_le_z)
#     estPx_lt_zj = 0.
#     E_f1_mod_run_tot = 0.0
#     fdr = 0.0
#     n_z_ge_w = 1
#     n_w_ge_w = 0 # N_{z>=w} and N_{w>=w}
#     if skipDecoysPlusOne:
#         n_z_ge_w = 0

#     decoyQueue = 0 # handles ties
#     targetQueue = 0
#     for idx in range(len(combined)):
#         if combined[idx][labelInd] == 1:
#             n_w_ge_w += 1
#             targetQueue += 1
#         else:
#             n_z_ge_w += 1
#             decoyQueue += 1

#         if idx==len(combined)-1 or combined[idx][scoreInd] != combined[idx+1][scoreInd]:
#             if pi0 < 1.0 and decoyQueue > 0:
#                 j = len(h_w_le_z) - (n_z_ge_w - 1)
#                 cnt_w = float(h_w_le_z[j])
#                 cnt_z = float(h_z_le_z[j])
#                 estPx_lt_zj = (cnt_w - pi0*cnt_z) / ((1.0 - pi0)*cnt_z)

#                 if estPx_lt_zj > 1.:
#                     estPx_lt_zj = 1.
#                 if estPx_lt_zj < 0.:
#                     estPx_lt_zj = 0.
#                 E_f1_mod_run_tot += float(decoyQueue) * estPx_lt_zj * (1.0 - pi0)
#                 if _debug and _verb >= 3:
#                     print "Mix-max num negatives correction: %f vs. %f" % ((1.0 - pi0)*float(n_z_ge_w), E_f1_mod_run_tot)

#             if _includeNegativesInResult:
#                 targetQueue += decoyQueue

#             fdr = (n_z_ge_w * pi0 + E_f1_mod_run_tot) / float(max(1, n_w_ge_w))
#             for i in range(targetQueue):
#                 qvals.append(min(fdr,1.))
#             decoyQueue = 0
#             targetQueue = 0
#     # Convert the FDRs into q-values.
#     # Below is equivalent to: partial_sum(qvals.rbegin(), qvals.rend(), qvals.rbegin(), min);
#     return list(accumulate(qvals[::-1], min))[::-1]
    
# def calcQ(scores, labels, thresh = 0.01, skipDecoysPlusOne = False):
#     """Returns q-values and the indices of the positive class such that q <= thresh
#     """
#     assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
#     # allScores: list of triples consisting of score, label, and index
#     allScores = zip(scores,labels, range(len(scores)))
#     #--- sort descending
#     allScores.sort(reverse=True)
#     pi0 = 1.
#     qvals = getQValues(pi0, allScores, skipDecoysPlusOne)
    
#     taq = []
#     daq = []
#     for idx, q in enumerate(qvals):
#         if q > thresh:
#             break
#         else:
#             curr_label = allScores[idx][1]
#             curr_og_idx = allScores[idx][2]
#             if curr_label == 1:
#                 taq.append(curr_og_idx)
#             else:
#                 daq.append(curr_og_idx)
#     return taq,daq, [qvals[i] for _,_,i in allScores]

# def calcQAndNumIdentified(scores, labels, thresh = 0.01, skipDecoysPlusOne = False):
#     """Returns q-values and the number of identified spectra at each q-value
#     """
#     assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
#     # allScores: list of triples consisting of score, label, and index
#     allScores = zip(scores,labels, range(len(scores)))
#     #--- sort descending
#     allScores.sort(reverse=True)
#     pi0 = 1.
#     qvals = getQValues(pi0, allScores, skipDecoysPlusOne)
    
#     posTot = 0
#     ps = []
#     for idx, q in enumerate(qvals):
#         curr_label = allScores[idx][1]
#         curr_og_idx = allScores[idx][2]
#         if curr_label == 1:
#             posTot += 1
#         ps.append(posTot)
#     return qvals, ps

#########################################################
#########################################################
################### I/O functions
#########################################################
#########################################################
def subsample_pin(filename, outputFile, outputFile2 = '', sampleRatio = 0.1):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number
        Normal tide features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	Peptide	Proteins
    """

    with open(filename, 'r') as f:
        r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
        headerInOrder = r.fieldnames
        l = headerInOrder
        if "ScanNr" not in l:
            raise ValueError("No ScanNr field, exitting")
        # if "Charge1" not in l:
        #     raise ValueError("No Charge1 field, exitting")

        totalPsms = 0
        totalTargets = 0
        targetInds = []
        decoyInds = []
        for i, row in enumerate(r):
            totalPsms += 1
            if row["Label"] == "1":
                totalTargets += 1
                targetInds.append(i+1)
            elif row["Label"] == "-1":
                decoyInds.append(i+1)

        subTotal = int(totalPsms * sampleRatio)
        print len(decoyInds), len(targetInds)
        print "%d total PSMs, %d targets, %d decoys" % (totalPsms, totalTargets, totalPsms - totalTargets)
        print "Subsampling %d PSMs" % (subTotal)
        numTargets = subTotal / 2
        numDecoys = subTotal - numTargets
        if numDecoys > len(decoyInds):
            m = numDecoys - len(decoyInds)
            numDecoys -= m
            numTargets += m
        else:
            if numTargets > len(targetInds):
                m = numTargets - len(targetInds)
                numTargets -= m
                numDecoys += m
        print len(decoyInds), len(targetInds), numTargets, numDecoys
        # Randomly sample target and decoy indices
        sampleInds = set(random.sample(targetInds, numTargets))
        # decoyInds = set(random.sample(decoyInds, numDecoys))
        sampleInds |= set(random.sample(decoyInds, numDecoys))
        sampleInds.add(0) # add header line

        f.seek(0)
        g = open(outputFile, 'w')
        g2 = []
        if outputFile2:
            g2 = open(outputFile2, 'w')
        for i,l in enumerate(f):
            if i in sampleInds:
                g.write(l)
            else:
                if outputFile2:
                    g2.write(l)
            if i==0:
                if outputFile2:
                    g2.write(l)
        g.close()
        if outputFile2:
            g2.close()

def load_pin_return_featureMatrix(filename):
    """ Load all PSMs and features from a percolator input (PIN) file
        
        For n input features and m total file fields, the file format is:
        header field 1: SpecId, or other PSM id
        header field 2: Label, denoting whether the PSM is a target or decoy
        header field 3: ScanNr, the scan number.  Note this string must be exactly stated
        header field 4 (optional): ExpMass, PSM experimental mass.  Not used as a feature
        header field 4 + 1 : Input feature 1
        header field 4 + 2 : Input feature 2
        ...
        header field 4 + n : Input feature n
        header field 4 + n + 1 : Peptide, the peptide string
        header field 4 + n + 2 : Protein id 1
        header field 4 + n + 3 : Protein id 2
        ...
        header field m : Protein id m - n - 4
    """

    f = open(filename, 'r')
    r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
    headerInOrder = r.fieldnames
    l = headerInOrder

    sids = [] # keep track of spectrum IDs
    # Check that header fields follow pin schema
    # spectrum identification key for PIN files
    # Note: this string must be stated exactly as the third header field
    sidKey = "ScanNr"
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")

    constKeys = [l[0]]
    # Check label
    m = l[1]
    if m.lower() == 'label':
        constKeys.append(l[1])
    # Exclude calcmass and expmass as features
    constKeys += [sidKey, "CalcMass", "ExpMass"]

    # Find peptide and protein ID fields
    psmStrings = [l[0]]
    isConstKey = False
    for key in headerInOrder:
        m = key.lower()
        if m=="peptide":
            isConstKey = True
        if isConstKey:
            constKeys.append(key)
            psmStrings.append(key)
            
    constKeys = set(constKeys) # exclude these when reserializing data

    keys = []
    for h in headerInOrder: # keep order of keys intact
        if h not in constKeys:
            keys.append(h)
    
    featureNames = []
    for k in keys:
        featureNames.append(k)
            
    targets = {}  # mapping between sids and indices in the feature matrix
    decoys = {}

    X = [] # Feature matrix
    Y = [] # labels
    pepstrings = []
    scoreIndex = _scoreInd # column index of the ranking score used by the search algorithm 
    numRows = 0
    
    # for i, l in enumerate(reader):
    for i, l in enumerate(r):
        try:
            sid = int(l[sidKey])
        except ValueError:
            print "Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1)

        try:
            y = int(l["Label"])
        except ValueError:
            print "Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1)
        if y != 1 and y != -1:
            print "Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1)
            exit(-1)

        el = []
        for k in keys:
            try:
                el.append(float(l[k]))
            except ValueError:
                print "Could not convert feature %s with value %s to float, exitting" % (k, l[k])

        # el_strings = (l["SpecId"], l["Peptide"], l["Proteins"])
        el_strings = [l[k] for k in psmStrings]
        if not _topPsm:
            X.append(el)
            Y.append(y)
            pepstrings.append(el_strings)
            sids.append(sid)
            numRows += 1
        else:
            if y == 1:
                if sid in targets:
                    featScore = X[targets[sid]][scoreIndex]
                    if el[scoreIndex] > featScore:
                        X[targets[sid]] = el
                        pepstrings[targets[sid]] = el_strings
                else:
                    targets[sid] = numRows
                    X.append(el)
                    Y.append(1)
                    pepstrings.append(el_strings)
                    sids.append(sid)
                    numRows += 1
            elif y == -1:
                if sid in decoys:
                    featScore = X[decoys[sid]][scoreIndex]
                    if el[scoreIndex] > featScore:
                        X[decoys[sid]] = el
                        pepstrings[decoys[sid]] = el_strings
                else:
                    decoys[sid] = numRows
                    X.append(el)
                    Y.append(-1)
                    pepstrings.append(el_strings)
                    sids.append(sid)
                    numRows += 1
    f.close()

    if _standardNorm:
        return pepstrings, preprocessing.scale(np.array(X)), np.array(Y), featureNames, sids
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        return pepstrings, min_max_scaler.fit_transform(np.array(X)), np.array(Y), featureNames, sids

def writeIdent(output, scores, Y, pepstrings,sids):
    """ Header is: (1) Kind (2) Sid (3) Peptide (4) Score
    """
    n = len(Y)
    fid = open(output, 'w')
    fid.write("Kind\tSid\tPeptide\tScore\n")
    counter = 0
    for i in range(n):
        sid = sids[i]
        p = pepstrings[i][1][2:-2]
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

def writeOutput(output, scores, Y, pepstrings,qvals):
    """ Header is: (1)PSMId (2)score (3)q-value (4)peptide (5)Label (6)proteinIds
    """
    n = len(Y)
    fid = open(output, 'w')
    fid.write("PSMId\tscore\tq-value\tpeptide\tLabel\tproteinIds\n")
    counter = 0
    for i in range(n): # preferable to using zip
        psm_id = pepstrings[i][0]
        p = pepstrings[i][1]
        prot_id = pepstrings[i][2]
        score = scores[i]
        q = qvals[i]
        l = Y[i]
        fid.write("%s\t%f\t%f\t%s\t%d\t%s\n"
                  % (psm_id, score, q, p, l, prot_id))
        counter += 1
    fid.close()
    print "Wrote %d PSMs" % counter

#########################################################
#########################################################
################### Initial score direction functions
#########################################################
#########################################################
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
                taq, _, _ = qvalues.calcQ(-1. * scores, Y, thresh, True, _verb)
            else:
                taq, _, _ = qvalues.calcQ(scores, Y, thresh, True, _verb)
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
                taq, _, _ = qvalues.calcQ(-1. * currScores, Y[trainSids], q, True, _verb)
            else:
                taq, _, _ = qvalues.calcQ(currScores, Y[trainSids], q, True, _verb)
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

#########################################################
#########################################################
################### Utility functions
#########################################################
#########################################################
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

def sortRowIndicesBySid(sids):
    """ Sort Scan Identification (SID) keys and retain original row indices of feature matrix X
    """
    keySids = sorted(zip(sids, range(len(sids))))
    xRowIndices = [j for (_,j) in keySids]
    sids = [i for (i,_) in  keySids]
    return sids, xRowIndices

def getDecoyIdx(labels, ids):
    return [i for i in ids if labels[i] != 1]

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

#########################################################
#########################################################
################### Learning algorithms
#########################################################
#########################################################
def doLdaSingleFold(thresh, kFold, features, labels, validateFeatures, validateLabels):
    """ Perform LDA on a CV bin
    """
    clf = lda()
    clf.fit(features, labels)
    validation_scores = clf.decision_function(validateFeatures)
    tp, _, _ = qvalues.calcQ(validation_scores, validateLabels, thresh, True, _verb)
    if _debug and _verb > 1:
        print "CV finished for fold %d: %d targets identified" % (kFold, len(tp))
    return validation_scores, len(tp), clf

def getPercWeights(currIter, kFold):
    """ Weights to debug overall pipeline
        Percolator optimal CV weights for worm01.pin
    """
    if currIter == 0:
        if kFold == 0:
            # cross-validation found 5048 training set PSMs with cpos = 0.1, cneg = 0.3
            clf = np.array([-0.0259891,0,0.122862,1.99297,-0.205812,0.131589,-0.240002,0.0422103,0.159012,-0.172794,0,0,0,-0.204852,0.00944616,-0.00376582,-3.02847])
        elif kFold == 1:
            # cross-validation found 5372 training set PSMs with cpos = 10, cneg = 100
            clf = np.array([0.0218895,0,0.161915,2.26042,-0.29484,0.21187,-0.0500683,0.0707332,0.258911,-0.282006,0,0,0,-0.0980667,0.0239779,-0.0278971,-3.75857])
        else:
            # cross-validation found 5228 training set PSMs with cpos = 0.1, cneg = 1
            clf = np.array([0.017683,0,0.152919,2.23658,-0.288384,0.248982,-0.350645,0.0369419,0.195835,-0.207899,0,0,0,-0.358095,-0.0027158,-0.0151766,-3.66699])
    elif currIter == 1:
        if kFold==0:
            # cross-validation found 5274 training set PSMs with cpos = 1, cneg = 10
            clf = np.array([-0.0159667,0,0.258948,2.35891,-0.427149,0.294673,-0.508974,0.110175,0.301756,-0.337725,0,0,0,-0.458718,0.019466,-0.0285029,-4.06692])
            # Note: tie with the weights commented out below
            # -0.00508149 0 0.24113 2.01383 -0.363366 0.256272 -0.407882 0.0928448 0.256659 -0.28697 0 0 0 -0.369307 0.0146373 -0.0180445 -3.50038 
            # - cross-validation found 5247 training set PSMs with cpos = 0.1, cneg = 1
        elif kFold == 1:
            # cross-validation found 5519 training set PSMs with cpos = 10, cneg = 30
            clf = np.array([0.0472272,0,0.284314,2.71909,-0.523136,0.397579,-0.120766,0.139299,0.425898,-0.471377,0,0,0,-0.211426,0.0274639,-0.0304848,-4.05817])
            # Note: tie with the weights commented out below
            # 0.0476343 0 0.278298 2.62417 -0.505862 0.389465 -0.116106 0.134484 0.408974 -0.45288 0 0 0 -0.205072 0.0261596 -0.0275075 -3.91696 
        else:
            # cross-validation found 5360 training set PSMs with cpos = 1, cneg = 3
            clf = np.array([0.0387359,0,0.314553,2.8278,-0.51302,0.434318,-0.65393,0.086333,0.365803,-0.393993,0,0,0,-0.644954,-0.00481508,-0.0170554,-4.6971])
            # Note: tie with the weights commented out below
            # 0.0418418 0 0.273516 2.18986 -0.382262 0.335036 -0.415145 0.0634514 0.268413 -0.289132 0 0 0 -0.389397 -0.00203847 0.000585443 -3.34466 
    elif currIter == 2:
        if kFold == 0:
            # - cross-validation found 5309 training set PSMs with cpos = 1, cneg = 1
            clf = np.array([-0.00843064,0,0.359603,2.68549,-0.539473,0.405523,-0.570098,0.155218,0.383265,-0.433937,0,0,0,-0.530636,0.0150884,-0.0371239,-4.02487])
        elif kFold == 1:
            # - cross-validation found 5541 training set PSMs with cpos = 10, cneg = 30
            clf = np.array([0.0556515,0,0.34885,2.64609,-0.60046,0.489418,-0.143493,0.198406,0.476925,-0.541695,0,0,0,-0.288446,0.0374692,-0.0266473,-4.09366])
            # Note: tie with the weights commented out below
            # 0.0556794 0 0.341036 2.55943 -0.580531 0.47699 -0.140124 0.191905 0.457938 -0.520586 0 0 0 -0.27987 0.0359987 -0.0240579 -3.95761 
        else:
            # - cross-validation found 5431 training set PSMs with cpos = 10, cneg = 100
            clf = np.array([0.0622362,0,0.402447,2.75538,-0.552603,0.447494,-0.73275,0.129017,0.430558,-0.472681,0,0,0,-0.695651,-0.0159191,-0.0226761,-4.75368])
            # Note: tie with the weights commented out below
            # 0.0828503 0 0.429292 3.01126 -0.607536 0.498555 -0.727165 0.118851 0.447518 -0.486324 0 0 0 -0.709421 -0.0121971 -0.00850306 -4.42439 
    elif currIter == 3:
        if kFold == 0:
            # - cross-validation found 5313 training set PSMs with cpos = 0.1, cneg = 1
            clf = np.array([0.020168,0,0.330994,1.89207,-0.404563,0.315711,-0.446399,0.143083,0.291405,-0.338112,0,0,0,-0.428735,0.00302618,-0.0356007,-3.46502])
        elif kFold == 1:
            # - cross-validation found 5538 training set PSMs with cpos = 0.1, cneg = 0.1
            clf = np.array([0.0660606,0,0.3263,2.00057,-0.477043,0.443968,-0.0596892,0.17502,0.335177,-0.39231,0,0,0,-0.207732,0.0317964,-0.00492672,-2.8646])
        else:
            # - cross-validation found 5450 training set PSMs with cpos = 0.1, cneg = 0.3
            clf = np.array([0.0701129,0,0.387639,2.10595,-0.460141,0.399405,-0.462105,0.100854,0.321063,-0.35399,0,0,0,-0.450039,-0.0191531,-0.00581594,-3.3918])
    else:
        raise ValueError("%d iteration weights called for, only four iterations of weights available, exitting." % (currIter))
    return clf

def getPercKimWeights(currIter, kFold):
    """ Weights to debug overall pipeline
        Percolator optimal CV weights for http://jthalloran.ucdavis.edu/kimDataset.pin.gz
    """
    if currIter == 0:
        if kFold == 0:
            # Found 933182 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.4355, 0.407106, -0.182639, -0.0029198, -0.025436, -0.0211001, 0.0393384, 0.0576993, 0.382542, 0.36422, -0.156119, 0.185237, 0.00459663, -0.15619, -2.55448])
        elif kFold == 1:
            # Found 930574 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001. 
            clf = np.array([0, 1.42124, 0.400207, -0.179377, -0.00695265, -0.0218471, -0.0212839, 0.0345309, 0.0594041, 0.380736, 0.363961, -0.153068, 0.185239, 0.0209587, -0.162228, -2.53709])
        else:
            # Found 931413 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.42639, 0.408076, -0.167477, -0.00511073, -0.0190763, -0.0208017, 0.0358508, 0.0467852, 0.381046, 0.371524, -0.154183, 0.181312, 0.0070681, -0.170513, -2.54931])
    elif currIter == 1:
        if kFold==0:
            # Found 976776 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.40404, 0.684771, -0.381992, -0.0030135, -0.0585104, -0.0255245, 0.0707433, 0.105767, 0.648636, 0.611938, -0.28366, 0.300422, 0.00270719, -0.346918, -2.63462])
        elif kFold == 1:
            # Found 974401 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.38993, 0.672239, -0.375274, -0.00544713, -0.0525887, -0.0257287, 0.0624373, 0.106192, 0.645127, 0.61033, -0.276387, 0.297176, 0.0226007, -0.351021, -2.61737])
        else:
            # Found 974512 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.40291, 0.68634, -0.360334, -0.00360264, -0.0516156, -0.0257881, 0.06897, 0.0906783, 0.648646, 0.619584, -0.282209, 0.292906, 0.0105015, -0.368127, -2.64271])
    elif currIter == 2:
        if kFold == 0:
            # Found 988020 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.23084, 0.799224, -0.452662, -0.000155801, -0.0696761, -0.0234961, 0.0774495, 0.117894, 0.715353, 0.670852, -0.323799, 0.299354, -0.00959294, -0.415832, -2.58927])
        elif kFold == 1:
            # Found 985496 training set PSMs with q<0.01 for hyperparameters Cpos=10, Cneg=100.001.
            clf = np.array([0, 1.21897, 0.786889, -0.445367, -0.000916953, -0.0640835, -0.0234464, 0.0697376, 0.116407, 0.711566, 0.66775, -0.3169, 0.293524, 0.00648674, -0.419575, -2.57393])
        else:
            # Found 985470 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=1.00001.
            clf = np.array([0, 1.22937, 0.801315, -0.431102, 0.000879794, -0.0641938, -0.0238616, 0.0775478, 0.1026, 0.7166, 0.675709, -0.323875, 0.290847, -0.00206429, -0.438003, -2.59866])
    elif currIter == 3:
        if kFold == 0:
            # Found 991720 training set PSMs with q<0.01 for hyperparameters Cpos=1, Cneg=3.00004.
            clf = np.array([0, 1.34536, 1.02, -0.551319, 0.00647906, -0.081274, -0.0303035, 0.0916025, 0.138379, 0.874836, 0.813276, -0.398534, 0.31878, -0.0242629, -0.535363, -2.66219])
        elif kFold == 1:
            # Found 989249 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=0.300004.
            clf = np.array([0, 1.33114, 1.00764, -0.54218, 0.00588772, -0.0750878, -0.0306167, 0.083969, 0.135917, 0.870543, 0.80922, -0.392101, 0.311066, -0.00908895, -0.538028, -2.64617])
        else:
            # Found 988958 training set PSMs with q<0.01 for hyperparameters Cpos=0.1, Cneg=0.300004.
            clf = np.array([0, 1.34116, 1.02427, -0.528916, 0.00756507, -0.076001, -0.0304979, 0.0929218, 0.121003, 0.87653, 0.818948, -0.39873, 0.308662, -0.0154511, -0.557602, -2.67202])
    else:
        raise ValueError("%d iteration weights called for, only four iterations of weights available, exitting." % (currIter))
    return clf

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
                # clf = getPercWeights(currIter, kFold)
                # clf = getPercKimWeights(currIter, kFold)
                clf = svmlin.ssl_train_with_data(features, labels, 0, Cn = alpha * cneg, Cp = alpha * cpos)
                validation_scores = np.dot(validateFeatures, clf[:-1]) + clf[-1]
            tp, _, _ = qvalues.calcQ(validation_scores, validateLabels, thresh, True, _verb)
            currentTaq = len(tp)
            if _debug and _verb > 2:
                print "CV fold %d: cpos = %f, cneg = %f separated %d validation targets" % (kFold, alpha * cpos, alpha * cneg, currentTaq)
            if currentTaq > bestTaq:
                topScores = validation_scores[:]
                bestTaq = currentTaq
                bestCp = cpos * alpha
                bestCn = cneg * alpha
                bestClf = deepcopy(clf)
    tp, _, _ = qvalues.calcQ(topScores, validateLabels, thresh, _verb)
    bestTaq = len(tp)
    if _debug and _verb > 1:
        print "CV finished for fold %d: best cpos = %f, best cneg = %f, %d targets identified" % (kFold, bestCp, bestCn, bestTaq)
    return topScores, bestTaq, bestClf

#########################################################
#########################################################
################### Calculate test scores
#########################################################
#########################################################
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
        tp, _, _ = qvalues.calcQ(testScores[testSids], Y[testSids], thresh, False, _verb)
        totalTaq += len(tp)
        kFold += 1
    return testScores, totalTaq

#########################################################
#########################################################
################### Main training functions
#########################################################
#########################################################
def doIter(thresh, keys, scores, X, Y,
           targetDecoyRatio, method = 0, currIter=1):
    """ Train a classifier on CV bins.
        Method 0: LDA
        Method 1: linear SVM, solver TRON
        Method 2: linear SVM, solver SVMLIN
    """
    totalTaq = 0 # total number of estimated true positives at q-value threshold
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
        taq, daq, _ = qvalues.calcQ(scores[kFold], Y[cvBinSids], thresh, True, _verb)
        gd = getDecoyIdx(Y, cvBinSids)
        # Debugging check
        if _debug and _verb >= 1:
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

def funcIter(options, output):
    q = options.q
    f = options.pin
    # target_rows: dictionary mapping target sids to rows in the feature matrix
    # decoy_rows: dictionary mapping decoy sids to rows in the feature matrix
    # X: standard-normalized feature matrix
    # Y: binary labels, true denoting a target PSM
    pepstrings, X, Y, featureNames, sids0 = load_pin_return_featureMatrix(f)
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

    # Iteratre
    fp = 0 # current number of identifications
    fpo = 0 # number of identifications from previous iteration
    fpoo = 0 # number of identifications from previous, previous iteration
    for i in range(options.maxIters):
        scores, fp, ws = doIter(q, trainKeys, scores, X, Y,
                                           targetDecoyRatio, options.method, i)
        print "Iter %d: estimated %d targets <= q = %f" % (i, fp, q)
        if _convergeCheck and fp > 0 and fpoo > 0 and (float(fp - fpoo) <= float(fpoo * _reqIncOver2Iters)):
            print "Algorithm seems to have converged over past two itertions, (%d vs %d)" % (fp, fpoo)
            break
        fpoo = fpo
        fpo = fp
    
    isSvmlin = (options.method==2)
    testScores, numIdentified = doTest(q, testKeys, X, Y, ws, isSvmlin)
    print "Identified %d targets <= %f pre-merge." % (numIdentified, q)
    if _mergescore:
        scores = doMergeScores(q, testKeys, testScores, Y)

    taq, _, qs = qvalues.calcQ(scores, Y, q, False, _verb)
    print "Could identify %d targets" % (len(taq))
    if not _identOutput:
        writeOutput(output, scores, Y, pepstrings, qs)
    else:
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
    parser.add_option('--outputfile', type = 'string', action= 'store')
    parser.add_option('--seed', type = 'int', action= 'store', default = 1)

    (options, args) = parser.parse_args()

    # Seed random number generator.  To make shuffling nondeterministic, input seed <= -1
    # if options.seed <= -1:
    #     random.seed()
    # else:
    #     random.seed(options.seed)
    _seed=options.seed
    _verb=options.verb
    funcIter(options, options.outputfile)
