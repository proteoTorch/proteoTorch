#!/usr/bin/env python
"""
Written by John Halloran <jthalloran@ucdavis.edu> (and Gregor Urban <gur9000@outlook.com>)

Copyright (C) 2020 John Halloran and Gregor Urban
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0
"""

from __future__ import with_statement

import csv
import optparse
import random

from sklearn.utils import check_random_state
from copy import deepcopy
from sklearn.svm import LinearSVC as svc
from sklearn import preprocessing
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import multiprocessing as mp

_mp_data = {}

try:
    from solvers import l2_svm_mfn
    svmlinReady = True
except:
    print("Loaded all solvers except L2-SVM-MFN")
    svmlinReady = False

try:
    from qvalues import calcQ, qMedianDecoyScore, calcQAndNumIdentified, numIdentifiedAtQ # load cython library
except:
    print("Cython q-value not found, loading strictly python q-value library")
    from pyfiles.qvalsBase import calcQ, getQValues, qMedianDecoyScore, calcQAndNumIdentified, numIdentifiedAtQ # import unoptimized q-value calculation

import dnn_code
import mini_utils


AUC_fn_001 = mini_utils.AUC_up_to_tol_singleQ(0.01)

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
#_indInd=2 # Used to keep track of feature matrix rows when sorting based on score


#########################################################
#########################################################
################### CV-bin score normalization
#########################################################
#########################################################
def doMergeScores(thresh, testSets, scores, Y, isSvm = False):
    # record new scores as we go
    newScores = np.zeros(scores.shape)
    if not isSvm:
        for testSids in testSets:
            for ts in testSids:
                newScores[ts] = scores[ts]
    else:
        for testSids in testSets:
            u, d = qMedianDecoyScore(scores[testSids], Y[testSids], thresh)
            diff = u - d
            if diff <= 0.:
                diff = 1.
            for ts in testSids:
                newScores[ts] = (scores[ts] - u) / (u-d)

    return newScores



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
        print(len(decoyInds), len(targetInds))
        print("%d total PSMs, %d targets, %d decoys" % (totalPsms, totalTargets, totalPsms - totalTargets))
        print("Subsampling %d PSMs" % (subTotal))
        numTargets = subTotal // 2
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
        print(len(decoyInds), len(targetInds), numTargets, numDecoys)
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

def givenPsmIds_writePin(filename, psmIdFile):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number

        Normal tide features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	Peptide	Proteins
    """
    with open(psmIdFile, 'r') as f0:
        psms = set([l["PSMId"] for l in csv.DictReader(f0)])
        print("Loaded %d PMS IDs" % len(psms))

    with open(filename, 'r') as f:
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
        for i, l in enumerate(r):
            if l["SpecId"] not in psms:
                continue
            try:
                y = int(l["Label"])
            except ValueError:
                print("Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1))
            if y != 1 and y != -1:
                print("Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1))
                exit(-1)
            el = []
            for k in keys:
                try:
                    el.append(float(l[k]))
                except ValueError:
                    print("Could not convert feature %s with value %s to float, exitting" % (k, l[k]))
            el_strings = [l[k] for k in psmStrings]
            X.append(el)
            Y.append(y)

    return np.array(X), Y, featureNames


def load_pin_return_featureMatrix(filename, normalize = True):
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
    sids = [] # keep track of spectrum IDs and exp masses for FDR post-processing
    expMasses = [] 
    # Check that header fields follow pin schema
    # spectrum identification key for PIN files
    # Note: this string must be stated exactly as the third header field
    sidKey = "ScanNr"
    if sidKey not in l:
        raise ValueError("No %s field, exitting" % (sidKey))
    expMassKey = "ExpMass"
    if expMassKey not in l:
        raise ValueError("No %s field, exitting" % (expMassKey))
    constKeys = [l[0]]

    # denote charge keys
    maxCharge = 1
    chargeKeys = set([])
    for i in l:
        m = i.lower()
        if m[:-1]=='charge':
            try:
                c = int(m[-1])
            except ValueError:
                print("Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1))

            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

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
            print("Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1))
        # try:
        #     expMass = float(l[expMassKey])
        # except ValueError:
        #     print("Could not convert exp mass %s on line %d to float, exitting" % (l[expMassKey], i+1))
        expMass = l[expMassKey]
        try:
            y = int(l["Label"])
        except ValueError:
            print("Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1))
        if y != 1 and y != -1:
            print("Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1))
            exit(-1)
        el = []
        for k in keys:
            try:
                el.append(float(l[k]))
            except ValueError:
                print("Could not convert feature %s with value %s to float, exitting" % (k, l[k]))
        el_strings = [l[k] for k in psmStrings]
        if not _topPsm:
            X.append(el)
            Y.append(y)
            pepstrings.append(el_strings)
            sids.append(sid)
            expMasses.append(expMass)
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
                    expMasses.append(expMass)
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
                    expMasses.append(expMass)
                    numRows += 1
    f.close()
    if not normalize:
        return pepstrings, np.array(X), np.array(Y), featureNames, sids, expMasses

    if _standardNorm:
        return pepstrings, preprocessing.scale(np.array(X)), np.array(Y), featureNames, sids, expMasses
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        return pepstrings, min_max_scaler.fit_transform(np.array(X)), np.array(Y), featureNames, sids, expMasses

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
            fid.write("t\t%d\t%s\t%f\n" % (sid,p,score))
            counter += 1
        else:
            fid.write("d\t%d\t%s\t%f\n" % (sid,p,score))
            counter += 1
    fid.close()
    print("Wrote %d PSMs" % counter)


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
    print("Wrote %d PSMs" % counter)


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
                taq, _, _ = calcQ(-1. * scores, Y, thresh, True)
            else:
                taq, _, _ = calcQ(scores, Y, thresh, True)
            if len(taq) > numIdentified:
                initDirection = i
                numIdentified = len(taq)
                negBest = checkNegBest==1
            if _debug and _verb >= 2:
                if checkNegBest==1:
                    print("Direction -%d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq)))
                else:
                    print("Direction %d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq)))
    return initDirection, numIdentified, negBest

def evalDirectionInThread(i, scores, Y, thresh, featureNames):
    negBest = False
    numIdentified = -1
    # Check scores multiplied by both 1 and positive -1
    for checkNegBest in range(2):
        if checkNegBest==1:
            taq, _, _ = calcQ(-1. * scores, Y, thresh, True)
        else:
            taq, _, _ = calcQ(scores, Y, thresh, True)
        if len(taq) > numIdentified:
            numIdentified = len(taq)
            negBest = checkNegBest==1
        if _debug and _verb >= 2:
            if checkNegBest==1:
                print("Direction -%d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq)))
            else:
                print("Direction %d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq)))
    return (i,numIdentified, negBest)

    
def findInitDirection_threaded(X, Y, thresh, featureNames, 
                               numThreads = 1):
    l = X.shape
    m = l[1] # number of columns/features
    initDirection = -1
    numIdentified = -1
    # TODO: add check verifying best direction idetnfies more than -1 spectra, otherwise something
    # went wrong
    negBest = False

    # create threadpool
    numThreads = min([mp.cpu_count() - 1, numThreads, m])
    pool = mp.Pool(processes = numThreads)

    # distribute jobs
    results = [pool.apply_async(evalDirectionInThread, 
                                args=(i, X[:,i], Y, thresh, featureNames)) 
               for i in range(m)]
    pool.close()
    pool.join() # wait for jobs to finish before continuing

    for d in results:
        direction = d.get()
        if(direction[1] > numIdentified):
            initDirection = direction[0]
            numIdentified = direction[1]
            negBest = direction[2]
    return initDirection, numIdentified, negBest

def givenInitialDirection_split(keys, X, Y, q, featureNames, initDir):
    """ Given initial search directions, returns the scores for the disjoint bins
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
            print("CV fold %d: could separate %d PSMs in supplied initial direction -%d, %s" % (kFold, numIdentified, initDir, featureNames[initDir]))
            scores.append(-1. * currScores)
        else:
            print("CV fold %d: could separate %d PSMs in supplied initial direction %d, %s" % (kFold, numIdentified, initDir, featureNames[initDir]))
            scores.append(currScores)
    return scores, initTaq

def load_and_score_dnns(thresh, keys, X, Y, hparams = {}, input_dir = None):
    """ Load dnns and generate test scores
    """
    newScores = []
    totalTaq = 0
    num_features = X.shape[1]
    for kFold, sids in enumerate(keys):
        w = dnn_code.loadDNNSingleFold(num_features, kFold, hparams, input_dir)
        scores = w.decision_function(X[sids])
        # Calculate true positives
        tp, _, _ = calcQ(scores, Y[sids], thresh, True)
        totalTaq += len(tp)

        newScores.append(scores)
    return newScores, totalTaq

def searchForInitialDirection_split(keys, X, Y, q, featureNames, numThreads = 1):
    """ Iterate through cross validation training sets and find initial search directions
        Returns the scores for the disjoint bins
    """
    initTaq = 0.
    scores = []
    kFold = 0
    for trainSids in keys:
        # Find initial direction
        if numThreads <=1:
            initDir, numIdentified, negBest = findInitDirection(X[trainSids], Y[trainSids], q, featureNames)
        else:
            initDir, numIdentified, negBest = findInitDirection_threaded(X[trainSids], Y[trainSids], q, featureNames, numThreads)

        initTaq += numIdentified
        if negBest:
            print("CV fold %d: could separate %d PSMs in initial direction -%d, %s" % (kFold, numIdentified, initDir, featureNames[initDir]))
            scores.append(-1. * X[trainSids,initDir])
        else:
            print("CV fold %d: could separate %d PSMs in initial direction %d, %s" % (kFold, numIdentified, initDir, featureNames[initDir]))
            scores.append(X[trainSids,initDir])
        kFold += 1
    return scores, initTaq

def deepDirectionSearch(keys, scores, X, Y,
                        dnn_hyperparams={}, ensemble = 50):
    """ Given initial set of scores, perform a deep direction search with a large number of ensembles
    """
    estTaq = 0
    newScores = []
    dds_params = dnn_hyperparams.copy()
    dds_params['snapshot_ensemble_count'] = ensemble
    thresh = dnn_hyperparams['deepq']
    q = dnn_hyperparams['q']
        
    for kFold, cvBinSids in enumerate(keys):
        # Find training set using q-value analysis
        taq, daq, _ = calcQ(scores[kFold], Y[cvBinSids], thresh, True)
        td = [cvBinSids[i] for i in taq]
        gd = getDecoyIdx(Y, cvBinSids)
        # Debugging check
        if _debug and _verb >= 1:
            print("CV fold %d: |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d" % (kFold, len(cvBinSids) - len(gd), len(gd), len(taq), len(daq)))
        trainSids = gd + td
        validation_Sids = cvBinSids
        features = X[trainSids]
        labels = Y[trainSids]
        validation_Features = X[validation_Sids]
        validation_Labels = Y[validation_Sids]
        
        topScores, bestTaq, bestClf = dnn_code.DNNSingleFold(thresh, kFold, features, labels, validation_Features, 
                                                             validation_Labels, hparams=dds_params)
        newScores.append(topScores)
        ps = numIdentifiedAtQ(topScores, validation_Labels, q)
        estTaq += len(ps)
    return newScores, estTaq


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
    keySids = sorted(list(zip(sids, range(len(sids)))))
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
    for k,sid in list(zip(featureMatRowIndices, sids)):
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
def doLdaSingleFold(thresh, kFold, features, labels, validation_Features, validation_Labels):
    """ Perform LDA on a CV bin
    """
    clf = lda()
    clf.fit(features, labels)
    validation_scores = clf.decision_function(validation_Features)
    tp, _, _ = calcQ(validation_scores, validation_Labels, thresh, True)
    if _debug and _verb > 1:
        print("CV finished for fold %d: %d targets identified" % (kFold, len(tp)))
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


def doSvmGridSearch(thresh, kFold, features, labels, validation_Features, validation_Labels, 
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
                validation_scores = clf.decision_function(validation_Features)
            else:
                clf = l2_svm_mfn.solver(features, labels, 0, Cn = alpha * cneg, Cp = alpha * cpos)
                validation_scores = np.dot(validation_Features, clf[:-1]) + clf[-1]
            tp, _, _ = calcQ(validation_scores, validation_Labels, thresh, True)
            currentTaq = len(tp)
            if _debug and _verb > 2:
                print("CV fold %d: cpos = %f, cneg = %f separated %d validation targets" % (kFold, alpha * cpos, alpha * cneg, currentTaq))
            if currentTaq > bestTaq:
                topScores = np.array(validation_scores[:])
                bestTaq = currentTaq
                bestCp = cpos * alpha
                bestCn = cneg * alpha
                bestClf = deepcopy(clf)
    tp, _, _ = calcQ(topScores, validation_Labels, thresh)
    bestTaq = len(tp)
    if _debug and _verb > 1:
        print("CV finished for fold %d: best cpos = %f, best cneg = %f, %d targets identified" % (kFold, bestCp, bestCn, bestTaq))
    return topScores, bestTaq, bestClf

def evalSvmCposCnegPair(features, labels, 
                        validation_Features, validation_Labels,
                        tron, cpos, cfrac, alpha, thresh, kFold):
    cneg = cfrac*cpos
    if tron:
        classWeight = {1: alpha * cpos, -1: alpha * cneg}
        clf = svc(dual = False, fit_intercept = True, class_weight = classWeight, tol = 1e-7)
        clf.fit(features, labels)
        validation_scores = clf.decision_function(validation_Features)
    else:
        clf = l2_svm_mfn.solver(features, labels, 0, Cn = alpha * cneg, Cp = alpha * cpos)
        validation_scores = np.dot(validation_Features, clf[:-1]) + clf[-1]
    tp, _, _ = calcQ(validation_scores, validation_Labels, thresh, True)
    currentTaq = len(tp)
    if _debug and _verb > 2:
        print("CV fold %d: cpos = %f, cneg = %f separated %d validation targets" % (kFold, alpha * cpos, alpha * cneg, currentTaq))
    return (cpos * alpha, cneg * alpha, currentTaq, np.array(validation_scores), clf)

def evalSvmCposCnegPair_globalDataMatrix(tron, cpos, cfrac, alpha, thresh, kFold):
    features = _mp_data[(kFold, 'X')]
    labels = _mp_data[(kFold, 'Y')]
    validation_Features = _mp_data[(kFold, 'validation_X')]
    validation_Labels = _mp_data[(kFold, 'validation_Y')]
    cneg = cfrac*cpos
    if tron:
        classWeight = {1: alpha * cpos, -1: alpha * cneg}
        clf = svc(dual = False, fit_intercept = True, class_weight = classWeight, tol = 1e-7)
        clf.fit(features, labels)
        validation_scores = clf.decision_function(validation_Features)
    else:
        clf = l2_svm_mfn.solver(features, labels, 0, Cn = alpha * cneg, Cp = alpha * cpos)
        validation_scores = np.dot(validation_Features, clf[:-1]) + clf[-1]
    tp, _, _ = calcQ(validation_scores, validation_Labels, thresh, True)
    currentTaq = len(tp)
    if _debug and _verb > 2:
        print("CV fold %d: cpos = %f, cneg = %f separated %d validation targets" % (kFold, alpha * cpos, alpha * cneg, currentTaq))
    return (kFold, cpos * alpha, cneg * alpha, currentTaq, np.array(validation_scores), clf)

def doSvmGridSearch_threaded(thresh, kFold, features, labels, validation_Features, validation_Labels, 
                             cposes, cfracs, alpha, tron = True, currIter=1, numThreads = 1):
    bestTaq = -1.
    bestCp = 1.
    bestCn = 1.
    bestClf = []
    cposCfracPairs = [(cpos, cfrac) for cpos in cposes for cfrac in cfracs]

    numThreads = min([mp.cpu_count() - 1, numThreads, len(cposCfracPairs)])
    pool = mp.Pool(processes = numThreads)
    results = [pool.apply_async(evalSvmCposCnegPair,
                                args=(features, labels, validation_Features, validation_Labels,
                                      tron, cpos, cfrac, alpha, thresh, kFold)) 
               for (cpos,cfrac) in cposCfracPairs]

    pool.close()
    pool.join() # wait for jobs to finish before continuing

    for p in results:
        cposCnegCandidate = p.get()
        currentTaq = cposCnegCandidate[2]
        if currentTaq > bestTaq:
            bestTaq = currentTaq
            bestCp = cposCnegCandidate[0]
            bestCn = cposCnegCandidate[1]
            topScores = np.array(cposCnegCandidate[3])
            bestClf = deepcopy(cposCnegCandidate[4])
    tp, _, _ = calcQ(topScores, validation_Labels, thresh)
    bestTaq = len(tp)
    if _debug and _verb > 1:
        print("CV finished for fold %d: best cpos = %f, best cneg = %f, %d targets identified" % (kFold, bestCp, bestCn, bestTaq))
    return topScores, bestTaq, bestClf


#########################################################
#########################################################
################### Calculate test scores
#########################################################
#########################################################
def doTest(thresh, keys, X, Y, trained_models, svmlin = False):
#    m = len(keys)/3
    testScores = np.zeros(Y.shape)
    totalTaq = 0
    for kFold, testSids in enumerate(keys):
        w = trained_models[kFold]
        if svmlin:
            testScores[testSids] = np.dot(X[testSids], w[:-1]) + w[-1]
        else:
            testScores[testSids] = w.decision_function(X[testSids])
        # Calculate true positives
        tp, _, _ = calcQ(testScores[testSids], Y[testSids], thresh, False)
        totalTaq += len(tp)
    return testScores, totalTaq





#########################################################
#########################################################
################### Main training functions
#########################################################
#########################################################
def doIter(thresh, keys, scores, X, Y, targetDecoyRatio, method = 0, currIter=1, 
           dnn_hyperparams={}, prev_iter_models=[], numThreads = 1):
    """ Train a classifier on CV bins.
        Method 0: LDA
        Method 1: linear SVM, solver TRON
        Method 2: linear SVM, solver SVMLIN
        Method 3: DNN (MLP)
        
        DNN will warm-start training the models the different folds between iterations.
    """
    # record new scores as we go
    # newScores = np.zeros(scores.shape)
    newScores = []
    clfs = [] # classifiers
    all_AUCs = []
    # C for SVM positive and negative classes
    cposes = [10., 1., 0.1]
    cfracs = [targetDecoyRatio, 3. * targetDecoyRatio, 10. * targetDecoyRatio]
    estTaq = 0
    tron = False
    alpha = 1. # Scale factor for classes, as L2-SVM-MFN and TRON assume different 
               # class weight scales
    # if prev_iter_models is None or len(prev_iter_models) < len(keys):
    #     prev_iter_models = [None] * len(keys)

    isSvm = False
    if method==1:
        tron = True
        alpha = 0.5
        isSvm = True
    elif method==2:
        isSvm = True

    if numThreads==1 or not isSvm: # check whether we need to parallelize the SVM grid search
        for kFold, cvBinSids in enumerate(keys):
            # Find training set using q-value analysis
            taq, daq, _ = calcQ(scores[kFold], Y[cvBinSids], thresh, True)
            td = [cvBinSids[i] for i in taq]
            gd = getDecoyIdx(Y, cvBinSids)
            # Debugging check
            if _debug and _verb >= 1:
                print("CV fold %d: |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d" % (kFold, len(cvBinSids) - len(gd), len(gd), len(taq), len(daq)))
            trainSids = gd + td
            validation_Sids = cvBinSids
            features = X[trainSids]
            labels = Y[trainSids]
            validation_Features = X[validation_Sids]
            validation_Labels = Y[validation_Sids]
        
            if method == 0:
                topScores, bestTaq, bestClf = doLdaSingleFold(thresh, kFold, features, labels, validation_Features, validation_Labels)
            elif method in [1, 2]: # helpful to keep this single-threaded SVM implementation in for profiling
                topScores, bestTaq, bestClf = doSvmGridSearch(thresh, kFold, features, labels,validation_Features, validation_Labels,
                                                              cposes, cfracs, alpha, tron, currIter)
            else:
                topScores, bestTaq, bestClf = dnn_code.DNNSingleFold(thresh, kFold, features, labels, validation_Features, 
                                                                     validation_Labels, hparams=dnn_hyperparams)
            all_AUCs.append( AUC_fn_001(topScores, validation_Labels) )
            newScores.append(topScores)
            clfs.append(bestClf)
            estTaq += bestTaq
    else: # parallelize over CV bins and SVM grid search with minimal data copying overhead
        newScores = [None, None, None]
        clfs = [None, None, None]
        all_AUCs = [None, None, None]
        bestTaqs = [-1, -1, -1]
        bestCps = [-1, -1, -1]
        bestCns = [-1, -1, -1]
        cposCfracPairs = [(cpos, cfrac, kFold) for cpos in cposes for cfrac in cfracs for kFold in range(len(keys))]
        for kFold, cvBinSids in enumerate(keys): # first make global training and validation sets
            # Find training set using q-value analysis
            taq, daq, _ = calcQ(scores[kFold], Y[cvBinSids], thresh, True)
            td = [cvBinSids[i] for i in taq]
            gd = getDecoyIdx(Y, cvBinSids)
            # Debugging check
            if _debug and _verb >= 1:
                print("CV fold %d: |targets| = %d, |decoys| = %d, |taq|=%d, |daq|=%d" % (kFold, len(cvBinSids) - len(gd), len(gd), len(taq), len(daq)))
            trainSids = gd + td
            validation_Sids = cvBinSids
            global _mp_data
            _mp_data[(kFold, 'X')] = X[trainSids]
            _mp_data[(kFold, 'Y')] = Y[trainSids]
            _mp_data[(kFold, 'validation_X')] = X[validation_Sids]
            _mp_data[(kFold, 'validation_Y')] = Y[validation_Sids]
        numThreads = min([mp.cpu_count() - 1, numThreads, len(cposCfracPairs)])
        pool = mp.Pool(processes = numThreads)
        results = [pool.apply_async(evalSvmCposCnegPair_globalDataMatrix,
                                    args=(tron, cpos, cfrac, alpha, thresh, kFold)) 
                   for (cpos,cfrac, kFold) in cposCfracPairs]
        
        pool.close()
        pool.join() # wait for jobs to finish before continuing
        for p in results:
            cposCnegCandidate = p.get()
            kFold = cposCnegCandidate[0]
            currentTaq = cposCnegCandidate[3]
            if currentTaq > bestTaqs[kFold]:
                bestTaqs[kFold] = currentTaq
                bestCps[kFold] = cposCnegCandidate[1]
                bestCns[kFold] = cposCnegCandidate[2]
                newScores[kFold] = np.array(cposCnegCandidate[4])
                clfs[kFold] = deepcopy(cposCnegCandidate[5])
        for kFold in range(len(keys)):
            tp, _, _ = calcQ(newScores[kFold], _mp_data[kFold, 'validation_Y'], thresh)
            bestTaqs[kFold] = len(tp)
            all_AUCs[kFold] = AUC_fn_001(newScores[kFold], _mp_data[kFold, 'validation_Y'] )
        estTaq = np.sum(bestTaqs)
    estTaq /= 2
    return newScores, estTaq, clfs, np.mean(all_AUCs)


def mainIter(hyperparams):
    """
    
    """
    global _seed, _verb
    _seed=hyperparams['seed']
    _verb=hyperparams['verbose']

    if hyperparams['method']==2 and not svmlinReady:
        print("Selected method 2, SVM learning with L2-SVM-MFN,")
        print("but this solver could be found.  Please build this solver")
        print("in the solvers directory or select a different method.")
        exit(-1)

    output_dir = hyperparams['output_dir']
    if output_dir is None:
        output_dir = 'model_output/{}/{}/'.format(hyperparams['pin'].split('/')[-1], mini_utils.TimeStamp())
    else:
        if not output_dir[-1]=='/':
            output_dir = output_dir + '/'
    mini_utils.mkdir(output_dir)
    q = hyperparams['q']
    
    isSvm = False
    if hyperparams['method'] in [1,2]:
        isSvm = True

    # target_rows: dictionary mapping target sids to rows in the feature matrix
    # decoy_rows: dictionary mapping decoy sids to rows in the feature matrix
    # X: standard-normalized feature matrix
    # Y: binary labels, true denoting a target PSM
    pepstrings, X, Y, featureNames, sids0, _ = load_pin_return_featureMatrix(hyperparams['pin'])
    sids, sidSortedRowIndices = sortRowIndicesBySid(sids0)
    l = X.shape
    m = l[1] # number of features
    targetDecoyRatio, numT, numD = calculateTargetDecoyRatio(Y)
    print("Loaded %d target and %d decoy PSMS with %d features, ratio = %f" % (numT, numD, l[1], targetDecoyRatio))
    if _debug and _verb >= 3:
        print(featureNames)
    trainKeys, testKeys = partitionCvBins(sidSortedRowIndices, sids)

    initDirectionFound = False
    if hyperparams['load_previous_dnn']:
        input_dir = hyperparams['previous_dnn_dir']
        if input_dir is not None:
            if not input_dir[-1]=='/':
                input_dir = input_dir + '/'

            print("Loading previously trained models")
            scores, initTaq = load_and_score_dnns(q, trainKeys, X, Y, hyperparams, input_dir)
            print("Could separate %d identifications" % ( initTaq / 2 ))
            initDirectionFound = True

    if not initDirectionFound:
        initTaq = 0.
        initDir = hyperparams['initDirection']
        if initDir > -1 and initDir < m:
            print("Using specified initial direction %d" % (initDir))
            scores, initTaq = givenInitialDirection_split(trainKeys, X, Y, q, featureNames, initDir)
        else:
            scores, initTaq = searchForInitialDirection_split(trainKeys, X, Y, q, featureNames, hyperparams['numThreads'])
        print("Could initially separate %d identifications" % ( initTaq / 2 ))
        if hyperparams['deepInitDirection']:
            print("Performing deep initial direction search")
            scores, initTaq = deepDirectionSearch(trainKeys, scores, X, Y,
                                                  dnn_hyperparams=hyperparams, ensemble = hyperparams['deep_direction_ensemble'])

    # initTaq = 0.
    # initDir = hyperparams['initDirection']
    # if initDir > -1 and initDir < m:
    #     print("Using specified initial direction %d" % (initDir))
    #     scores, initTaq = givenInitialDirection_split(trainKeys, X, Y, q, featureNames, initDir)
    # else:
    #     scores, initTaq = searchForInitialDirection_split(trainKeys, X, Y, q, featureNames, hyperparams['numThreads'])
    # print("Could initially separate %d identifications" % ( initTaq / 2 ))
    # if hyperparams['deepInitDirection']:
    #     print("Performing deep initial direction search")
    #     scores, initTaq = deepDirectionSearch(trainKeys, scores, X, Y,
    #                                           dnn_hyperparams=hyperparams, ensemble = hyperparams['deep_direction_ensemble'])

    # Iterate
    fp = 0 # current number of identifications
    fpo = 0 # number of identifications from previous iteration
    fpoo = 0 # number of identifications from previous, previous iteration
    trained_models = []
    drop_out_rate = hyperparams['dnn_dropout_rate']
    hyperparams['dnn_dropout_rate'] = hyperparams['starting_dropout_rate']
    if hyperparams['method'] == 3:
        q = hyperparams['deepq']
    else:
        q = hyperparams['q']

    for i in range(hyperparams['maxIters']):
        if i>0:
            hyperparams['dnn_dropout_rate'] = drop_out_rate
        scores, fp, trained_models, validation_AUC = doIter(
            q, trainKeys, scores, X, Y, targetDecoyRatio, hyperparams['method'], i, 
            dnn_hyperparams=hyperparams, prev_iter_models = trained_models, numThreads = hyperparams['numThreads'])
        print("Iter %d: estimated %d targets <= q = %f" % (i, fp, q))
        if _convergeCheck and fp > 0 and fpoo > 0 and (float(fp - fpoo) <= float(fpoo * _reqIncOver2Iters)):
            print("Algorithm seems to have converged over past two itertions, (%d vs %d)" % (fp, fpoo))
            break
        fpoo = fpo
        fpo = fp

        # write output for iteration
        isSvmlin = (hyperparams['method']==2)
        testScores, numIdentified = doTest(q, testKeys, X, Y, trained_models, isSvmlin)
        testScores = doMergeScores(q, testKeys, testScores, Y, isSvm)
        taq, _, qs = calcQ(testScores, Y, q, False)
        if not _identOutput:
            writeOutput(output_dir+'output_iter' + str(i) + '.txt', testScores, Y, pepstrings, qs)
        else:
            writeOutput(output_dir+'output_iter' + str(i) + '.txt', testScores, Y, pepstrings, sids0)
        # save current models
        if hyperparams['method']==3:
            for fold, clf in enumerate(trained_models):
                    dnn_code.saveDNNSingleFold(clf.get_single_model(), fold, output_dir)

    isSvmlin = (hyperparams['method']==2)
    testScores, numIdentified = doTest(q, testKeys, X, Y, trained_models, isSvmlin)
    print("Identified %d targets <= %f pre-merge." % (numIdentified, q))
    if _mergescore:
        scores = doMergeScores(q, testKeys, testScores, Y, isSvm)
    taq, _, qs = calcQ(scores, Y, q, False)
    print("Could identify %d targets" % (len(taq)))
    mini_utils.save_text(output_dir+'hparams.txt', str(hyperparams))
    if not _identOutput:
        writeOutput(output_dir+'output.txt', scores, Y, pepstrings, qs)
    else:
        writeOutput(output_dir+'output.txt', scores, Y, pepstrings, sids0)
    return None, validation_AUC, AUC_fn_001(testScores, Y), output_dir




if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('--q', type = 'float', action= 'store', default = 0.01)
    parser.add_option('--deepq', type = 'float', action= 'store', default = 0.07)
    parser.add_option('--tol', type = 'float', action= 'store', default = 0.01)
    parser.add_option('--load_previous_dnn', action= 'store_true', help = 'Start iterations from previously trained model saved in output_dir')
    parser.add_option('--previous_dnn_dir', type = 'string', action= 'store', default=None, help='Previous output directory containing trained dnn weights.')
    parser.add_option('--deepInitDirection', action= 'store_true', help = 'Perform initial direction search using deep models.')
    parser.add_option('--initDirection', type = 'int', action= 'store', default=-1)
    parser.add_option('--numThreads', type = 'int', action= 'store', default=1)
    parser.add_option('--verbose', type = 'int', action= 'store', default = 3)
    parser.add_option('--method', type = 'int', action= 'store', default = 3, 
                      help = 'Method 0: LDA; Method 1: linear SVM, solver TRON; Method 2: linear SVM, solver SVMLIN; Method 3: DNN (MLP)')
    parser.add_option('--methods', type = 'string', action= 'store', default = '3', 
                      help = 'String binding which method to run at which iteration.  See method input for more info about available methods.')
    parser.add_option('--maxIters', type = 'int', action= 'store', default = 10, help='number of iterations; runs on multiple splits per iterations.') #4
    parser.add_option('--pin', type = 'string', action= 'store', help='input file in PIN format')
    parser.add_option('--output_dir', type = 'string', action= 'store', default=None, help='Defaults to model_output/<data_file_name>/<time_stamp>/')
    parser.add_option('--seed', type = 'int', action= 'store', default = 1)
    parser.add_option('--dnn_num_epochs', type = 'int', action= 'store', default = 60, help='number of epochs for training the DNN model.')
    parser.add_option('--dnn_lr', type = 'float', action= 'store', default = 0.001, help='learning rate for training the DNN model.')
    parser.add_option('--dnn_lr_decay', type = 'float', action= 'store', default = 0.02, 
                      help='learning rate reduced by this factor during training overall (a fraction of this is applied after each epoch).')
    parser.add_option('--dnn_num_layers', type = 'int', action= 'store', default = 3)
    parser.add_option('--dnn_layer_size', type = 'int', action= 'store', default = 200, help='number of neurons per hidden layerin the DNN model.')
    parser.add_option('--dnn_dropout_rate', type = 'float', action= 'store', default = 0.0, help='dropout rate; must be 0 <= rate < 1.')
    parser.add_option('--starting_dropout_rate', type = 'float', action= 'store', default = 0.0, help='dropout rate for first iteration, must be 0 <= rate < 1.  Values > 0 promote initial exploration of the parameter space')
    parser.add_option('--dnn_gpu_id', type = 'int', action= 'store', default = 0, 
                      help='GPU ID to use for the DNN model (starts at 0; will default to CPU mode if no GPU is found or CUDA is not installed)')
    parser.add_option('--dnn_label_smoothing_0', type = 'float', action= 'store', default = 0.99, help='Label smoothing class 0 (negatives)')
    parser.add_option('--dnn_label_smoothing_1', type = 'float', action= 'store', default = 0.99, help='Label smoothing class 1 (positives)')
    parser.add_option('--dnn_train_qtol', type = 'float', action= 'store', default = 0.1, help='AUC q-value tolerance for validation set.')
    parser.add_option('--snapshot_ensemble_count', type = 'int', action= 'store', default = 10, help='Number of ensembles to train.')
    parser.add_option('--deep_direction_ensemble', type = 'int', action= 'store', default = 20, help='Number of ensembles to train.')
    parser.add_option('--false_positive_loss_factor', type = 'float', action= 'store', default = 4.0, help='Multiplicative factor to weight false positives')
    parser.add_option('--dnn_optimizer', type = 'string', action= 'store', default= 'adam', help='DNN solver to use.')
    (_options, _args) = parser.parse_args()
    
    mainIter(_options.__dict__)
