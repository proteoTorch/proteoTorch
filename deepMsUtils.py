#!/usr/bin/env python
"""
Written by John Halloran <jthalloran@ucdavis.edu>

Copyright (C) 2020 John Halloran
Licensed under the Open Software License version 3.0
See COPYING or http://opensource.org/licenses/OSL-3.0
"""
from __future__ import print_function

import os
import numpy as np
import optparse
import sys
import csv

def err_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

try:
    import matplotlib
    matplotlib.use('Agg')
    import pylab
except ImportError:
    err_print('Module "matplotlib" not available.')
    exit(-1)

import itertools
import numpy

from deepMs import (calcQAndNumIdentified, givenPsmIds_writePin, 
                    load_pin_return_featureMatrix, load_pin_return_scanExpmassPairs,
                    calculateTargetDecoyRatio, searchForInitialDirection_split,
                    calcQ, getDecoyIdx, sortRowIndicesBySid)
from scipy.spatial import distance

def calcDistanceMat(testMat,trainMat, metric = 'euclidean'):
    """ Distance/similarity metrics: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    """
    z = distance.cdist(testMat, trainMat, metric)
    return z

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

def disagreedPsms_computeSimilarity(pin, disagreedPsmsFile, seed = 1):
    global _seed 
    _seed = seed
    q=0.01
    thresh=q
    pepstrings, X, Y, featureNames, sids0 = load_pin_return_featureMatrix(pin)
    # PSMId is first entry of each tuple in pepstrings
    sids, sidSortedRowIndices = sortRowIndicesBySid(sids0)
    l = X.shape
    m = l[1] # number of features
    targetDecoyRatio, numT, numD = calculateTargetDecoyRatio(Y)
    print("Loaded %d target and %d decoy PSMS with %d features, ratio = %f" % (numT, numD, l[1], targetDecoyRatio))
    trainKeys, testKeys = partitionCvBins(sidSortedRowIndices, sids)
    initTaq = 0.
    scores, initTaq = searchForInitialDirection_split(trainKeys, X, Y, q, featureNames)

    # load disagreed PSMs
    psmIds = []
    with open(disagreedPsmsFile, 'r') as f:
        try:
            psmIds = set([l["PSMId"] for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)])
        except ValueError:
            print("Error: no PSMId field in disagreed PSM file %s, exitting" % (disagreedPsmsFile))
            exit(-1)
    print("Loaded %d PSMIds" % (len(psmIds)))
    # sets of test set indices
    testIndSets = [set(tk) for tk in testKeys]
    
    submatrixTestIndices = []
    for i in range(len(testKeys)):
        submatrixTestIndices.append([])
    # find testing bins for each disagreed PSM
    psmIdDict = {}
    for i, p in enumerate(pepstrings):
        if p[0] not in psmIds:
            continue
        for j, tkSet in enumerate(testIndSets):
            if i in tkSet:
                submatrixTestIndices[j].append(i)
    metrics = ['euclidean', 'cosine', 'correlation', 'jaccard']
    for kFold, (s,cvBinSids) in enumerate(zip(submatrixTestIndices,trainKeys)):
        # form submatrix, A, of testing PSMs
        # gather matrix, B, of training PSMs
        print("Fold %d: %s test instances, %d train instances" % (kFold, len(s), len(cvBinSids)))
        taq, daq, _ = calcQ(scores[kFold], Y[cvBinSids], thresh, True)
        gd = getDecoyIdx(Y, cvBinSids)
        trainSids = gd + taq
        # calculate similarity between A,B
        z = calcDistanceMat(X[s],X[trainSids], 'euclidean')
        excludePsms = 'kim_excludeHighlyCorreleatedPsmsList.txt'
        with open(excludePsms, 'w') as f:
            for ind, zrow in enumerate(z.T):
                if sum(zrow > 10.):
                    psmId = pepstrings[trainSids[ind]]
                    f.write("%s" % (psmId))
        # for m in metrics:
        #     print(m)
        #     z = calcDistanceMat(X[s],X[trainSids], m)
        #     similarityPlot(z,kFold, False, m)

def similarityPlot(z, fold, distMat=False, base = 'L2', bins = 500, prob = False):
    if distMat:
        output = 'fold' + str(fold) + '_' + base + 'DistanceMatrix.pdf'
        matplotlib.pyplot.matshow(z)
        matplotlib.pyplot.show()
        matplotlib.pyplot.savefig(output)
        matplotlib.pyplot.close()
    else:
        output = 'fold' + str(fold) + '_' + base + 'DistanceMatrixHist.pdf'
        pylab.clf()
        pylab.xlabel(base)
        pylab.ylabel('Frequency')
        if prob:
            pylab.ylabel('Pr(' + base +  ')')
        
        _, _, h1 = pylab.hist(z.flatten(), bins = bins)
        pylab.savefig('%s' % output)



def load_percolator_output(filename, scoreKey = "score", maxPerSid = False, idKey = "PSMId"):
    """ filename - percolator tab delimited output file
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    if not maxPerSid:
        with open(filename, 'r') as f:
            scores = []
            ids = []
            for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
                scores.append(float(l[scoreKey]))
                ids.append(l[idKey])
            # scores = [float(l[scoreKey]) for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
            return scores, ids
    f = open(filename)
    reader = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
    scoref = lambda r: float(r[scoreKey])
    # add all psms
    psms = {}
    ids = {}
    for psmid, rows in itertools.groupby(reader, lambda r: r["PSMId"]):
        records = list(rows)
        l = psmid.split('_')
        sid = int(l[2])
        charge = int(l[3])
        ids[sid] = psmid
        if sid in psms:
            psms[sid] += records
        else:
            psms[sid] = records
    f.close()
    max_scores = []
    max_ids = []
    # take max over psms
    for sid in psms:
        top_psm = max(psms[sid], key = scoref)
        max_scores.append(float(top_psm[scoreKey]))
        max_ids.append(ids[sid])
    return max_scores, max_ids


def load_percolator_target_decoy_files(filenames, scoreKey = "score", maxPerSid = False):
    """ filenames - list of percolator tab delimited target and decoy files
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    # Load target and decoy PSMs
    targets, tids = load_percolator_output(filenames[0], scoreKey, maxPerSid)
    decoys, dids = load_percolator_output(filenames[1], scoreKey, maxPerSid)
    scores = targets + decoys
    ids = tids + dids
    labels = [1]*len(targets) + [-1]*len(decoys)
    return scores, labels, ids

def load_percolator_target_decoy_files_tdc(filenames,  mapIdToScanMass, mapScanMassToId,
                                           scoreKey = "score", maxPerSid = False,
                                           writeOutput = False, outputDirectory = None, 
                                           percolatorTdcFile = 'percolatorTdc.txt'):
    """ 
        Take max PSM for each (scan, expmass) pair.  Check whether PIN file contains a mix of TDC and mix-max
        results, only perform competition for (scan, expmass) pairs with at least one target and one decoy PSM
        supplied.

    filenames - list of percolator tab delimited target and decoy files
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    # Load target and decoy PSMs
    targets, tids = load_percolator_output(filenames[0], scoreKey, maxPerSid)
    decoys, dids = load_percolator_output(filenames[1], scoreKey, maxPerSid)
    allScores = {}
    targetCheck = set([]) # check whether both a target and decoy were supplied
    decoyCheck = set([])
    for t,tid in zip(targets,tids):
        scanMassPair = mapIdToScanMass[tid]
        if scanMassPair in allScores:
            allScores[scanMassPair].append((t,1))
        else:
            allScores[scanMassPair] = [(t,1)]
            targetCheck.add(scanMassPair)

    for d,did in zip(decoys,dids):
        scanMassPair = mapIdToScanMass[did]
        decoyCheck.add(scanMassPair)
        if scanMassPair in allScores:
            allScores[scanMassPair].append((d,-1))
        else:
            allScores[scanMassPair] = [(d,-1)]

    print(len(decoyCheck), len(targetCheck), len(allScores))
    scores = []
    labels = []
    if writeOutput: # Write new output file, avoid rerunning tdc post-processing
        if not outputDirectory: # no output directory specified, just skip writing
            print("Write tdc output selected but no output directory specified, moving on without writing.")
            writeOutput = False
        else:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
            # Use same filename, place in new directory
            fid = open(os.path.join(outputDirectory, percolatorTdcFile), 'w')
            fid.write("%s\t%s\t%s\n" % ('PSMId', 'score', 'label'))

    for scanMassPair in allScores:
        if scanMassPair in targetCheck and scanMassPair in decoyCheck:
            s,l = max(allScores[scanMassPair])
            scores.append(s)
            labels.append(l)
            psmid = mapScanMassToId[scanMassPair]
            if writeOutput:
                fid.write("%s\t%f\t%d\n" % (psmid, s, l))
    if writeOutput:
        fid.close()
    print("Read %d scores" % (len(scores)))
    return scores, labels

def load_percolator_target_decoy_files_bucket_tdc(filenames,  mapIdToScanMass, mapScanMassToId,
                                                  scoreKey = "score", maxPerSid = False,
                                                  writeOutput = False, outputDirectory = None, 
                                                  percolatorTdcFile = 'percolatorTdc.txt'):
    """ Take max PSM for each (scan, expmass) pair.  This skips checking whether PIN file contains a mix of TDC and mix-max
        results.

    filenames - list of percolator tab delimited target and decoy files
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    # Load target and decoy PSMs
    targets, tids = load_percolator_output(filenames[0], scoreKey, maxPerSid)
    decoys, dids = load_percolator_output(filenames[1], scoreKey, maxPerSid)
    allScores = {}
    targetCheck = set([]) # check whether both a target and decoy were supplied
    decoyCheck = set([])
    for t,tid in zip(targets,tids):
        scanMassPair = mapIdToScanMass[tid]
        if scanMassPair in allScores:
            if t > allScores[scanMassPair][0]:
                allScores[scanMassPair] = (t,1)
        else:
            allScores[scanMassPair] = (t,1)

    for d,did in zip(decoys,dids):
        scanMassPair = mapIdToScanMass[did]
        if scanMassPair in allScores:
            if d > allScores[scanMassPair][0]:
                allScores[scanMassPair] = (d,-1)
        else:
            allScores[scanMassPair] = (d,-1)

    scores = []
    labels = []
    if writeOutput: # Write new output file, avoid rerunning tdc post-processing
        if not outputDirectory: # no output directory specified, just skip writing
            print("Write tdc output selected but no output directory specified, moving on without writing.")
            writeOutput = False
        else:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
            # Use same filename, place in new directory
            fid = open(os.path.join(outputDirectory, percolatorTdcFile), 'w')
            fid.write("%s\t%s\t%s\n" % ('PSMId', 'score', 'label'))

    for scanMassPair in allScores:
        s,l = allScores[scanMassPair]
        scores.append(s)
        labels.append(l)
        psmid = mapScanMassToId[scanMassPair]
        if writeOutput:
            fid.write("%s\t%f\t%d\n" % (psmid, s, l))
    if writeOutput:
        fid.close()

    numTargets = np.sum([l==1 for l in labels])
    numDecoys = len(scores) - numTargets
    print("Performed target-decoy competition, loaded %d target and %d decoy PSMs" % (numTargets, numDecoys))
    return scores, labels

def load_percolator_target_decoy_files_tdc_crux(filenames, scoreKey = "score", maxPerSid = False,
                                                targetString = 'target', decoyString = 'decoy'):
    """ filenames - list of percolator tab delimited target and decoy files
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    # Load target and decoy PSMs
    targets, tids = load_percolator_output(filenames[0], scoreKey, maxPerSid)
    decoys, dids = load_percolator_output(filenames[1], scoreKey, maxPerSid)
    allScores = {}
    targetCheck = set([]) # check whether both a target and decoy were supplied
    decoyCheck = set([])
    for t,tid in zip(targets,tids):
        psmid = tid.split(targetString)[1]
        if psmid in allScores:
            allScores[psmid].append((t,1))
        else:
            allScores[psmid] = [(t,1)]
            targetCheck.add(psmid)

    for d,did in zip(decoys,dids):
        psmid = did.split(decoyString)[1]
        if psmid in allScores:
            allScores[psmid].append((d,-1))
            decoyCheck.add(psmid)
        else:
            allScores[psmid] = [(d,-1)]
            decoyCheck.add(psmid)

    print(len(decoyCheck), len(targetCheck), len(allScores))
    scores = []
    labels = []
    for psmid in allScores:
        if psmid in targetCheck and psmid in decoyCheck:
            s,l = max(allScores[psmid]) # , key = lambda r: r[0])
            scores.append(s)
            labels.append(l)
    print("Read %d scores" % (len(scores)))
    return scores, labels


def parse_arg(argument):
    """Parse positional arguments of the form 'desc:scoreKey:filename'
    """
    result = argument if not isinstance(argument, str) else argument.split(':')
    if not result or len(result) < 3:
        err_print('Argument %s not correctly specified.' % argument)
        exit(-2)
    else:
        label = result[0]
        scoreKey = result[1]
        fns = []
        fn = os.path.expanduser(result[2])
        if not os.path.exists(fn):
            err_print('%s does not exist.' % fn)
            exit(-3)
        fns.append(fn)
        if len(result) == 4:
            fn = os.path.expanduser(result[3])
            if not os.path.exists(fn):
                err_print('%s does not exist.' % fn)
                exit(-3)
            fns.append(fn)
        return (label, scoreKey, fns)

def load_pin_scores(filename, scoreKey = "score", labelKey = "Label", idKey = "PSMId"):
    scores = []
    labels = []
    ids = []
    lineNum = 0
    with open(filename, 'r') as f:
        for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
            lineNum += 1
            label = int(l[labelKey])
            if label != 1 and label != -1:
                raise ValueError('Labels must be either 1 or -1, encountered value %d in line %d\n' % (label, lineNum))
            labels.append(label)
            scores.append(float(l[scoreKey]))
            if idKey in l:
                ids.append(l[idKey])
            else:
                if 'SpecId' in l:
                    idKey = 'SpecId'
                    ids.append(l[idKey])
    print("Read %d scores" % (lineNum-1))
    return scores, labels, ids

def load_pin_scores_tdc(filename, mapIdToScanMass, mapScanMassToId,
                        scoreKey = "score", labelKey = "Label", idKey = "PSMId",
                        writeOutput = False, outputDirectory = None):
    """ Take max PSM for each (scan, expmass) pair.  Check whether PIN file contains a mix of TDC and mix-max
        results, only perform competition for (scan, expmass) pairs with at least one target and one decoy PSM
        supplied.
    """
    allScores = {}
    targetCheck = set([]) # simple check to make sure we have both a target and decoy PSM per spectrum
    decoyCheck = set([]) # simple check to make sure we have both a target and decoy PSM per spectrum

    lineNum = 0
    with open(filename, 'r') as f:
        for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
            lineNum += 1
            # todo: add try excepts to the below
            s = float(l[scoreKey])
            label = int(l[labelKey])
            # make sure idKey is valid
            if idKey in l:
                psmid = l[idKey]
            else:
                if 'SpecId' in l:
                    idKey = 'SpecId'
                    psmid = l[idKey]
                else:
                    raise ValueError("idKey is invalid, neither SpecId or PSMId.  Exitting")
                    exit(-1)

            scanMassPair = mapIdToScanMass[psmid]
            if label == 1:
                targetCheck.add(scanMassPair)
                if scanMassPair in allScores:
                    allScores[scanMassPair].append((s,1))
                else:
                    allScores[scanMassPair] = [(s,1)]
            elif label == -1:
                decoyCheck.add(scanMassPair)
                if scanMassPair in allScores:
                    allScores[scanMassPair].append((s,-1))
                else:
                    allScores[scanMassPair] = [(s,-1)]
            else:
                raise ValueError('Labels must be either 1 or -1, encountered value %d in line %d\n' % (label, lineNum))
    print("Read %d scores" % (lineNum-1))
    # Perform the competition
    scores = []
    labels = []
    ids = []
    if writeOutput: # Write new output file, avoid rerunning tdc post-processing
        if not outputDirectory: # no output directory specified, just skip writing
            print("Write tdc output selected but no output directory specified, moving on without writing.")
            writeOutput = False
        else:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
            # Use same filename, place in new directory
            fid = open(os.path.join(outputDirectory, os.path.basename(filename)), 'w')
            fid.write("%s\t%s\t%s\n" % (idKey, scoreKey, labelKey))
    for scanMassPair in allScores:
        if scanMassPair in targetCheck and scanMassPair in decoyCheck:
            s,l = max(allScores[scanMassPair], key = lambda r: r[0])
            psmid = mapScanMassToId[scanMassPair]
            scores.append(s)
            labels.append(l)
            ids.append(psmid)
            if writeOutput:
                fid.write("%s\t%f\t%d\n" % (psmid, s, l))
    if writeOutput:
        fid.close()
    return scores, labels, ids

def load_pin_scores_bucket_tdc(filename, mapIdToScanMass, mapScanMassToId,
                               scoreKey = "score", labelKey = "Label", idKey = "PSMId",
                               writeOutput = False, outputDirectory = None):
    """ Take max PSM for each (scan, expmass) pair.  This skips checking whether PIN file contains a mix of TDC and mix-max
        results.
    """
    allScores = {}
    targetCheck = set([]) # simple check to make sure we have both a target and decoy PSM per spectrum
    decoyCheck = set([]) # simple check to make sure we have both a target and decoy PSM per spectrum

    lineNum = 0
    # Perform the competition
    with open(filename, 'r') as f:
        for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
            lineNum += 1
            # todo: add try excepts to the below
            s = float(l[scoreKey])
            label = int(l[labelKey])
            # make sure idKey is valid
            if idKey in l:
                psmid = l[idKey]
            else:
                if 'SpecId' in l:
                    idKey = 'SpecId'
                    psmid = l[idKey]
                else:
                    raise ValueError("idKey is invalid, neither SpecId or PSMId.  Exitting")
                    exit(-1)

            scanMassPair = mapIdToScanMass[psmid]
            if scanMassPair in allScores:
                if s > allScores[scanMassPair][0]:
                    allScores[scanMassPair] = (s,label)
            else:
                allScores[scanMassPair] = (s,label)
    print("Read %d scores" % (lineNum-1))
    # Perform the competition
    scores = []
    labels = []
    ids = []
    if writeOutput: # Write new output file, avoid rerunning tdc post-processing
        if not outputDirectory: # no output directory specified, just skip writing
            print("Write tdc output selected but no output directory specified, moving on without writing.")
            writeOutput = False
        else:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
            # Use same filename, place in new directory
            fid = open(os.path.join(outputDirectory, os.path.basename(filename)), 'w')
            fid.write("%s\t%s\t%s\n" % (idKey, scoreKey, labelKey))
    for scanMassPair in allScores:
        s,l = allScores[scanMassPair]
        psmid = mapScanMassToId[scanMassPair]
        scores.append(s)
        labels.append(l)
        ids.append(psmid)
        if writeOutput:
            fid.write("%s\t%f\t%d\n" % (psmid, s, l))
    if writeOutput:
        fid.close()
    return scores, labels, ids

def load_pin_scores_tdc_crux(filename, scoreKey = "score", labelKey = "Label", idKey = "PSMId",
                             targetString = 'target', decoyString = 'decoy'):
    allScores = {}
    targetCheck = set([]) # simple has to make sure we have both a target and decoy PSM per spectrum
    decoyCheck = set([]) # simple has to make sure we have both a target and decoy PSM per spectrum

    lineNum = 0
    with open(filename, 'r') as f:
        for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
            lineNum += 1
            # todo: add try excepts to the below
            s = float(l[scoreKey])
            label = int(l[labelKey])
            # make sure idKey is valid
            if idKey in l:
                psmid = l[idKey]
            else:
                if 'SpecId' in l:
                    idKey = 'SpecId'
                    psmid = l[idKey]
                else:
                    raise ValueError("idKey is invalid, neither SpecId or PSMId.  Exitting")
                    exit(-1)

            if label == 1:
                psmid = psmid.split(targetString)[1]
                if psmid in allScores:
                    allScores[psmid].append((s,1))
                else:
                    allScores[psmid] = [(s,1)]
                    targetCheck.add(psmid)
            elif label == -1:
                psmid = psmid.split(decoyString)[1]
                if psmid in allScores:
                    allScores[psmid].append((s,-1))
                    decoyCheck.add(psmid)
                else:
                    allScores[psmid] = [(s,-1)]
                    decoyCheck.add(psmid)
            else:
                raise ValueError('Labels must be either 1 or -1, encountered value %d in line %d\n' % (label, lineNum))
    print("Read %d scores" % (lineNum-1))
    # Perform the competition
    scores = []
    labels = []
    ids = []
    for psmid in allScores:
        if psmid in targetCheck and psmid in decoyCheck:
            s,l = max(allScores[psmid], key = lambda r: r[0])
            scores.append(s)
            labels.append(l)
            ids.append(psmid)
    return scores, labels, ids

def postprocess_results_tdcOrMixmax(pinfile, ptDirectory, outputDirectory,
                                    scoreKey = "score", labelKey = "Label", idKey = "PSMId"):
    """ Post-process proteoTorch results to either compute target-decoy competition or (todo) mix-max method
        for q-value esimtation

        todo: add mixmax procedure
    """

    pepstrings, _, _, _, sids, expMasses = load_pin_return_featureMatrix(pinfile, normalize = False)
    # determine mapping from PSMId/SpecId to (scannr, expmass)
    mapIdToScanMass = {}
    mapScanMassToId = {}
    for p, s, em in zip(pepstrings,sids,expMasses):
        psmId = p[0]
        mapIdToScanMass[psmId] = (s,em)
        mapScanMassToId[(s,em)] = psmId
    print("%d unique Sids, %d unique exp masses" % (len(set(sids)), len(set(expMasses))))
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    for f in os.listdir(ptDirectory):
        if 'output' in f:
            scores, labels, ids = load_pin_scores(os.path.join(ptDirectory, f), scoreKey, labelKey, idKey)
            allScores = {}
            lineNum = 0
            targetCheck = set([]) # simple has to make sure we have both a target and decoy PSM per spectrum
            decoyCheck = set([]) # simple has to make sure we have both a target and decoy PSM per spectrum
            for s,l,i in zip(scores,labels,ids):
                lineNum += 1
                scanMassPair = mapIdToScanMass[i]
                if l==1:
                    targetCheck.add(scanMassPair)
                    if scanMassPair in allScores:
                        allScores[scanMassPair].append((s,1))
                    else:
                        allScores[scanMassPair] = [(s,1)]
                elif l==-1:
                    decoyCheck.add(scanMassPair)
                    if scanMassPair in allScores:
                        allScores[scanMassPair].append((s,-1))
                    else:
                        allScores[scanMassPair] = [(s,-1)]
                else:
                    raise ValueError('Labels must be either 1 or -1, encountered value %d in line %d\n' % (label, lineNum))
            print("Input file %s: read %d scores" % (os.path.join(ptDirectory, f),lineNum-1))
            # Perform the competition
            with open(os.path.join(outputDirectory, f), 'w') as fid:
                lineNum = 0
                # Write header: (1) id (2) score (3) label
                fid.write("%s\t%s\t%s\t%s\t%s\n" % (idKey, scoreKey, labelKey, "ScanNr", "ExpMass"))
                for scanMassPair in allScores:
                    if scanMassPair in targetCheck and scanMassPair in decoyCheck:
                        lineNum += 1
                        s,l = max(allScores[scanMassPair], key = lambda r: r[0])
                        psmid = mapScanMassToId[scanMassPair] 
                        fid.write("%s\t%f\t%d\t%d\t%s\n" % (psmid, s, l,scanMassPair[0], scanMassPair[1]))
                print("Output file %s: wrote %d scores" % (os.path.join(outputDirectory, f), lineNum))

def psmId_diffs(inputPin, percFiles, pinFile, percScoreKey = 'score', pinScoreKey = 'score',
                labelKey = "Label", idKey = "PSMId"):
    """ Check differences in PSM ids between output files
    """
    pepstrings, _, _, _, sids, expMasses = load_pin_return_featureMatrix(inputPin, normalize = False)
    # determine mapping from PSMId/SpecId to (scannr, expmass)
    mapIdToScanMass = {}
    mapScanMassToId = {}
    for p, s, em in zip(pepstrings,sids,expMasses):
        psmId = p[0]
        mapIdToScanMass[psmId] = (s,em)
        mapScanMassToId[(s,em)] = psmId
    print("%d unique Sids, %d unique exp masses" % (len(set(sids)), len(set(expMasses))))

    _, _, percIds = load_percolator_target_decoy_files(percFiles, percScoreKey)
    _, _, pinIds = load_pin_scores(pinFile, pinScoreKey)
    percScans = set([mapIdToScanMass[psmId][0] for psmId in percIds])
    pinScans = set([mapIdToScanMass[psmId][0] for psmId in pinIds])
    print("%d ids in %s,%s not in %s:" % (len(percScans - pinScans), percFiles[0], percFiles[1], pinFile))
    for sid in percScans - pinScans:
        print("%d" % (sid))

    print("%d ids in %s not in %s,%s:" % (len(pinScans - percScans), pinFile, percFiles[0], percFiles[1]))
    for sid in pinScans - percScans:
        print("%d" % (sid))
    

def load_test_scores(filenames, scoreKey = 'score', qTol = 0.01, qCurveCheck = 0.001, 
                     tdc = False, psmIdToScanMass = {}, scanMassTopsmId = {},
                     writeTdc = False, tdcDir = ''):
    """ Load all PSMs and features file
    """
    if len(filenames)==1:
        if tdc:
            scores, labels, _ = load_pin_scores_bucket_tdc(filenames[0], psmIdToScanMass, scanMassTopsmId, scoreKey,
                                                           writeOutput = writeTdc, outputDirectory = tdcDir)
        else:
            scores, labels, _ = load_pin_scores(filenames[0], scoreKey)
    elif len(filenames)==2: # Assume these are Percolator results, where target is specified followed by decoy
        if tdc:
            scores, labels = load_percolator_target_decoy_files_bucket_tdc(filenames, psmIdToScanMass, scanMassTopsmId, scoreKey,
                                                                           writeOutput = writeTdc, outputDirectory = tdcDir)
        else:
            scores, labels, _ = load_percolator_target_decoy_files(filenames, scoreKey)
    else:
        raise ValueError('Number of filenames supplied for a single method was > 2, exitting.\n')
    # Check scores multiplied by both 1 and positive -1
    taq, _, _ = calcQ(scores, labels, qTol, False)
    taq2, _, _ = calcQ([-1. * s for s in scores], labels, qTol, False)

    if(len(taq)>len(taq2)):
        qs, ps = calcQAndNumIdentified(scores, labels)
    else:
        qs, ps = calcQAndNumIdentified([-1. * s for s in scores], labels)
    numIdentifiedAtQ = 0
    quac = []
    den = float(len(scores))
    ind0 = -1    
    for ind, (q, p) in enumerate(zip(qs, ps)):
        if q > qTol:
            break
        numIdentifiedAtQ = float(p)
        quac.append(numIdentifiedAtQ / den)
        if q < qCurveCheck:
            ind0 = ind
    # print "Accuracy = %f%%" % (numIdentifiedAtQ / float(len(qs)) * 100)
    # set AUC weights to uniform 
    # auc = np.trapz(quac)#/len(quac)#/quac[-1]
    # if qTol > qCurveCheck:
    #     auc = 0.3 * auc + 0.7 * np.trapz(quac)#/ind0#/quac[ind0-1]
    return qs, ps, numIdentifiedAtQ


#############################################
# Scoring method plotting utilities
#############################################


def plot(scorelists, output, qrange = 0.1, labels = None, **kwargs):
    linestyle = [ '-', '-', '-', '-', '-', '-', '-', '-', '-' ]
    linecolors = [  (0.0, 0.0, 0.0),
                    (0.8, 0.4, 0.0),
                    (0.0, 0.45, 0.70),
                    (0.8, 0.6, 0.7),
                    (0.0, 0.6, 0.5),
                    (0.9, 0.6, 0.0),
                    (0.95, 0.9, 0.25), 
                    (0.35, 0.7, 0.9),
                    (0.43, 0.17, 0.60)]
    xlabel = 'q-value'
    ylabel = 'Spectra identified'
    pylab.clf()
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    # pylab.nipy_spectral()
    pylab.gray()
    h = -1
    for i,(q,p) in enumerate(scorelists):
        h = max(itertools.chain([h], (b for a, b in zip(q, p) if a <= qrange)))
        pylab.plot(q, p, color = linecolors[i], linewidth = 2, linestyle = linestyle[i])
    pylab.xlim([0,qrange])
    pylab.ylim([0, h])
    pylab.legend(labels, loc = 'lower right')
    pylab.savefig(output, bbox_inches='tight')

def scatterDecoyRanks(ranksA, ranksB):
    fn = "decoyRanks.png"
    pylab.clf()
    pylab.scatter(ranksA, ranksB, color = 'b')
    pylab.xlabel("DeepMS")
    pylab.ylabel("Percolator")
    pylab.savefig(fn)

def disagreedDecoys(psmsA, labelsA, psmsB, labelsB, psmsIds,
                    outputFile,
                    threshA = 0.9, threshB = 0.8):
    # Filter out decoy PSMs for which scoring method A assigns scores with rank >= threshA and method B 
    # assigns scores with rank <= 0.5
    assert(len(psmsA)==len(psmsB))

    # g = open('decoyRanks', 'w')
    with open(outputFile, 'w') as f:
        denom = float(len(psmsA))
        ranksA = {}
        for i, (score, label, psmId) in enumerate(sorted(zip(psmsA, labelsA, psmsIds), key = lambda x: x[0])):
            ranksA[psmId] = float(i) / denom

        # rA = []
        # rB = []
        counter=0
        f.write("PSMId\n")
        for i, (score, label, psmId) in enumerate(sorted(zip(psmsB, labelsB, psmsIds), key = lambda x: x[0])):
            if(label==-1):
                rankB = i / denom
                rankA = ranksA[psmId]
                # g.write("%f\t%f\n" % (rankA, rankB))
                if(rankB <= threshB and rankA >= threshA):
                    f.write("%s\n" % psmId)
                    counter += 1

        print("%d decoy PSMs where method A ranks >= %f and method B ranks <= %f" % (counter, threshA, threshB))

                # rA.append(rankA)
                # rB.append(rankB)
        # scatterDecoyRanks(rA, rB)

    # g.close()
    
def decileInfo(scores, labels):
    # Print target/decoy ratios per decile
    decileRatios = [0.] * 10
    decileTargets = [0.] * 10
    decileDecoys = [0.] * 10
    inc = int(np.ceil(float(len(scores)) / 10.))
    decile = inc

    currDecile = 0
    numTargetsInDecile = 0
    numDecoysInDecile = 0
    for ind, (score,label) in enumerate(sorted(zip(scores, labels), key = lambda x: x[0])):
        if ind < decile:
            if label==1:
                numTargetsInDecile += 1
            else:
                numDecoysInDecile += 1
        else:
            decileTargets[currDecile] = numTargetsInDecile
            decileDecoys[currDecile] = numDecoysInDecile
            decileRatios[currDecile] = float(numTargetsInDecile) / float(numDecoysInDecile)
            currDecile += 1
            decile += inc
            numTargetsInDecile = 0
            numDecoysInDecile = 0
    decileTargets[currDecile] = numTargetsInDecile
    decileDecoys[currDecile] = numDecoysInDecile
    decileRatios[currDecile] = float(numTargetsInDecile) / float(numDecoysInDecile)

    print("Target/decoy info per decile")
    print("Decile\t#Targets/#Decoys\t#Targets\t#Decoys")
    for i, (r, t, d) in enumerate(zip(decileRatios, decileTargets, decileDecoys)):
        print("%d\t%f\t%d\t%d" % (i, r, t, d))

def refineDms(deepMsFile):
    # load scores and take max over unique PSM ids
    scores, labels, ids = load_pin_scores(deepMsFile)
    print("DeepMS decile info")
    decileInfo(scores, labels)
    decoys = {}
    targets = {}
    # take max per PSM id
    for s,l,i in zip(scores,labels,ids):
        if l==1:
            if i in targets:
                if s > targets[i]:
                    targets[i] = s
            else:
                targets[i] = s
        else:
            if i in decoys:
                if s > decoys[i]:
                    decoys[i] = s
            else:
                decoys[i] = s
    print("Read %d deepMS Target PSMs, %d deepMS Decoy PSMs" % (len(targets), len(decoys)))
    return targets, decoys


def refinePerc(percolatorTargetFile, percolatorDecoyFile):
    # load percolator target and decoy files and take max over unique PSM ids
    target_scores, target_ids = load_percolator_output(percolatorTargetFile)
    decoy_scores, decoy_ids = load_percolator_output(percolatorDecoyFile)
    print("Percolator decile info")
    decileInfo(target_scores + decoy_scores, [1]*len(target_scores) + [-1]*len(decoy_scores))

    print("Read %d Percolator Target PSMs, %d Percolator Decoy PSMs" % (len(target_ids), len(decoy_ids)))
    targets = {}
    decoys = {}
    # first take max over target PSMs, filtering out PSMs not in targetIntersect
    for s,i in zip(target_scores, target_ids):
        if i in targets:
            if s > targets[i]:
                targets[i] = s
        else:
            targets[i] = s
    # next decoys
    for s,i in zip(decoy_scores, decoy_ids):
        if i in decoys:
            if s > decoys[i]:
                decoys[i] = s
        else:
            decoys[i] = s
    return targets, decoys


def histogram(targets, decoys, output, bins = 40, prob = False):
    """Histogram of the score distribution between target and decoy PSMs.
    Arguments:
        targets: Iterable of floats, each the score of a target PSM.
        decoys: Iterable of floats, each the score of a decoy PSM.
        fn: Name of the output file. The format is inferred from the
            extension: e.g., foo.png -> PNG, foo.pdf -> PDF. The image
            formats allowed are those supported by matplotlib: png,
            pdf, svg, ps, eps, tiff.
        bins: Number of bins in the histogram [default: 40].
    Effects:
        Outputs the image to the file specified in 'output'.
    """
    pylab.clf()
    pylab.xlabel('Score')
    pylab.ylabel('Frequency')
    if prob:
        pylab.ylabel('Pr(Score)')

    l = min(min(decoys), min(targets))
    h = max(max(decoys), max(targets))
    _, _, h1 = pylab.hist(targets, bins = bins, range = (l,h), density = prob,
                          color = 'b', alpha = 0.25)
    _, _, h2 = pylab.hist(decoys, bins = bins, range = (l,h), density = prob,
                          color = 'm', alpha = 0.25)
    pylab.legend((h1[0], h2[0]), ('Target Scores', 'Decoy Scores'), loc = 'best')
    pylab.savefig('%s' % output)


def scatterplot(deepMsFile, percolatorTargetFile, percolatorDecoyFile, fn, plotLabels = None):
    """Scatterplot of the PSM scores for deepMS and Percolator.
    """
    # Gather intersection of target/decoy PSMs between the two methods
    dms_targetDict, dms_decoyDict = refineDms(deepMsFile)
    perc_targetDict, perc_decoyDict = refinePerc(percolatorTargetFile, percolatorDecoyFile)
    # Plot histograms for scoring distributions
    baseFileName = os.path.splitext(fn)[0]
    if plotLabels:
        histA = baseFileName + plotLabels[0] + 'Hist.png'
        histB = baseFileName + plotLabels[1] + 'Hist.png'
    else:
        histA = baseFileName + "deepMsHist.png"
        histB = baseFileName + "percolatorHist.png"
    histogram(dms_targetDict.values(), dms_decoyDict.values(), histA, 100)
    histogram(perc_targetDict.values(), perc_decoyDict.values(), histB, 100)
    target_ids = list(set(dms_targetDict) & set(perc_targetDict))
    decoy_ids = list(set(dms_decoyDict) & set(perc_decoyDict))
    t1 = [dms_targetDict[t] for t in target_ids]
    d1 = [dms_decoyDict[d] for d in decoy_ids]
    t2 = [perc_targetDict[t] for t in target_ids]
    d2 = [perc_decoyDict[d] for d in decoy_ids]
    
    threshA = 0.9
    threshB = 0.85
    disagreedOutputFile = baseFileName + "aThresh%.2f_bThresh%.2f_decoyPsms.txt" % (threshA, threshB)
    disagreedDecoys(t1+d1, [1]*len(t1) + [-1]*len(d1), 
                    t2+d2, [1]*len(t2) + [-1]*len(d2), target_ids+decoy_ids,
                    disagreedOutputFile, threshA, threshB)

    pylab.clf()
    pylab.scatter(t1, t2, color = 'b', alpha = 0.20, s = 2)
    pylab.scatter(d1, d2, color = 'r', alpha = 0.10, s = 1)
    pylab.xlim( (min(min(t1), min(d1)), max(max(t1), max(d1))) )
    if plotLabels:
        pylab.xlabel(plotLabels[0])
        pylab.ylabel(plotLabels[1])
    pylab.savefig(fn)

def feature_histograms(pin, psmIds, output_dir, bins = 40, prob = False):
    """ Plot histograms for all features in a given feature matrix
    """
    print(pin)
    _, X0, Y0, _, _ = load_pin_return_featureMatrix(pin, normalize = False)
    X, Y, featureNames = givenPsmIds_writePin(pin, psmIds)

    print(X.shape)
    try:
        os.mkdir(output_dir)
    except OSError:
        print("Failed to create output directory %s, exitting." % output_dir)

    for i, feature in enumerate(featureNames):
        output = output_dir + '/' + feature + '0.png'
        # First plot total feature histogram
        x = X0[:,i]
        targets = []
        decoys = []
        for x,l in zip(x,Y0):
            if l == 1:
                targets.append(x)
            else:
                decoys.append(x)

        pylab.clf()
        pylab.xlabel('Score')
        pylab.ylabel('Frequency')
        if prob:
            pylab.ylabel('Pr(Score)')

        l = min(min(decoys), min(targets))
        h = max(max(decoys), max(targets))
        _, _, h1 = pylab.hist(targets, bins = bins, range = (l,h), density = prob,
                              color = 'b', alpha = 0.25)
        _, _, h2 = pylab.hist(decoys, bins = bins, range = (l,h), density = prob,
                              color = 'm', alpha = 0.25)
        pylab.legend((h1[0], h2[0]), ('Target Scores', 'Decoy Scores'), loc = 'best')
        pylab.savefig('%s' % output)

        # Next, subset
        output = output_dir + '/' + feature + '.png'
        # First plot total feature histogram
        x = X[:,i]
        pylab.clf()
        pylab.xlabel('Score')
        pylab.ylabel('Frequency')
        if prob:
            pylab.ylabel('Pr(Score)')

        l = min(x)
        h = max(x)
        _, _, h1 = pylab.hist(x, bins = bins, range = (l,h), density = prob,
                              color = 'b')
        pylab.savefig('%s' % output)
    

def main(args, output, maxq, doTdc = False, dataset = None, writeTdcResults = False, tdcOutputDir = ''):
    '''
    input:
        
        list of tuples; each tuple contains either:
            
            ("Percolator", "score", "percTargets.txt", "percDecoys.txt")
            
            ("DNN", "score", "dnn_output.txt")
    '''
    methods = []
    scorelists = []
    if doTdc and not dataset:
        raise Exception('TDC post-processing selected, but analyzed PIN not supplied.  Exitting')

    mapIdToScanMass = {}
    mapScanMassToId = {}
    if doTdc:
        # Note: (scannr, expmass) pairs are used to determine PSMs for target-decoy competition
        # pepstrings, _, _, _, sids, expMasses = load_pin_return_featureMatrix(dataset, normalize = False)
        pepstrings, _, sids, expMasses = load_pin_return_scanExpmassPairs(dataset)
        # determine mapping from PSMId/SpecId to (scannr, expmass)
        for p, s, em in zip(pepstrings,sids,expMasses):
            psmId = p[0]
            mapIdToScanMass[psmId] = (s,em)
            mapScanMassToId[(s,em)] = psmId
        
    def process(arg):
        desc, scoreKey, fn = parse_arg(arg)
        methods.append(desc)
        qs, ps, naq = load_test_scores(fn, scoreKey, tdc = doTdc, 
                                       psmIdToScanMass = mapIdToScanMass, scanMassTopsmId = mapScanMassToId,
                                       writeTdc = writeTdcResults, tdcDir = tdcOutputDir)
        scorelists.append( (qs, ps) )
        print ('%s: %d identifications, %d identified at q <= 0.01' % (desc, len(qs), naq))
    for argument in args:
        process(argument)
    plot(scorelists, output, maxq, methods)



if __name__ == '__main__':
    usage = """Usage: %prog [options] label1:IDENT1 ... labeln:IDENTn\n\n
            
             Example for using on percolator:     python THIS_SCRIPT.py --output Perc.png "Percolator":score:percTargets.txt:percDecoys.txt
             Example for using on DNN classifier: python THIS_SCRIPT.py --output DNN.png "DNN":score:dnn_output.txt
             """             
    desc = ('The positional arguments IDENT1, IDENT2, ..., IDENTn are the '
            'names of spectrum identification files.')
    parser = optparse.OptionParser(usage = usage, description = desc)
    parser.add_option('--output', type = 'string', default='figure.png', help = 'Output file name where the figure will be stored.')
    parser.add_option('--maxq', type = 'float', default = 1.0, help = 'Maximum q-value to plot to: 0 < q <= 1.0')
    parser.add_option('--tdc', action = "store_true", help = 'Perform target-decoy competition')
    parser.add_option('--dataset', type = 'string', help = 'Original processed dataset in PIN format.  Only necessary if tdc or mixmax set to True.')
    parser.add_option('--writeTdcResults', action = "store_true", help = 'Write the results of TDC for all methods to new files.')
    parser.add_option('--tdcOutputDir', type = 'string', default = '', help = 'Output directory to write TDC competition results.')

    (OPTIONS, ARGS) = parser.parse_args()

    assert len(ARGS) >= 1, 'No identification and model output files listed.'
    main(ARGS, OPTIONS.output, OPTIONS.maxq, OPTIONS.tdc, OPTIONS.dataset, 
         OPTIONS.writeTdcResults, OPTIONS.tdcOutputDir)
    
