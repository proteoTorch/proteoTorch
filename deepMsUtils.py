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

from deepMs import calcQAndNumIdentified, givenPsmIds_writePin, load_pin_return_featureMatrix
 #, _scoreInd, _labelInd, _indInd, _includeNegativesInResult

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
    targets, _ = load_percolator_output(filenames[0], scoreKey, maxPerSid)
    decoys, _ = load_percolator_output(filenames[1], scoreKey, maxPerSid)
    scores = targets + decoys
    labels = [1]*len(targets) + [-1]*len(decoys)
    return scores, labels


def refine(scorelists, tdc = True):
    """Create a list of method scores which contain only shared spectra.

    Arguments:
       scorelists: List of [(targets,decoys)] pairs, where each of targets
           and decoys is itself an iterable of (sid, peptide, score) records.
           Each method is represented by a (targets,decoys) pair.

    Returns:
       newscorelists: List of [(targets,decoys)] pairs, where each of targets
           and decoys is a list of scores for the spectra that are scored in
           all of the methods.
    """
    # Find the sids common to all the methods.
    sids = [ ]
    for targets, decoys in scorelists:
        sids.append(set(r[0] for r in targets) & set(r[0] for r in decoys))
    final = sids[0]
    for ids in sids[1:]:
        final = final & ids
    # Filter the (sid, peptide, score) records to include those in sids.
    newscorelists = [ ]
    pred = lambda r: r[0] in final
    for targets, decoys in scorelists:
        if tdc:
            newtargets = [] 
            newdecoys = []
            for t,d in zip(sorted(list(itertools.ifilter(pred, targets))),sorted(list(itertools.ifilter(pred, decoys)))):
                if t[2] > d[2]:
                    newtargets.append(t)
                else:
                    newdecoys.append(d)
        else:
            newtargets = list(itertools.ifilter(pred, targets))
            newdecoys = list(itertools.ifilter(pred, decoys))
        newscorelists.append( (list(r[2] for r in newtargets),
                               list(r[2] for r in newdecoys)) )
    return newscorelists


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



def load_test_scores(filenames, scoreKey = 'score', is_perc=0, qTol = 0.01, qCurveCheck = 0.001):
    """ Load all PSMs and features file
    """
    if len(filenames)==1:
        scores, labels, _ = load_pin_scores(filenames[0], scoreKey)
    elif len(filenames)==2: # Assume these are Percolator results, where target is specified followed by decoy
        scores, labels = load_percolator_target_decoy_files(filenames, scoreKey)
    else:
        raise ValueError('Number of filenames supplied for a single method was > 2, exitting.\n')
    qs, ps = calcQAndNumIdentified(scores, labels)
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
    auc = np.trapz(quac)#/len(quac)#/quac[-1]
    if qTol > qCurveCheck:
        auc = 0.3 * auc + 0.7 * np.trapz(quac[:ind0])#/ind0#/quac[ind0-1]
    return qs, ps, auc


#############################################
# Scoring method plotting utilities
#############################################


def plot(scorelists, output, qrange = 0.1, labels = None, **kwargs):
    linestyle = [ '-', '-', '-', '-', '-', '-', '-', '-' ]
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

def main(args, output, maxq):
    '''
    input:
        
        list of tuples; each tuple contains either:
            
            ("Percolator", "score", "percTargets.txt", "percDecoys.txt")
            
            ("DNN", "score", "dnn_output.txt")
    '''
    methods = []
    scorelists = []
    def process(arg, silent = False):
        desc, scoreKey, fn = parse_arg(arg)
        methods.append(desc)
        qs, ps, auc = load_test_scores(fn, scoreKey)
        scorelists.append( (qs, ps) )
        print ('%s: %d identifications, AUC = %f' % (desc, len(qs), auc))
    for argument in args:
        process(argument)
    # scorelists = refine(scorelists)
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
#    parser.add_option('--is_perc', type = 'int', default = 0, help = 'Bool: pass "1" if using this script on percolator output, otherwise 0.')

    (OPTIONS, ARGS) = parser.parse_args()

    assert len(ARGS) >= 1, 'No identification and model output files listed.'
    main(ARGS, OPTIONS.output, OPTIONS.maxq)
    
