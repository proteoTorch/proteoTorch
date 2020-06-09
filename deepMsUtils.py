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

#import operator
import itertools
import numpy

from deepMs import calcQAndNumIdentified #, _scoreInd, _labelInd, _indInd, _includeNegativesInResult


def load_percolator_output(filename,  scoreKey = "score", maxPerSid = False, idKey = "PSMId"):
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

def load_percolator_target_decoy_files(filenames,  scoreKey = "score", maxPerSid = False):
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
    result = argument.split(':')
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
  linecolors = [ (0.0, 0.0, 0.0),
                 (0.8, 0.4, 0.0),
                 (0.0, 0.45, 0.70),
                 (0.8, 0.6, 0.7),
                 (0.0, 0.6, 0.5),
                 (0.9, 0.6, 0.0),
                 (0.95, 0.9, 0.25)]
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

def refineDms(deepMsFile):
    # load scores and take max over unique PSM ids
    scores, labels, ids = load_pin_scores(deepMsFile)
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

def scatterplot(deepMsFile, percolatorTargetFile, percolatorDecoyFile, fn, plotLabels = None):
    """Scatterplot of the PSM scores for deepMS and Percolator.
    """
    dms_targetDict, dms_decoyDict = refineDms(deepMsFile)
    perc_targetDict, perc_decoyDict = refinePerc(percolatorTargetFile, percolatorDecoyFile)

    # Plot histograms for scoring distributions
    histogram(dms_targetDict.values(), dms_decoyDict.values(), "deepMsHist.png", 100)
    histogram(perc_targetDict.values(), perc_decoyDict.values(), "percolatorHist.png", 100)

    target_ids = list(set(dms_targetDict.iterkeys()) & set(perc_targetDict.iterkeys()))
    decoy_ids = list(set(dms_decoyDict.iterkeys()) & set(perc_decoyDict.iterkeys()))

    t1 = [dms_targetDict[t] for t in target_ids]
    d1 = [dms_decoyDict[d] for d in decoy_ids]
    t2 = [perc_targetDict[t] for t in target_ids]
    d2 = [perc_decoyDict[d] for d in decoy_ids]

    pylab.clf()
    pylab.scatter(t1, t2, color = 'b', alpha = 0.20, s = 2)
    pylab.scatter(d1, d2, color = 'r', alpha = 0.10, s = 1)
    pylab.xlim( (min(min(t1), min(d1)), max(max(t1), max(d1))) )
    if plotLabels:
        pylab.xlabel(plotLabels[0])
        pylab.ylabel(plotLabels[1])

    pylab.savefig(fn)

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
    # decoys = []
    # targets = []
    # for s,l in zip(scores,labels):
    #     if l==1:
    #         targets.append(s)
    #     else:
    #         decoys.append(s)

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

if __name__ == '__main__':

    usage = """Usage: %prog [options] label1:IDENT1 ... labeln:IDENTn\n\n
            
             Example for using on percolator:     python THIS_SCRIPT.py --output Perc.png "Percolator":score:percTargets.txt:percDecoys.txt
             Example for using on DNN classifier: python THIS_SCRIPT.py --output DNN.png "MLP":score:dnn_output.txt
             """
             
    desc = ('The positional arguments IDENT1, IDENT2, ..., IDENTn are the '
            'names of spectrum identification files.')

    parser = optparse.OptionParser(usage = usage, description = desc)

    parser.add_option('--output', type = 'string', default='figure.png', help = 'Output file name where the figure will be stored.')
    parser.add_option('--maxq', type = 'float', default = 1.0, help = 'Maximum q-value to plot to: 0 < q <= 1.0')
#    parser.add_option('--is_perc', type = 'int', default = 0, help = 'Bool: pass "1" if using this script on percolator output, otherwise 0.')

    (options, args) = parser.parse_args()

    assert len(args) >= 1, 'No identification and model output files listed.'

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
    plot(scorelists, options.output, options.maxq, methods)
