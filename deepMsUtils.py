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


def load_percolator_output(filename, scoreKey = "score", maxPerSid = False):
    """ filename - percolator tab delimited output file
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    if not maxPerSid:
        with open(filename, 'r') as f:
            return [float(l[scoreKey]) for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]

    f = open(filename)
    reader = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
    scoref = lambda r: float(r[scoreKey])
    # add all psms
    psms = {}
    for psmid, rows in itertools.groupby(reader, lambda r: r["PSMId"]):
        records = list(rows)
        l = psmid.split('_')
        sid = int(l[2])
        # charge = int(l[3])
        if sid in psms:
            psms[sid] += records
        else:
            psms[sid] = records
    f.close()
    max_scores = []
    # take max over psms
    for sid in psms:
        top_psm = max(psms[sid], key = scoref)
        max_scores.append(float(top_psm[scoreKey]))
    return max_scores


def load_percolator_target_decoy_files(filenames, scoreKey = "score", maxPerSid = False):
    """ filenames - list of percolator tab delimited target and decoy files
    header:
    (1)PSMId (2)score (3)q-value (4)posterior_error_prob (5)peptide (6)proteinIds
    Output:
    List of scores
    """
    # Load targets
    targets = load_percolator_output(filenames[0], scoreKey, maxPerSid)
    decoys = load_percolator_output(filenames[1], scoreKey, maxPerSid)
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


def load_pin_scores(filename, scoreKey = "score", labelKey = "Label"):
    scores = []
    labels = []
    lineNum = 0
    with open(filename, 'r') as f:
        print(f)
        for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True):
            lineNum += 1
            label = int(l[labelKey])
            if label != 1 and label != -1:
                raise ValueError('Labels must be either 1 or -1, encountered value %d in line %d\n' % (label, lineNum))
            labels.append(label)
            scores.append(float(l[scoreKey]))
    print("Read %d scores" % (lineNum-1))
    return scores, labels


def load_test_scores(filenames, scoreKey = 'score', is_perc=0, qTol = 0.01, qCurveCheck = 0.001):
    """ Load all PSMs and features file
    """
    if len(filenames)==1:
        scores, labels = load_pin_scores(filenames[0], scoreKey)
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
