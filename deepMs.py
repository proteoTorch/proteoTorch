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
from sklearn import preprocessing
from pprint import pprint
import util.args
import util.iterables
import struct
import array

from random import shuffle
from scipy import linalg, stats
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import mixture

#########################################################
################### Global variables
#########################################################
_debug=True
_verb=1
_mergescore=True
_includeNegativesInResult=True
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
    # all: list of triples consisting of score, label, and index
    all = zip(scores,labels, range(len(scores)))
    #--- sort descending
    all.sort( key = lambda r: -r[0])
    pi0 = 1.
    qvals = getQValues(pi0, all, skipDecoysPlusOne)

    # Calculate minimum score which achieves q-value thresh
    u = all[0][scoreInd]
    for idx, q in enumerate(qvals):
        if q > thresh:
            break
        u = all[idx][scoreInd]

    # find median decoy score
    dScores = [score for score,l in zip(scores,labels) if l != 1]
    d = sorted(dScores)[len(dScores) / 2]
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
    # all: list of triples consisting of score, label, and index
    all = zip(scores,labels, range(len(scores)))
    #--- sort descending
    all.sort( key = lambda r: -r[0])
    pi0 = 1.
    qvals = getQValues(pi0, all, skipDecoysPlusOne)
    
    taq = []
    daq = []
    for idx, q in enumerate(qvals):
        if q > thresh:
            break
        else:
            curr_label = all[idx][1]
            curr_og_idx = all[idx][2]
            if curr_label == 1:
                taq.append(curr_og_idx)
            else:
                daq.append(curr_og_idx)
    return taq,daq, [qvals[i] for _,_,i in all]

# Note: below does not handle ties at all, which becomes very problematic when handling discrete features
# or p-value scores (where many PSMs have zero or unity scores)
def calcQOld(scores, labels,
             thresh = 0.01):
    """Returns q-values and the indices of the positive class such that q <= thresh
    """
    assert len(scores)==len(labels), "Number of input scores does not match number of labels for q-value calculation"
    # all: list of triples consisting of score, label, and index
    all = zip(scores,labels, range(len(scores)))
    #--- sort descending
    all.sort( key = lambda r: -r[0])
    fdrs = []
    posTot = 0.0
    fpTot = 0.0
    fdr = 0.0

    #--- iterates through scores
    for item in all:
        if item[1] == 1: 
            posTot += 1.0
        else: 
            fpTot += 1.0
    
        #--- check for zero positives
        if posTot == 0.0: 
            fdr = 100.0
        else: 
            fdr = fpTot / posTot
        #--- note the q
        fdrs.append(fdr)

    qs = []
    lastQ = 100.0
    for idx in range(len(fdrs)-1, -1, -1):
        q = 0.0
        #--- q can never go up. 
        if lastQ < fdrs[idx]:
            q = lastQ
        else:
            q = fdrs[idx]
        lastQ = q
        qs.append(q)
    qs.reverse()

    taq = []
    daq = []
    for idx, q in enumerate(qs):
        if q > thresh:
            break
        else:
            curr_label = all[idx][1]
            curr_og_idx = all[idx][2]
            if curr_label == 1:
                taq.append(curr_og_idx)
            else:
                daq.append(curr_og_idx)
    return taq,daq, [qs[i] for _,_,i in all]

def load_pin_file(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number
        Todo: add a parser to load PSMs from a DRIP run, returning each PSM as an instance of the 
        dripPSM class

        Normal tide with DRIP features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	insertions	deletions	peaksScoredA	theoPeaksUsedA	SumScoredIntensities	SumScoredMzDist	Peptide	Proteins
    """

    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time
    
    numPeps = 0

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    keys = []
    for k in l:
        if k not in psmKeys and k not in chargeKeys:
            keys.append(k)
            
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

        assert charge > 0, "No charge denoted with value 1 for PSM on line %d, exitting" % (i+1)

        el = {}
        for k in keys:
            el[k] = l[k]

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        try:
            el = PSM(l["Peptide"],
                     float(l[scoreKey]),
                     int(l[sidKey]),
                     kind,
                     charge,
                     el,
                     l["Proteins"],
                     l["SpecId"])
        except KeyError:
            print "Standard PSM field not encountered, exitting"
            exit(-1)

        if kind == 't':
            if (sid,charge) not in targets:
                targets[sid,charge] = []
            targets[sid,charge].append(el)
            numPeps += 1
        elif kind == 'd':
            if (sid,charge) not in decoys:
                decoys[sid,charge] = []
            decoys[sid,charge].append(el)
            numPeps += 1

    return targets,decoys,numPeps

def load_pin_return_dict(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number
        Todo: add a parser to load PSMs from a DRIP run, returning each PSM as an instance of the 
        dripPSM class

        Normal tide with DRIP features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	insertions	deletions	peaksScoredA	theoPeaksUsedA	SumScoredIntensities	SumScoredMzDist	Peptide	Proteins
    """

    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    numPeps = 0

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    keys = list(set(l.iterkeys()) - psmKeys)
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

        assert charge > 0, "No charge denoted with value 1 for PSM on line %d, exitting" % (i+1)

        el = {}
        for k in keys:
            el[k] = l[k]

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        try:
            el = PSM(l["Peptide"],
                     float(l[scoreKey]),
                     int(l[sidKey]),
                     kind,
                     charge,
                     el,
                     l["Proteins"],
                     l["SpecId"])
        except KeyError:
            print "Standard PSM field not encountered, exitting"
            exit(-1)

        sid = el.scan
        if kind == 't':
            if sid in targets:
                if el.score > targets[sid].score:
                    targets[sid] = el
            else:
                targets[sid] = el
                numPeps += 1
        elif kind == 'd':
            if sid in decoys:
                if el.score > decoys[sid].score:
                    decoys[sid] = el
            else:
                decoys[sid] = el
                numPeps += 1
            # targets[el.scan, el.peptide] = el # hash key for PSM class is (scan, peptide string), so
            #                                   # we shouldn't get collisions without adding charge as a key
        #     numPeps += 1
        # elif kind == 'd':
        #     decoys[el.scan, el.peptide] = el # hash key for PSM class is (scan, peptide string), so
        #                                      # we shouldn't get collisions without adding charge as a key
        #     numPeps += 1
    return targets,decoys

def load_ident_return_dict(filename):
    """ Load all PSMs and features ident file
    """
    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "Sid" not in l:
        raise ValueError("No Sid field, exitting")
    # if "Charge" not in l:
    #     raise ValueError("No Charge field, exitting")

    # spectrum identification key for PIN files
    sidKey = "Sid" # note that this typically denotes retention time
    numPeps = 0
    # look at score key and charge keys
    scoreKey = 'Score'
    for i, l in enumerate(reader):
        try:
            sid = int(l[sidKey])
        except ValueError:
            print "Could not convert scan number %s on line %d to int, exitting" % (l[sidKey], i+1)

        if l["Kind"] == 't':
            kind = 't'
        elif l["Kind"] == 'd':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be t or d, exitting" % l["Label"]
            exit(-1)

        charge = int(l["Charge"])

        el = []
        el.append(sid)
        el.append(float(l[scoreKey]))
        el.append(charge)
        el.append(l["Peptide"])

        if kind == 't':
            targets[sid, charge] = el
            numPeps += 1
        elif kind == 'd':
            decoys[sid, charge] = el
            numPeps += 1
        # if kind == 't':
        #     if sid in targets:
        #         if el[1] > targets[sid][1]:
        #             targets[sid] = el
        #     else:
        #         targets[sid] = el
        #         numPeps += 1
        # elif kind == 'd':
        #     if sid in decoys:
        #         if el[1] > decoys[sid][1]:
        #             decoys[sid] = el
        #     else:
        #         decoys[sid] = el
        #         numPeps += 1

    return targets,decoys

def load_pin_return_dictOfArrs(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number

        Normal tide features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	Peptide	Proteins
    """

    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    numPeps = 0

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins"]) # exclude these when reserializing data
    keys = list(set(l.iterkeys()) - constKeys)
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

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        el = []
        # el.append(sid)
        el.append(float(l[scoreKey]))
        el.append(charge)
        el.append(len(l["Peptide"]))
        for k in keys:
            el.append(float(l[k]))

        if kind == 't':
            if sid in targets:
                if el[0] > targets[sid][1]:
                    targets[sid] = el
            else:
                targets[sid] = el
                numPeps += 1
        elif kind == 'd':
            if sid in decoys:
                if el[0] > decoys[sid][1]:
                    decoys[sid] = el
            else:
                decoys[sid] = el
                numPeps += 1
    return targets,decoys

def peptideProphetProcess(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number
    """
    consts = [0.646, -0.959, -1.460, -0.774, -0.598, -0.598, -0.598]
    xcorrs = [5.49, 8.362, 9.933, 1.465, 3.89, 3.89, 3.89]
    deltas = [4.643, 7.386, 11.149, 8.704, 7.271, 7.271, 7.271]
    ranks = [-0.455, -0.194, -0.201, -0.331, -0.377, -0.377, -0.377]
    massdiffs =  [-0.84, -0.314, -0.277, -0.277, -0.84, -0.84, -0.84]
    max_pep_lens = [100, 15, 25, 50, 100, 100, 100]
    num_frags = [2, 2, 3, 4, 6, 6, 6]

    eps = 0.001

    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    numPeps = 0

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        if not i:
            continue
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i and i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins"]) # exclude these when reserializing data
    keys = list(set(l.iterkeys()) - constKeys)
    min_xcorr = min([float(l[scoreKey]) for l in reader]) - eps

    for i, l in enumerate(reader):
        psmid = l["SpecId"]
        psmSplit = psmid.split('_')
        counter = int(psmSplit[1])
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

        assert charge > 0, "No charge denoted with value 1 for PSM on line %d, exitting" % (i+1)

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        ind = charge - 1
        xcorr = float(l[scoreKey]) - min_xcorr
        lp = len(l["Peptide"])
        nl = num_frags[ind] * lp
        lc = max_pep_lens[ind]
        nc = num_frags[ind] * lc

        if lp < lc:
            try:
                xcorr = math.log(xcorr) / math.log(nl)
            except ValueError:
                print "xcorr=%f, nl=%f" % (xcorr, nl)
        else:
            try:
                xcorr = math.log(xcorr) / math.log(nc)
            except ValueError:
                print "xcorr=%f, nc=%f" % (xcorr, nc)

        s = consts[ind] + xcorr * xcorrs[ind] + float(l["deltCn"]) * deltas[ind] + float(l["absdM"]) * massdiffs[ind] + float(l["lnrSp"]) * ranks[ind]
        # s = consts[ind] + xcorr * xcorrs[ind] + float(l["deltCn"]) * deltas[ind] + float(l["absdM"]) * massdiffs[ind]

        el = []
        el.append(sid)
        el.append(s)
        el.append(charge)
        el.append(l["Peptide"])
        for k in keys:
            if not k:
                continue
            el.append(float(l[k]))

        if kind == 't':
            targets[sid, charge, counter] = el
            numPeps += 1
        elif kind == 'd':
            decoys[sid, charge, counter] = el
            numPeps += 1
    print "Evaluated %d PSMs" % numPeps
    return targets,decoys

def load_pin_return_dictOfArrs_peptideProphet(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number
    """
    consts = [0.646, -0.959, -1.460, -0.774, -0.598, -0.598, -0.598]
    xcorrs = [5.49, 8.362, 9.933, 1.465, 3.89, 3.89, 3.89]
    deltas = [4.643, 7.386, 11.149, 8.704, 7.271, 7.271, 7.271]
    ranks = [-0.455, -0.194, -0.201, -0.331, -0.377, -0.377, -0.377]
    massdiffs =  [-0.84, -0.314, -0.277, -0.277, -0.84, -0.84, -0.84]
    max_pep_lens = [100, 15, 25, 50, 100, 100, 100]
    num_frags = [2, 2, 3, 4, 6, 6, 6]

    eps = 0.001

    targets = {}
    decoys = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    numPeps = 0

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        if not i:
            continue
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i and i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    # constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins"]) # exclude these when reserializing data
    constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins", "enzN", "enzC", "enzInt", "deltLCn"]) # exclude these when reserializing data
    keys = list(set(l.iterkeys()) - constKeys)
    min_xcorr = min([float(l[scoreKey]) for l in reader]) - eps

    for i, l in enumerate(reader):
        psmid = l["SpecId"]
        psmSplit = psmid.split('_')
        counter = int(psmSplit[1])
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

        assert charge > 0, "No charge denoted with value 1 for PSM on line %d, exitting" % (i+1)

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        ind = charge - 1
        xcorr = float(l[scoreKey]) - min_xcorr
        lp = len(l["Peptide"])
        nl = num_frags[ind] * lp
        lc = max_pep_lens[ind]
        nc = num_frags[ind] * lc

        if lp < lc:
            try:
                xcorr = math.log(xcorr) / math.log(nl)
            except ValueError:
                print "xcorr=%f, nl=%f" % (xcorr, nl)
        else:
            try:
                xcorr = math.log(xcorr) / math.log(nc)
            except ValueError:
                print "xcorr=%f, nc=%f" % (xcorr, nc)

        el = []
        # el.append(sid)
        el.append(xcorr)
        el.append(charge)
        el.append(len(l["Peptide"]))
        for k in keys:
            if not k:
                continue
            el.append(float(l[k]))

        if kind == 't':
            targets[sid, charge, counter] = el
            numPeps += 1
        elif kind == 'd':
            decoys[sid, charge, counter] = el
            numPeps += 1
    print "Evaluated %d PSMs" % numPeps
    return targets,decoys

def load_pin_return_dictOfArrs_peptideProphet_db(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number
    """
    consts = [0.646, -0.959, -1.460, -0.774, -0.598, -0.598, -0.598]
    xcorrs = [5.49, 8.362, 9.933, 1.465, 3.89, 3.89, 3.89]
    deltas = [4.643, 7.386, 11.149, 8.704, 7.271, 7.271, 7.271]
    ranks = [-0.455, -0.194, -0.201, -0.331, -0.377, -0.377, -0.377]
    massdiffs =  [-0.84, -0.314, -0.277, -0.277, -0.84, -0.84, -0.84]
    max_pep_lens = [100, 15, 25, 50, 100, 100, 100]
    num_frags = [2, 2, 3, 4, 6, 6, 6]

    eps = 0.001

    targets = {}
    decoys = {}
    t_db = {}
    d_db = {}
    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    numPeps = 0

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        if not i:
            continue
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i and i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    # constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins"]) # exclude these when reserializing data
    constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins", "enzN", "enzC", "enzInt", "deltLCn"]) # exclude these when reserializing data
    keys = list(set(l.iterkeys()) - constKeys)
    min_xcorr = min([float(l[scoreKey]) for l in reader]) - eps

    for i, l in enumerate(reader):
        psmid = l["SpecId"]
        psmSplit = psmid.split('_')
        counter = int(psmSplit[1])
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

        assert charge > 0, "No charge denoted with value 1 for PSM on line %d, exitting" % (i+1)

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        ind = charge - 1
        xcorr = float(l[scoreKey]) - min_xcorr
        lp = len(l["Peptide"])
        nl = num_frags[ind] * lp
        lc = max_pep_lens[ind]
        nc = num_frags[ind] * lc

        if lp < lc:
            try:
                xcorr = math.log(xcorr) / math.log(nl)
            except ValueError:
                print "xcorr=%f, nl=%f" % (xcorr, nl)
        else:
            try:
                xcorr = math.log(xcorr) / math.log(nc)
            except ValueError:
                print "xcorr=%f, nc=%f" % (xcorr, nc)

        el = []
        el.append(sid)
        el.append(xcorr)
        el.append(charge)
        el.append(len(l["Peptide"]))
        for k in keys:
            if not k:
                continue
            el.append(float(l[k]))

        if kind == 't':
            targets[sid, charge, counter] = el
            t_db[sid, charge, counter] = l
            numPeps += 1
        elif kind == 'd':
            decoys[sid, charge, counter] = el
            d_db[sid, charge, counter] = l
            numPeps += 1
    print "Evaluated %d PSMs" % numPeps
    return targets,decoys,t_db,d_db

def load_pin_return_featureMatrix(filename):
    """ Load all PSMs and features from a percolator PIN file, or any tab-delimited output of a mass-spec experiment with field "Scan" to denote
        the spectrum identification number

        Normal tide features
        SpecId	Label	ScanNr	lnrSp	deltLCn	deltCn	score	Sp	IonFrac	Mass	PepLen	Charge1	Charge2	Charge3	enzN	enzC	enzInt	lnNumSP	dm	absdM	Peptide	Proteins
    """

    with open(filename, 'r') as f:
        reader = [l for l in csv.DictReader(f, delimiter = '\t', skipinitialspace = True)]
    l = reader[0]
    if "ScanNr" not in l:
        raise ValueError("No ScanNr field, exitting")
    if "Charge1" not in l:
        raise ValueError("No Charge1 field, exitting")

    # spectrum identification key for PIN files
    sidKey = "ScanNr" # note that this typically denotes retention time

    maxCharge = 1
    chargeKeys = set([])
    # look at score key and charge keys
    scoreKey = ''
    for i in l:
        m = i.lower()
        if m == 'score':
            scoreKey = i
        if m[:-1]=='charge':
            chargeKeys.add(i)
            maxCharge = max(maxCharge, int(m[-1]))

    if not scoreKey:
        for i in l:
            if i.lower() == 'xcorr':
                scoreKey = i            

    # fields we have to keep track of
    psmKeys = set(["SpecId", "Label", sidKey, scoreKey, "Peptide", "Proteins"])
    constKeys = set(["SpecId", "Label", sidKey, scoreKey, "charge", "Peptide", "Proteins"]) # exclude these when reserializing data
    keys = list(set(l.iterkeys()) - constKeys)

    targets = {}  # mapping between sids and indices in the feature matrix
    decoys = {}

    X = [] # Feature matrix
    Y = [] # labels
    pepstrings = []
    scoreIndex = _scoreInd # column index of the ranking score used by the search algorithm 
    numRows = 0
    # feature descriptions
    featureNames = [scoreKey, "Charge", "pepLen"]
    for k in keys:
        featureNames.append(k)
    
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

        if l["Label"] == '1':
            kind = 't'
        elif l["Label"] == '-1':
            kind = 'd'
        else:
            print "Error: encountered label value %s, can only be -1 or 1, exitting" % l["Label"]
            exit(-1)

        el = []
        # el.append(sid)
        el.append(float(l[scoreKey]))
        el.append(charge)
        el.append(len(l["Peptide"])-4)
        for k in keys:
            el.append(float(l[k]))

        # note: in the below, we can perform target-decoy competition
        
        # Note: could be problematic taking the max wrt sid below
        if kind == 't':
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
                numRows += 1
        elif kind == 'd':
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
                numRows += 1
    # Standard-normalize the feature matrix
    return targets,decoys, pepstrings, preprocessing.scale(np.array(X)), np.array(Y), featureNames

def findInitDirection(X, Y, thresh, featureNames):
    l = X.shape
    m = l[1]
    initDirection = -1
    numIdentified = 0
    for i in range(m):
        taq, _, _ = calcQ(X[:,i], Y, thresh, True)
        if len(taq) > numIdentified:
            initDirection = i
            numIdentified = len(taq)

        if _debug and _verb >= 2:
            print "Direction %d, %s: Could separate %d identifications" % (i, featureNames[i], len(taq))
    return initDirection, numIdentified

def getDecoyIdx(labels, ids):
    return [i for i in ids if labels[i] != 1]

def searchForInitialDirection(keys, X, Y, q, featureNames):
    """ Iterate through cross validation training sets and find initial search directions
        Returns the scores for the disjoint bins
    """
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
        initDir, numIdentified = findInitDirection(X[trainSids], Y[trainSids], q, featureNames)
        print "CV fold %d: could separate %d PSMs in initial direction %d, %s" % (kFold, numIdentified, initDir, featureNames[initDir])
        scores[trainSids] = X[trainSids,initDir]
    return scores

def doLda(thresh, keys, scores, X, Y, 
          t_scores, d_scores, 
          target_rowsToSids, decoy_rowsToSids):
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

        if _mergescore:
            u, d = qMedianDecoyScore(iter_scores, Y[testSids], thresh = 0.01)
            iter_scores = (iter_scores - u) / (u-d)

        for i, score in zip(testSids, iter_scores):
            newScores[i] = score
            if Y[i] == 1:
                sid = target_rowsToSids[i]
                t_scores[sid] = score
            else:
                sid = decoy_rowsToSids[i]
                d_scores[sid] = score

    scores = newScores
    return totalTaq

def writeOutput(output, Y, pepstrings,
                target_rowsToSids, t_scores,
                decoy_rowsToSids, d_scores):
    n = len(Y)
    fid = open(output, 'w')
    fid.write("Kind\tSid\tPeptide\tScore\n")
    counter = 0
    for i in range(n):
        if Y[i] == 1:
            sid = target_rowsToSids[i]
            score = t_scores[sid]
            p = pepstrings[i]
            fid.write("t\t%d\t%s\t%f\n"
                      % (sid,p,score))
            counter += 1
        else:
            sid = decoy_rowsToSids[i]
            score = d_scores[sid]
            p = pepstrings[i]
            fid.write("d\t%d\t%s\t%f\n"
                      % (sid,p,score))
            counter += 1
    fid.close()
    print "Wrote %d PSMs" % counter


def discFunc(options, output):
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

    shuffle(keys)
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
        scores = searchForInitialDirection(keys, X, Y, q, featureNames)

    numIdentified = doLda(q, keys, scores, X, Y, 
                          t_scores, d_scores, 
                          target_rowsToSids, decoy_rowsToSids)
    print "LDA finished, identified %d targets at q=%.2f" % (numIdentified, q)

    writeOutput(output, Y, pepstrings,
                target_rowsToSids, t_scores,
                decoy_rowsToSids, d_scores)

def discFuncIter(options, output):
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

    shuffle(keys)
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
        scores = searchForInitialDirection(keys, X, Y, q, featureNames)

    for i in range(10):
        numIdentified = doLda(q, keys, scores, X, Y, 
                              t_scores, d_scores, 
                              target_rowsToSids, decoy_rowsToSids)
        print "iter %d: %d targets" % (i, numIdentified)

    writeOutput(output, Y, pepstrings,target_rowsToSids, t_scores,
                decoy_rowsToSids, d_scores)

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--q', type = 'float', action= 'store', default = 0.5)
    parser.add_option('--tol', type = 'float', action= 'store', default = 0.01)
    parser.add_option('--initDirection', type = 'int', action= 'store', default = -1)
    parser.add_option('--verb', type = 'int', action= 'store', default = -1)
    parser.add_option('--maxIters', type = 'int', action= 'store', default = 10)
    parser.add_option('--pin', type = 'string', action= 'store')
    parser.add_option('--filebase', type = 'string', action= 'store')

    (options, args) = parser.parse_args()

    _verb=options.verb
    discOutput = '%s_ppProcess.txt' % (options.filebase)
    # discFunc(options, discOutput)
    discFuncIter(options, discOutput)
    # TODO: add q-value post-processing: mix-max and TDC
