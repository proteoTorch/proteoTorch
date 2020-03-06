#!/usr/bin/env python
#
# Written by John Halloran <jthalloran@ucdavis.edu>
#
# Copyright (C) 2020 John Halloran
# Licensed under the Open Software License version 3.0
# See COPYING or http://opensource.org/licenses/OSL-3.0
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from deepMs import getQValues

scores=[0.78086,0.10307,0.32862,0.30650,0.73567,0.55191,0.33581,0.84533,0.93682,0.72977]
labels=[1,0,1,0,1,0,1,0,1,0]
pi0=1.
combined = zip(scores,labels)

skipDecoysPlusOne=True
qvals=getQValues(pi0, combined, skipDecoysPlusOne)
print qvals

skipDecoysPlusOne=False
qvals=getQValues(pi0, combined, skipDecoysPlusOne)
print qvals

# Flip the labels and test
labels=[0,1,0,1,0,1,0,1,0,1]
combined = zip(scores,labels)

skipDecoysPlusOne=True
qvals=getQValues(pi0, combined, skipDecoysPlusOne)
print qvals

skipDecoysPlusOne=False
qvals=getQValues(pi0, combined, skipDecoysPlusOne)
print qvals
