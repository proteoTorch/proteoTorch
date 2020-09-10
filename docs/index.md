<!--- ProteoTorch documentation master file, created by
   sphinx-quickstart on Fri Sep  4 12:51:58 2020.-->

# ProteoTorch
_A Python package for deep learning (and fast general machine learning) analysis of MS/MS database search results._

## Key Features
ProteoTorch accepts as input a Percolator INput (PIN) file containing target/decoy PSM features.
By default, several iterations of deep semi-supervised learning are then performed to classify
target and decoy PSMs, and the output PSM scores are recalibrated using the resulting learned parameters.

ProteoTorch provides the following semi-supervised machine learning classifiers:
* Deep neural networks
* Fast Linear SVMs using the L2-SVM-MFN algorithm (this is equivalent to [the recently sped-up Percolator algorithm](https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.9b00288))
* Linear SVMs using the TRON algorithm
* Linear Discriminant Analysis (+ Gaussian Mixture Models, in development)
* Support to easily swap in any supervised classifier implemented in Python which follow the design
 of [scikit-learn clf object instances](https://scikit-learn.org/stable/tutorial/basic/tutorial.html), with  
 training function *fit* and testing function *decision_function*

Additionally, ProteoTorch provides an ultrafast q-value library (heavily optimized for Python),
plotting tools to benchmark/compare MS/MS post-processor results, and an easy-to-use Python API for
the MS/MS semi-supervised learning algorithm (with cross-validation) originally implemented in the Percolator C++ package.

## Contents
* [Install](install.md)
* [Quickstart](quickstart.md)
* [Recalibrating MS/MS search results](analyze.md)
* [Plotting utilities](analyze.md)
* [API reference](api.md)
* [Contact](contact.md)