#ProteoTorch

A Python package for semi-supervised deep learning-based analysis of MS/MS database search results of shotgun proteomics data, in addition to fast SVM implementations.

ProteoTorch accepts as input a Percolator INput (PIN) file containing target/decoy PSM features. By default, several iterations of deep semi-supervised learning are then performed to classify target and decoy PSMs, and the output PSM scores are recalibrated using the resulting learned parameters.

The full documentation can be found at:

https://proteotorch.readthedocs.io/en/latest/
