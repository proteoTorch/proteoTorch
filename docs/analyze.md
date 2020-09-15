# Post-processing MS/MS search results using **proteoTorch.analyze**

The default DNN settings have been exhaustively tested and should be sufficient for most datasets.

## Parallelization
Within each iteration of the algorithm, nested cross-validation (CV) is performed.  When a DNN classifier is selected (i.e., _--method 3_), the CV folds are run sequentially, safeguarding against the GPU running out of memory and ProteoTorch crashing during post-processing.  In contrast, when an SVM is selected (i.e., _--method 2_ for the Percolator compliant L2-SVM-MFN solver or _--method 3_ for LIBLINEAR's TRON solver), the CV folds are run in parallel using the number of threads specified by _numThreads_.