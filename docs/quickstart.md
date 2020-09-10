# Quickstart

Here, we briefly describe post-processing a PIN file, _test.pin_, using a deep neural network (DNN) as the classifier during semi-supervised learning.  The main ProteoTorch module is **analyze**, which may be run from the command using

    python3 -m proteoTorch.analyze --pin test.pin --method 3 --output_dir testOutput --numThreads 10

In the above, _method_ specifies a DNN, *output_dir* specifies the directory to write results to, and _numThreads_ specifies the number of CPU threads to use for ProteoTorch computation which is parallelizable.  The default DNN settings have been exhaustively tested and should be sufficient for most datasets.  When finished, the recalibrated PSM scores will be written to *output_dir/output.txt*.


It is important to note that within each iteration of the algorithm, nested cross-validation (CV) is performed and when a DNN is selected (i.e., _--method 3_), the CV folds are run sequentially.  This safeguards against the GPU running out of memory and ProteoTorch crashing during post-processing.  In contrast, when an SVM is selected (i.e., _--method 2_ for the Percolator compliant L2-SVM-MFN solver or _--method 3_ for LIBLINEAR's TRON solver), the CV folds are run in parallel using the number of threads specified by _numThreads_.