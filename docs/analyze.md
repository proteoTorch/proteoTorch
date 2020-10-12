# Analysis options
The analysis source is provided in the module **proteoTorch.analyze**.  Available options are listed below.

## Main Options
The following is a list of post-processing options when calling **proteoTorch** from the command line.
* *\-\-pin*: input file in PIN format
* *\-\-method*: machine learning classifier to use during semi-supervised learning
  * Method 0: LDA
  * Method 1: linear SVM, solver TRON
  * Method 2: linear SVM, solver L2-SVM-MFN (Percolator's solver)
  * Method 3: DNN (deep multi-layer perceptron, *default value*)
* *\-\-output_dir*: where to write result files.  **Default = model_output/<data_file_name>/<time_stamp>/**
* *\-\-numThreads*: Number of CPU threads to use for parallelizable computations. **Default = 1**)
* *\-\-initDirection*: If >= 0, specifies which feature to use as initial PSM scores during semi-supervised learning.  If = -1, automatically find and use the most discriminative feature. **Default = -1**
* *\-\-q*: q-value tolerance when estimating positive training samples. **Default = 0.01**
* *\-\-verbose*: Verbosity. **Default = 1**
* *\-\-output_per_iter_granularity*: Specifies number of iterations to write recalibrated PSM scores. **Default = 5**
* *\-\-write_output_per_iter*: Write recalibrated PSM scores after every *output_per_iter_granularity* iterations (boolean). **Default = true**
* *\-\-maxIters*: Number of semi-supervised learning iterations to run. **Default = 20**
* *\-\-seed*: Random seed when partitioning PSMs into cross-validation bins. **Default = 1**

## Deep learning options
* *\-\-dnn_optimizer*: DNN training algorithm to use (sgd or Adam). **Default = Adam**
* *\-\-dnn_num_epochs*: Number of epochs to train DNN. **Default = 50**
* *\-\-deepq*: DNN q-value tolerance when estimating positive training samples. **Default = 0.07**
* *\-\-dnn_lr*: DNN learning rate. **Default = 0.001**
* *\-\-dnn_lr_decay*: Reduce learning rate by this total for all epochs (dnn_lr_decay / dnn_num_epochs applied after each epoch). **Default = 0.02**
* *\-\-dnn_num_layers*: Number of hidden DNN layers. **Default = 3**
* *\-\-dnn_layer_size*: Number of neurons per hidden layer. **Default = 200**
* *\-\-starting_dropout_rate*: Dropout rate for first iteration. **Default = 0.5**
* *\-\-dnn_dropout_rate*: Dropout rate for iterations > 1. **Default = 0.0**
* *\-\-dnn_gpu_id*: GPU ID to use for the DNN model (will switch to CPU mode if no GPU is found or CUDA is not installed). **Default = 0**
* *\-\-dnn_label_smoothing_0*: Label smoothing for training class 0 (decoys). **Default = 0.99**
* *\-\-dnn_label_smoothing_1*: Label smoothing for training class 1 (targets within q-value tolerance). **Default = 0.99**
* *\-\-dnn_train_qtol*: AUC q-value tolerance to measure validation performance. **Default = 0.1**
* *\-\-false_positive_loss_factor*: Multiplicative factor to weight false positives during training. **Default = 4.0**
* *\-\-deepInitDirection*: Produce initial PSM scores by training a large ensemble DNN to speed up training convergence (boolean). **Default = true if *method*=3**
* *\-\-deep_direction_ensemble*: Number of DNN ensembles to train during deep initial direction search. **Default = 30**
* *\-\-load_previous_dnn*: Start iterations from previously saved model (boolean). **Default = false**
* *\-\-previous_dnn_dir*: Previous output directory containing trained dnn weights.


## Note on parallelization
Within each iteration of the algorithm, nested cross-validation (CV) is performed.  If a DNN classifier is selected (i.e., _\-\-method 3_), the CV folds are run sequentially.  This safeguards against the GPU running out of memory and ProteoTorch crashing during analysis.

When an SVM is selected (i.e., _\-\-method 2_ or _\-\-method 3_), the CV folds are run in parallel using the number of CPU threads specified by _\-\-numThreads_.