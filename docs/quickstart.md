# Quickstart

Here, we briefly describe post-processing a PIN file, **test.pin**, using a deep neural network (DNN) as the classifier during semi-supervised learning.  The main ProteoTorch module is **analyze**, which may be run from the command line using:

    python3 -m proteoTorch.analyze --pin test.pin --output_dir testOutput --method 3 --numThreads 10

In the above, **method** specifies a DNN classifier, **output_dir** specifies the directory to write results to, and **numThreads** specifies the number of CPU threads to use for ProteoTorch computation which is parallelizable (discussed further in the [next section](analyze.md)).

When finished, the recalibrated PSM scores will be written to the tab-delimited file **output_dir/output.txt**.  The first few lines of an example output file are:

    PSMId   score   q-value peptide Label   proteinIds
    target_0_6395_3_1       0.999988        0.385129        K.IDSAAETHADAPVVDASPAEDQASEVTEAPHVESAK.S        1       F52H3.7a
    target_0_8821_3_1       0.999975        0.385129        R.LHCTAQPMPDGLADDIEGGTVNAR.D    1       F25H5.4
    target_0_10580_2_1      0.999968        0.385129        K.LANALEPGAVEVAAAEENADAAAQK.E   1       C10G11.7
    target_0_16058_2_1      0.999959        0.385129        K.TVDVISDTGTSFLGGPQSVVDGLAK.A   1       F21F8.7
    target_0_17089_2_1      0.999956        0.385129        K.HTDAVAELTDQLDQLNK.A   1       F11C3.3
    target_0_16837_2_1      0.999955        0.385129        K.HTDAVAELTDQLDQLNK.A   1       F11C3.3
    target_0_27067_3_1      0.999952        0.385129        R.FQSSAVMALQEAAEAYLVGLFEDTNLCAIHAK.R    1       ZK131.3
    target_0_15165_2_1      0.999945        0.385129        R.NLQIAQGTPGGLITYGAIDTVNCAK.Q   1       Y39B6A.20
    target_0_17414_2_1      0.999943        0.385129        K.VTLEDQIVQTNPVLEAFGNAK.T       1       F11C3.3

## Q-value plots
The accuracy of the recalibrated scores, as a function of PSM q-values, may be quickly plotted and compared against other methods.  Assuming **test.pin** had a feature _XCorr_ specified in its header, the following may be run on the command line to plot the results of a target-decoy competition for both ProteoTorch's DNN classifier and the uncalibrated XCorr scores:

    python3 -m proteoTorch.plotQvals --output test.pdf --maxq 0.1  \
        --tdc --dataset test.pin \
        "ProteoTorch DNN":"score":output_dir/output.txt \
        "XCorr":'XCorr':test.pin

In the above, **output** specified the resulting plot file name (and format), **maxq** specifies the maximum q-value threshold to plot, **tdc** turns on target-decoy competition for all methods (in which case, the original pin file must be specified by **dataset** to properly calculate the **experimental mass**, **scan number** identifiers for each PSM), and tab-delimited PSM files are specified by the triple **method:header field:PSM file**.

Further plotting options are discussed in an [upcoming section](plotting.md).
