# Quickstart

Here, we briefly describe post-processing a PIN file, **test.pin**, using a deep neural network (DNN) as the classifier during semi-supervised learning.  

## Recalibrating PSMs
After installation, the main ProteoTorch module, **analyze**, may be run from the command line using:

    proteoTorch --pin test.pin --output_dir testOutput --method 3 --numThreads 10

In the above, *\-\-method* specifies a DNN classifier, *\-\-output_dir* specifies the directory to write results to, and *\-\-numThreads* specifies the number of CPU threads to use for all parallelizable ProteoTorch computation (discussed further in the [next section](analyze.md)).

When finished, the recalibrated PSM scores will be written to the tab-delimited file *testOutput/output.txt*.  The first few lines of an example output file are:

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

Further analysis options are discussed in the [next section](analyze.md).

## Q-value analysis plots
Resulting PSM identifications vs q-values may be easily plotted and compared against other methods.  Assuming **test.pin** has feature **XCorr** specified in its header, the following may be run on the command line to plot target-decoy competition (TDC) results for both recalibrated ProteoTorch scores and the uncalibrated XCorr scores:

    proteoTorchPlot --output test.pdf --maxq 0.1  \
        --tdc --dataset test.pin \
        "ProteoTorch DNN":"score":output_dir/output.txt \
        "XCorr":'XCorr':test.pin

In the above, *\-\-output* specifies the resulting plot file name (and format), *\-\-maxq* specifies the maximum q-value threshold to plot, *\-\-tdc* runs TDC for all methods, and tab-delimited PSM files are specified by the triple *method:header field:PSM file*.  Note that, when TDC is specified, the original pin file must be specified using *\-\-dataset* to properly calculate the *experimental mass*, *scan number* identifiers for each PSM.  

Further details and examples are discussed on the [plotting page](plotting.md).

<!--- Below is an example plot for a draft of the human proteome dataset run initially searched with the high-res MS2 p-value score function, *residue-evidence combined p-value*, from the Crux toolkit.

![](kim_resev.png)

Further plotting options are discussed on the [plotting page](plotting.md).-->
