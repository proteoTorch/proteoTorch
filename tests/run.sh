#!/bin/bash

function testingWorm {
    PIN="worm01.pin"
    OUTPUT="worm_svmlin.txt"
    q0="0.01"
    dr=0.0
    python3 setup.py build_ext --inplace
    outputBase=worm_svmLin
    for mi in 10
    do
	output=${outputBase}_mi${mi}
	python3 deepMs.py --pin $PIN --q $q0 \
    	    --initDirection -1 \
	    --maxIters ${mi} \
    	    --verb 0 \
    	    --method 2 \
	    --dnn_dropout_rate ${dr} \
    	    --output_dir $output
    done
}

testingWorm
