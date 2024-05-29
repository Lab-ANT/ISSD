#!/bin/bash

# list of methods to run
method_list=(issd pca umap ecs ecp lda sfm mi)
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        # echo $method
        python -u experiments/run_dmethods.py ticc $dataset $method
    done
done
