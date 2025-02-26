#!/bin/bash

# run all selection methods
# method_list=(issd ecs ecp lda sfm pca umap)
method_list=(issd)
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        python experiments/selection.py $dataset $method 4
    done
done
