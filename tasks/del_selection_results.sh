#!/bin/bash

# delete selection and reduction results
method_list=(issd issd-qf issd-cf mi pca umap ecs ecp lda sfm pca umap human sfs)
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        echo data/$dataset/$method deleted
        rm -rf data/$dataset/$method
    done
done

rm -rf downstream_methods/AutoPlait/data
rm -rf downstream_methods/AutoPlait/output