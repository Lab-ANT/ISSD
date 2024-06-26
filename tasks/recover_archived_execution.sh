#!/bin/bash

execution_num=$1

dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)
method_list=(issd issd-qf issd-cf pca umap ecs ecp lda sfm)

for dataset in ${dataset_list[@]}; do
  for method in ${method_list[@]}; do
    cp -r archive/execution$execution_num/data/$dataset/$method data/$dataset
  done
done

cp -r archive/execution$execution_num output