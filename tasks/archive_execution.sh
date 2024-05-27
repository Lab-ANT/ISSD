#!/bin/bash

execution_num=$1

mkdir -p output/data
mkdir -p archive

dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

for dataset in ${dataset_list[@]}; do
    cp -r data/$dataset output/data
done

mv output archive/execution$execution_num