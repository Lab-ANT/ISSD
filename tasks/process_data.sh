#!/bin/bash

dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

cd data/raw
for dataset in ${dataset_list[@]}; do
  unzip -o $dataset.zip
done
cd ../..

for dataset in ${dataset_list[@]}; do
  mkdir -p data/$dataset
done

cp -r data/raw/MoCap/ data/MoCap
mv data/MoCap/MoCap data/MoCap/raw # rename
cp -r data/raw/SynSeg/ data/SynSeg
mv data/SynSeg/SynSeg/ data/SynSeg/raw # rename

python datautils/convert_data_format.py