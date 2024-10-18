#!bin/bash

mkdir -p archive/corr_effect
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

# activate conda
source ~/anaconda3/etc/profile.d/conda.sh
# switch to selection environment
conda activate selection
echo "current environment: $CONDA_DEFAULT_ENV"

for dataset in ${dataset_list[@]}; do
    for corr in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        echo "current correlation: $corr"
        # run selection on all datasets
        # run selection
        echo "current dataset: $dataset, current correlation: $corr"
        python experiments/selection.py $dataset issd 4 $corr
    done
done