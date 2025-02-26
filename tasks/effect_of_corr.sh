#!bin/bash

mkdir -p archive/corr_effect
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

# activate conda
source ~/anaconda3/etc/profile.d/conda.sh

for corr in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "current correlation: $corr"
    # switch to selection environment
    conda activate selection
    echo "current environment: $CONDA_DEFAULT_ENV"

    # run selection on all datasets
    for dataset in ${dataset_list[@]}; do
        # run selection
        echo "current dataset: $dataset, current correlation: $corr"
        python experiments/selection.py $dataset issd 4 $corr
    done

    # switch to downstream environment
    conda activate downstream
    echo "current environment: $CONDA_DEFAULT_ENV"

    # run time2state
    for dataset in ${dataset_list[@]}; do
        for i in 1 2 3 4 5; do
            mkdir -p archive/corr_effect/$dataset/$corr/time2state$i
            python experiments/run_dmethods.py time2state $dataset issd
            echo "use time2state"
            cp -r output/results/time2state/$dataset/issd/ archive/corr_effect/$dataset/$corr/time2state$i
        done
    done

    # run e2usd
    for dataset in ${dataset_list[@]}; do
        for i in 1 2 3 4 5; do
            mkdir -p archive/corr_effect/$dataset/$corr/e2usd$i
            python experiments/run_dmethods.py e2usd $dataset issd
            echo "use e2usd"
            cp -r output/results/e2usd/$dataset/issd/ archive/corr_effect/$dataset/$corr/e2usd$i
        done
    done

    # run ticc
    for dataset in ${dataset_list[@]}; do
        mkdir -p archive/corr_effect/$dataset/$corr/ticc
        python experiments/run_dmethods.py ticc $dataset issd
        cp -r output/results/ticc/$dataset/issd/ archive/corr_effect/$dataset/$corr/ticc
    done

    # run autoplait
    for dataset in ${dataset_list[@]}; do
        mkdir -p archive/corr_effect/$dataset/$corr/autoplait
        bash downstream_methods/AutoPlait/experiments/run_AutoPlait_parallel.sh $dataset issd
        cp -r output/results/autoplait/$dataset/issd/ archive/corr_effect/$dataset/$corr/autoplait
    done
done