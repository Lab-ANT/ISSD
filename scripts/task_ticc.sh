# list of methods to run
# method_list=(issd sfs pca umap ecs ecp lda sfm)
# dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)
method_list=(issd pca umap ecs ecp lda sfm)
dataset_list=(PAMAP2)

# time2state
for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        # echo $method
        python -u experiments/run_dmethods.py ticc $dataset $method
    done
done
