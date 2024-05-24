# list of methods and datasets to run
method_list=(issd pca umap ecs ecp lda sfm pca umap)
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        echo $method $dataset
    done
done
