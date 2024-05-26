# list of methods to run
method_list=(issd pca umap ecs ecp lda sfm)
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

cd downstream_methods/AutoPlait
python experiments/convert_to_AutoPlait_format.py
for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        # echo $method
        bash experiments/run_AutoPlait.sh $dataset $method
    done
done
python experiments/rediredt_results.py