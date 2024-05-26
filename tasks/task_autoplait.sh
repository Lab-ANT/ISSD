# list of methods to run
method_list=(issd pca umap ecs ecp lda sfm)
dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)

python downstream_methods/AutoPlait/experiments/convert_to_AutoPlait_format.py
cd downstream_methods/AutoPlait
for dataset in ${dataset_list[@]}; do
    for method in ${method_list[@]}; do
        # echo $method
        bash experiments/run_AutoPlait.sh $dataset $method
    done
done
cd ../..
python downstream_methods/AutoPlait/experiments/rediredt_results.py