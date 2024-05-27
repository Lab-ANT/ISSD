execution_num=$1

# activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selection
echo "current environment: $CONDA_DEFAULT_ENV"

bash tasks/run_selection.sh

conda activate statecorr
echo "current environment: $CONDA_DEFAULT_ENV"

# sequentially run all tasks
bash tasks/task_autoplait.sh > output/task_autoplait.out
bash tasks/task_time2state.sh > output/task_time2state.out
bash tasks/task_e2usd.sh > output/task_e2usd.out
bash tasks/task_ticc.sh > output/task_ticc.out

python downstream_methods/AutoPlait/experiments/redirect_results.py

bash tasks/archive_execution.sh $execution_num
bash tasks/del_selection_results.sh