#!/bin/bash

execution_num=$1
specified_method=$2

# activate conda
source ~/anaconda3/etc/profile.d/conda.sh

# switch to selection environment
conda activate selection
echo "current environment: $CONDA_DEFAULT_ENV"

# run selection
mkdir -p output
bash tasks/run_selection.sh > output/log_selection.out

# switch to downstream environment
conda activate downstream
echo "current environment: $CONDA_DEFAULT_ENV"

# sequentially run all tasks
bash tasks/task_autoplait.sh $specified_method > output/log_autoplait.out
bash tasks/task_time2state.sh $specified_method > output/log_time2state.out
bash tasks/task_e2usd.sh $specified_method > output/log_e2usd.out
bash tasks/task_ticc.sh $specified_method > output/log_ticc.out

# redirect results for AutoPlait
python downstream_methods/AutoPlait/experiments/redirect_results.py

# summary results and plot figures
bash tasks/draw_figures.sh

bash tasks/archive_execution.sh $execution_num
bash tasks/del_selection_results.sh