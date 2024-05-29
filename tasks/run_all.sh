#!/bin/bash

execution_num=$1

# activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate selection
echo "current environment: $CONDA_DEFAULT_ENV"

mkdir -p output
bash tasks/run_selection.sh > output/log_selection.out

conda activate statecorr
echo "current environment: $CONDA_DEFAULT_ENV"

# sequentially run all tasks
bash tasks/task_autoplait.sh > output/log_autoplait.out
bash tasks/task_time2state.sh > output/log_time2state.out
bash tasks/task_e2usd.sh > output/log_e2usd.out
bash tasks/task_ticc.sh > output/log_ticc.out

python downstream_methods/AutoPlait/experiments/redirect_results.py

# summary results and plot figures
python experiments/summary.py
python paper_figures/plot_overall_performance.py
python paper_figures/plot_individual_performance.py
python paper_figures/plot_cd.py

bash tasks/archive_execution.sh $execution_num
bash tasks/del_selection_results.sh