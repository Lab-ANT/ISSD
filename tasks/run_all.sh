# sequentially run all tasks, waiting for each to finish before starting the next one
bash tasks/task_autoplait.sh > output/task_autoplait.out
bash tasks/task_time2state.sh > output/task_time2state.out
bash tasks/task_e2usd.sh > output/task_e2usd.out
bash tasks/task_ticc.sh > output/task_ticc.out