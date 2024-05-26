# sequentially run all tasks, waiting for each to finish before starting the next one
bash tasks/task_autoplait.sh > output/tasks_autoplait.out
bash tasks/task_time2state.sh > output/tasks_time2state.out
bash tasks/task_e2usd.sh > output/tasks_e2usd.out
bash tasks/task_ticc.sh > output/tasks_ticc.out