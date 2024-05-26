bash tasks/del_selection_results.sh
bash tasks/run_selection.sh
bash nohup tasks/task_time2state.sh > output/tasks_time2state.out &
bash nohup tasks/task_e2usd.sh > output/tasks_e2usd.out &
bash nohup tasks/task_ticc.sh > output/tasks_ticc.out &
bash nohup tasks/task_AutoPlait.sh > output/tasks_AutoPlait.out &