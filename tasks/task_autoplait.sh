#!/bin/bash

# use at most 50% of the CPU cores to run the tasks in for loop
# cpu_core_num=$(cat /proc/cpuinfo | grep "processor" | wc -l)
cpu_cores=$(nproc)
echo "Total CPU cores: $cpu_cores"
max_cores=$((cpu_cores / 2))
echo "Max cores to use: $max_cores"

# method_list=(raw issd issd-qf issd-cf pca umap ecs ecp lda sfm)
method_list=(issd issd-qf issd-cf)
dataset_list=(PAMAP2 USC-HAD MoCap ActRecTut SynSeg)

# check if the argument is specified,
# if specified, run the specified method only
if [ -n "$1" ]; then
  method_list=($1)
fi
echo "Method list: ${method_list[@]}"

# convert the dataset to AutoPlait format
python downstream_methods/AutoPlait/experiments/convert_to_AutoPlait_format.py
cd downstream_methods/AutoPlait

# save pids for easy stopping
pids=()

# cleanup on exit
trap 'cleanup' INT TERM

# cleanup function
# TODO: this function is not working as expected
# there are always # max_cores processes left running
cleanup() {
  echo "Cleaning up..."
  for pid in "${pids[@]}"; do
    echo "Killing process $pid"
    kill "$pid"
  done
  exit 0
}

# run tasks in parallel
run_tasks() {
  for dataset in "${dataset_list[@]}"; do
    for method in "${method_list[@]}"; do
      bash experiments/run_AutoPlait.sh $dataset $method &
      pid=$!
      pids+=($pid)
      echo "Started task with PID $pid for dataset $dataset and method $method"

      # wait until there are less than max_cores processes running
      while [[ ${#pids[@]} -ge $max_cores ]]; do
        wait -n
        # remove finished pids
        for i in "${!pids[@]}"; do
          if ! kill -0 "${pids[$i]}" 2>/dev/null; then
            unset 'pids[i]'
          fi
        done
      done
    done
  done
}

run_tasks
wait

cd ../..

echo "All tasks completed"
