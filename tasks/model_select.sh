#!/bin/bash

# activate conda
source ~/anaconda3/etc/profile.d/conda.sh

# switch to selection environment
conda activate downstream
echo "current environment: $CONDA_DEFAULT_ENV"

# use at most 2 cores
max_cores=2

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

dataset_list=(MoCap ActRecTut PAMAP2 USC-HAD SynSeg)
method_list=(time2state e2usd)

# run tasks in parallel
run_tasks() {
  for dataset in "${dataset_list[@]}"; do
    for method in "${method_list[@]}"; do
      python experiments/model_select.py $method $dataset &
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