#!/bin/sh
# arguments: $1: dataset, $2: method
# must be run in the downstream_methods/AutoPlait/ directory
# e.g., bash experiments/test.sh MoCap issd

python downstream_methods/AutoPlait/experiments/convert_one.py $1 $2
cd downstream_methods/AutoPlait

# check if autoplait is compiled
if [ ! -f "./src/autoplait" ]; then
  echo "Compiling AutoPlait..."
  cd ./src
  make cleanall
  make
  cd ..
fi

# Configuration
data_source="data/"
OUTDIR="output/"

INPUTDIR=$data_source"$1/$2/"
outdir=$OUTDIR"$1/$2/"
dblist=$INPUTDIR"list"
info=$INPUTDIR"info"

# Count total lines (data size)
n=$(wc -l < $dblist)

# Load dimensions from info file
dimlist=$(awk '{print $1}' $info)
dimlist=($dimlist)

# Prepare output directory
echo $INPUTDIR
mkdir -p $outdir

# Set max parallel processes
cpu_cores=$(nproc)
echo "Total CPU cores: $cpu_cores"
max_cores=$((cpu_cores / 2))
echo "Max cores to use: $max_cores"

# Parallel processing with process limit
current_jobs=0

for (( i=1; i<=$n; i++ ))
do
  output=$outdir"dat"$i"/"
  mkdir -p $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}' $dblist > $input
  ./src/autoplait ${dimlist[$i-1]} $input $output &

  # Increment job count
  current_jobs=$((current_jobs + 1))

  # If reaching max cores, wait for background jobs to complete
  if [ "$current_jobs" -ge "$max_cores" ]; then
    wait
    current_jobs=0
  fi
done

# Wait for any remaining background jobs to complete
wait

echo "All processes complete."

cd ../..
python downstream_methods/AutoPlait/experiments/redirect_one.py $1 $2