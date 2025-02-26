#!/bin/sh
# Created by Chengyu on 2024/2/24.
# arguments: $1: dataset
# must be run in the downstream_methods/AutoPlait/ directory
# e.g., bash experiments/test.sh CaseStudy

# check if autoplait is compiled
# if src/autoplait does not exist, compile it.
if [ ! -f "./src/autoplait" ]; then
  echo "Compiling AutoPlait..."
  cd ./src
  make cleanall
  make
  cd ..
fi

# compile.
# cd ./src
# make cleanall
# make
# cd ..

# Configuration
data_source="data/"
OUTDIR="output/"

INPUTDIR=$data_source"$1/"
outdir=$OUTDIR"$1/"
dblist=$INPUTDIR"list"
info=$INPUTDIR"info"
# statistic the number of lines in dblist
n=$(wc -l < $dblist) # data size
# load dimlsit from info,
# each row is the number of dimensions of the corresponding dataset
dimlist=$(awk '{print $1}' $info)
# convert to array
dimlist=($dimlist)

echo $INPUTDIR
mkdir -p $outdir

for (( i=1; i<=$n; i++ ))
do
  output=$outdir"dat"$i"/"
  mkdir -p $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}'# $dblist > $input
  ./src/autoplait ${dimlist[$i-1]} $input $output
done