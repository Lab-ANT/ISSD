# Note
This is the repository for the paper entitled "ISSD: Indicator Selection for Time Series State Detection".

# Usage
If you only want to use ISSD, please refer to ```usage_example.py```, which contains detailed examples of ISSD.  
Note that the master branch is mainly for the reproducibility, A lite version of ISSD will soon be available at the lite branch.

# Reproducibility
To reproduce the results reported in the paper, please refer to the following guideline.
## Datasets
download all datasets and place them in ```data/raw```

```bash
python datautils/convert_data_format.py
```

## Environments
We strongly recommend configuring the environments separately for selection/reduction methods and downstream methods to avoid conflicts. Please prepare two virtual environments and install the following requirements separately.
```
# selection/reduction environment
conda create -n selection python=3.6
conda activate selection
pip install -r requirements/selection.txt

# downstream environment
conda create -n selection python=3.9
conda activate downstream
pip install -r requirements/downstream.txt
```
## Automatic reproduction
Automatic scripts for running all downstream methods are placed in the ```tasks``` folder. You can automatically reproduce all experimental results reported in the paper by the following command:
```
bash tasks/run_all.sh <execution number>
```
In the paper, the experiment is independently conducted 5 times, you need to change the ```<execution number>``` and run the above command five times.

Once done, please use the following command to obtain final results:
```
bash tasks/summary.sh
```

## Run Selection/Reduction Methods
**Convert to the selection environment**  
```bash
python experiments/selection.py [e.g., MoCap] [e.g., issd] [dim, e.g., 4]
```
The above command takes 3 positional args:  
Datasets: ```[MoCap|SynSeg|USC-HAD|SynSeg|PAMAP2]```  
Downstream methods: ```[issd-qf|issd-cf|sfm|ecp|ecs|pca|umap]```  
Desired #channels: ```e.g., 4```

## Run Downstream Methods
**Convert to downstream environment**

Time2State, E2USD, TICC:
```bash
python experiments/run_dmethods.py [e.g.,time2state] [e.g., MoCap] [e.g., issd]
```

AutoPlait is special, which uses c implementation:  
In root folder, run

```bash
python downstream_methods/AutoPlait/experiments/convert_to_AutoPlait_format.py
```

In downstream_methods/AutoPlait folder
```bash
bash experiments/runAutoPlait.sh [e.g.,MoCap] [e.g., pca]
```

In root folder
```bash
python downstream_methods/AutoPlait/experiments/redirect_results.py
```

Automatic scripts for running all downstream methods are placed in the ```tasks``` folder
## Results
To obtain the results, please run
```bash
python experiments/summary.py [ari|nmi|purity]
python paper_figure/plot_overall_performance.py [ari|nmi|purity]
```
The summarized txt files, figures will be saved in the output folder

## Visualization of Results
```bash
python experiments/visualization.py
```
The visulalization results will be saved in output/case-studies/
