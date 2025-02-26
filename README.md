This is the repository for the paper entitled "ISSD: Indicator Selection for Time Series State Detection".

Feel free to contect chengyu@nudt.edu.cn if you have any problem.

# Usage and Note

The default branch (**main**) is for the reproducibility. A lite version (only ISSD) will be released on the **lite** branch. If you only want to use ISSD, please refer to ```usage_example.py```, which contains detailed examples of ISSD.  

# Reproducibility
To reproduce the results, please refer to the following guideline.

## Environments
The selection and downstream environment must be separately configured to avoid conflicts. Please prepare two virtual environments and install the following requirements separately.
```bash
# selection/reduction environment
conda create -n selection python=3.8.20
conda activate selection
pip install -r requirements/selection.txt
# use other source as needed, e.g., -i https://pypi.tuna.tsinghua.edu.cn/simple

# downstream environment
conda create -n selection python=3.9.21
conda activate downstream
pip install -r requirements/downstream.txt
# use other source as needed, e.g., -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Datasets
Download PAMAP2, USC-HAD, and ActRecTut and place them in ```data/raw```  
The MoCap and SynSeg datasets have already been placed in the repository.

```
.
├── data
│   ├── raw
│   │    ├── ActRecTut
│   │    │   ├── subject1_gesture
│   │    │   ├── ...
│   │    ├── PAMAP2
│   │    │   ├── Protocol
│   │    │   │   ├── subject101.dat
│   │    │   │   ├── ...
│   │    ├── USC-HAD
│   │    │   ├── Subject1
│   │    │   │   ├── subject101.dat
│   │    │   │   ├── ...
│   │    │   ├── ...
│   ├── MoCap
│   ├── SynSeg
```

run in **selection** environment:
```bash
python datautils/convert_data_format.py
```

**Note**: The data generation and statistic tools are also placed in the ```datautils``` folder, you may use these scripts to generate your own data.

## Config conda home

Remember to configure the conda args in the 7-th line in ```tasks/run_all.sh``` and ```tasks/run_one.sh```.

```bash
# modify line 7 in tasks/run_all.sh and tasks/run_one.sh
source <your conda home>/etc/profile.d/conda.sh
# e.g., ~/anaconda3/etc/profile.d/conda.sh
```

## Run selection and downstream methods
```
bash tasks/run_all.sh <execution id>
```
In the paper, the experiment is independently conducted 5 times, you need to change the ```<execution id>``` from 1 to 5 and run the above command five times.

Once done, please use the following command to obtain final results:
```bash
python experiments/average_execution.py
```

## Only run selection/reduction methods
You can also separately run each selection/reduction methods.

**Convert to the selection environment**  
```bash
python experiments/selection.py <dataset, e.g., MoCap> <method, e.g., issd> <dim, e.g., 4>
```
The above command takes 3 positional args:  
Datasets: ```[MoCap|SynSeg|USC-HAD|SynSeg|PAMAP2]```  
Selection/Reduction methods: ```[issd|sfm|lda|ecp|ecs|pca|umap]```  
Desired #channels: ```e.g., 4```

## Only run downstream methods
You can also separately run each downstream methods.

**Convert to downstream environment**

If you want to run Time2State, E2USD, TICC:
```bash
python experiments/run_dmethods.py <dmethod, e.g.,time2state> <dataset, e.g., MoCap> <method, e.g., issd>
```

The above command takes 3 positional args:  
Downstream methods: ```[time2state|e2usd|ticc]```  
Datasets: ```[MoCap|SynSeg|USC-HAD|SynSeg|PAMAP2]```  
Selection/Reduction methods: ```[issd|sfm|lda|ecp|ecs|pca|umap]```

You can specify the correlation threshold for ISSD through ```--corr```, e.g., ```--corr 0.8```

**AutoPlait is special, which uses c implementation:  
In root folder, run**

```bash
python downstream_methods/AutoPlait/experiments/convert_to_AutoPlait_format.py
```

In downstream_methods/AutoPlait folder
```bash
bash experiments/runAutoPlait.sh <dataset, e.g.,MoCap> <method, e.g., pca>
```

In root folder
```bash
python downstream_methods/AutoPlait/experiments/redirect_results.py
```

## Results
To inspect the results, please run
```bash
python experiments/summary.py <ari|nmi|purity>
python paper_figures/plot_overall_performance.py <ari|nmi|purity>
python paper_figures/plot_cd.py
python paper_figures/plot_box.py
# use other scripts in the paper_figures/ folder to draw other figures.
```
The summarized txt files, figures will be saved in the output folder

## Visualization of selection results

The visualization tool is placed in the ```experiments/``` folder. Please run the selection/reduction methods that you want to visualization in advance, then run the following command.

```bash
python experiments/visualization.py
```
The visulalization results will be saved in output/visualization/

This tool can still run without running all selection/reduction methods, but it will skip methods without results.

## Other experiments

Other experiments, including case studies and more experiments are placed in ```experiments/``` folder. The corresponding scripts for visualizing results are placed in ```paper_figures``` folder. You can simply run the experiment first, and then run the corresponding visualization script.

## Archive and recover execution

To archive or recover an execution, run

```bash
bash tasks/archive_execution.sh <execution_id>
```

the ```output``` folder and selection results will be archived to the ```archive``` folder, e.g., ```bash tasks/archive_execution.sh 1``` will generate a ```execution1``` folder in the ```archive``` folder.

To recover the archive, run

```bash
bash recover_archived_execution.sh <execution_id>
```
the archived selection results and output will be recovered to ```data/``` and ```output/``` folder respectively.

This feature is used to conduct multiple independent experiments and save the environment for each experiment.

