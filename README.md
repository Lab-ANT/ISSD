This is the repository for the paper entitled "ISSD: Indicator Selection for Time Series State Detection"

# Usage
If you only want to use ISSD or IRSD, please refer to ```usage_example.py```, which contains detailed examples of ISSD and IRSD.

# Reproducibility
To reproduce the results reported in the paper, please refer to the following guideline.
## Datasets
download all datasets and place the them in ```data/raw```

```bash
python datautils/convert_data_format.py
```

## Environments
We strongly recommend configuring the environments separately for selection methods, reduction methods, and downstream methods to avoid conflicts. Please prepare three virtual environments and install the following requirements separately
```
requirements/selection.txt
requirements/downstream.txt
requirements/reduction.txt
```

## Run Selection/Reduction Methods
**Convert to the reduction environment**  
Dimension reduction methods and human selection:
```bash
python experiments/reduction.py
```

**Convert to the selection environment**  
ISSD, SFS, ECP, ECS:
```bash
python experiments/selection.py [e.g., MoCap] [e.g., issd-qf] [dim, e.g., 4]
```

## Run Downstream Methods
**Convert to downstream environment**

Time2State, E2USD, TICC, ClaSP, GHMM:
```bash
python experiments/run_dmethods.py [e.g.,time2state] [e.g., MoCap] [e.g., issd]
```

AutoPlait is special, which uses c implementation:  
In root folder, sequentially run

```bash
python downstream_methods/AutoPlait/experiments/convert_to_AutoPlait_format.py
```

In downstream_methods/AutoPlait folder
```bash
bash experiments runAutoPlait.sh [dataset] [pca|umap|human|raw]
```

In root folder
```bash
python downstream_methods/AutoPlait/experiments/redirect_results.py
```

## Results
```bash
python experiments/summary.py
python paper_figure/plot_overall_performance.py
```

## Visualization of Results
```bash
python experiments/visualization.py
```
The summary is saved in output/