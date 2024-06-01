#!/bin/bash
python experiments/summary.py
python paper_figures/plot_cd.py
python paper_figures/plot_overall_performance.py
python paper_figures/plot_individual_performance.py
python experiments/visualization.py