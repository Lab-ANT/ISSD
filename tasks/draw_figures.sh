#!/bin/bash
python experiments/summary.py
python paper_figures/plot_cd.py
python paper_figures/plot_overall_performance.py
python paper_figures/plot_individual_performance.py
python paper_figures/plot_method_box.py
python paper_figures/plot_box.py
python experiments/visualization.py