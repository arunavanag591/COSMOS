# COSMOS
Code base for the paper: "COSMOS: A Data-Driven Probabilistic Time Series Simulator for Chemical Plumes Across Spatial Scales" 

## Files:
1. [Training COSMOS Spatial](train.ipynb) : Demonstrates how to train a cosmos spatial model
2. [Testing Trajectory with COSMOS](test.ipynb) : Demonstration of how to use the spatial model and test a trajectory using the cosmos algorithm
3. [Agent Based Tracking](agent_tracking.ipynb) : Surge and Cast implementation using COSMOS and CFD [Rigolli](https://elifesciences.org/articles/72196) for odor experience.
4. [Agent Tracking trajectory comparison](trajectory_comparison.ipynb) : Agent based tracking using COSMOS and CFD, trajectory comparison and timing diagram
5. [COSMOS Algorithm for testing Trajectories](cosmos_batch.py)
6. [COSMOS Algorithm for use with agent tracking](cosmos_tracking.py)
5. [Helper for CFD methods](cfd_rigolli.py)
6. [Helper for odor statistics Calculation](odor_stat_calculations.py)


## Figures

 Below are interactive notebooks, which can be used using Jupyter Notebook and run using python 3.8 and inskcape to generate the figures and results. These figures were generated using [figurefirst](https://github.com/FlyRanch/figurefirst) 

#### Main Text Figures: 
1. [Figure 1](figure/algorithm_figure_v3.ipynb) : Overview of COSMOS algorithm 
2. [Figure 2](figure/results_hws.ipynb) : COSMOS results on HWS desert data
3. [Figure 3](figure/results_rigolli.ipynb) : COSMOS results on Rigolli odor simulator data
4. [Figure 4](figure/results_trackingv1.ipynb) : Agent based tracking using COSMOS and CFD, trajectory comparison and timing diagram


#### Supplemental Figure

1. [Figure 5](figure/results_lws.ipynb) : COSMOS results on HWS desert data
2. [Figure 6](figure/results_forest.ipynb) : COSMOS results on HWS desert data
3. [Figure 7](figure/S1.ipynb) : Binning of whiff statistics, in depth flow diagram for concentration modeling and intermittency modeling.



