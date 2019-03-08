#!/bin/sh

# Replace 'X' below with the optimal values found
# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below
# Add  --dim X  --skiprerun after running this one. 
python run_experiment.py --ica --skyserver  --verbose --threads -1 > ica-skyserver-clustering.log 2>&1
python run_experiment.py --ica --ausweather --verbose --threads -1 > ica-ausweather-clustering.log   2>&1
python run_experiment.py --pca --skyserver --verbose --threads -1 > pca-skyserver-clustering.log 2>&1
python run_experiment.py --pca --ausweather --verbose --threads -1 > pca-ausweather-clustering.log   2>&1
python run_experiment.py --rp  --skyserver --verbose --threads -1 > rp-skyserver-clustering.log  2>&1
python run_experiment.py --rp  --ausweather --verbose --threads -1 > rp-ausweather-clustering.log    2>&1
python run_experiment.py --rf  --skyserver --verbose --threads -1 > rf-skyserver-clustering.log  2>&1
python run_experiment.py --rf  --ausweather --verbose --threads -1 > rf-ausweather-clustering.log    2>&1
python run_experiment.py --svd --skyserver --verbose --threads -1 > svd-skyserver-clustering.log 2>&1
python run_experiment.py --svd --ausweather --verbose --threads -1 > svd-ausweather-clustering.log   2>&1
