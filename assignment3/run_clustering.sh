#!/bin/sh

# Replace 'X' below with the optimal values found
# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below

python run_experiment.py --ica --skyserver --dim 8  --verbose --threads -1 > ica-skyserver-clustering.log 2>&1
python run_experiment.py --ica --ausweather   --dim 10  --skiprerun --verbose --threads -1 > ica-ausweather-clustering.log   2>&1
python run_experiment.py --pca --skyserver --dim 8  --skiprerun --verbose --threads -1 > pca-skyserver-clustering.log 2>&1
python run_experiment.py --pca --ausweather   --dim 10  --skiprerun --verbose --threads -1 > pca-ausweather-clustering.log   2>&1
python run_experiment.py --rp  --skyserver --dim 8  --skiprerun --verbose --threads -1 > rp-skyserver-clustering.log  2>&1
python run_experiment.py --rp  --ausweather   --dim 10  --skiprerun --verbose --threads -1 > rp-ausweather-clustering.log    2>&1
python run_experiment.py --rf  --skyserver --dim 8  --skiprerun --verbose --threads -1 > rf-skyserver-clustering.log  2>&1
python run_experiment.py --rf  --ausweather   --dim 10  --skiprerun --verbose --threads -1 > rf-ausweather-clustering.log    2>&1
#python run_experiment.py --svd --skyserver --dim X  --skiprerun --verbose --threads -1 > svd-skyserver-clustering.log 2>&1
#python run_experiment.py --svd --ausweather   --dim X  --skiprerun --verbose --threads -1 > svd-ausweather-clustering.log   2>&1
