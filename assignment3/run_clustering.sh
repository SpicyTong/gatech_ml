#!/bin/sh

# Replace 'X' below with the optimal values found
# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below

python run_experiment.py --ica --skyserver --dim 10  --verbose --threads 3 > ica-skyserver-clustering.log 2>&1
python run_experiment.py --ica --ausweather   --dim 10  --skiprerun --verbose --threads 3 > ica-ausweather-clustering.log   2>&1
python run_experiment.py --pca --skyserver --dim 5 --skiprerun --verbose --threads 3 > pca-skyserver-clustering.log 2>&1
python run_experiment.py --pca --ausweather   --dim 8  --skiprerun --verbose --threads 3 > pca-ausweather-clustering.log   2>&1
python run_experiment.py --rp  --skyserver --dim 8  --skiprerun --verbose --threads 3 > rp-skyserver-clustering.log  2>&1
python run_experiment.py --rp  --ausweather   --dim 9  --skiprerun --verbose --threads 3 > rp-ausweather-clustering.log    2>&1
python run_experiment.py --rf  --skyserver --dim 3 --skiprerun --verbose --threads 3 > rf-skyserver-clustering.log  2>&1
python run_experiment.py --rf  --ausweather   --dim 3  --skiprerun --verbose --threads 3 > rf-ausweather-clustering.log    2>&1
#python run_experiment.py --svd --skyserver --dim X  --skiprerun --verbose --threads 3 > svd-skyserver-clustering.log 2>&1
#python run_experiment.py --svd --ausweather   --dim X  --skiprerun --verbose --threads 3 > svd-ausweather-clustering.log   2>&1
