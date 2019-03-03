import numpy as np
import pandas as pd 
import os
import logging

from plotting import plot_complexity


OUTPUTDIR = './output/Neural'

logger = logging.getLogger()

def load_datasets(dataset_name, algorithms):
    datasets = {}

    for prefix in algorithms:
        path = os.path.join(OUTPUTDIR, prefix + '_' + dataset_name + '_reg' + '.csv')
        if not os.path.exists(path):
            logger.warn("Unable to open file " + path)
            continue
        datasets[prefix] = pd.read_csv(path)

    return datasets


def plot_dataset_complexity(dataset, dataset_name, algorithm_name):
    # For the dataset, get names of columns that start with "params_"
    params = []
    for key in dataset.keys():
        if "param_" in key:
            params.append(key)

    # For each param, make a plot of the mean score with fills on stdev.
    for par in params:
        select = dataset[pd.notna(dataset[par])]
        partial_path = os.path.join(OUTPUTDIR, 'images', dataset_name + '_' + algorithm_name)
        plot_complexity(select, par, partial_path, pretty_name=dataset_name + " - " + algorithm_name)

        




if __name__ == '__main__':
    datasets = ['SkyServer', 'AusWeather']
    algorithms = ['random_hill_climb', 'simulated_anneal', 'genetic_alg' , 'gradient_descent']

    for dataset_name in datasets:
        all_data = load_datasets(dataset_name, algorithms)
        for alg_name in all_data.keys():
            plot_dataset_complexity(all_data[alg_name], dataset_name, alg_name)

