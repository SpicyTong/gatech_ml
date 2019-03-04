import numpy as np
import pandas as pd 
import os
import logging

from plotting import plot_complexity, plot_neural_net_analysis


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


def analyze_complexity(dataset, dataset_name, algorithm_name):
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

        
def analyze_iterations(all_datasets, set_name):
    iteration_data = pd.DataFrame()
    # For each dataset, extract the iteration data.
    iter_param_name = 'param_max_iters'
    to_plot = 'mean_train_score'
    for alg in all_datasets.keys():
        dataset = all_datasets[alg]
        # We have to assume they're all the same length...
        iteration_runs = dataset[pd.notna(dataset[iter_param_name])]
        # Select the mean train score column
        iteration_data[alg] = iteration_runs[to_plot]


    # For whatever the last alg is, grab the iteration column itself.
    iteration_data['Iterations'] = all_datasets[alg][iter_param_name]

    iteration_data = iteration_data.set_index('Iterations')

    plot_neural_net_analysis(iteration_data, set_name, output_dir=OUTPUTDIR)





def plot_all_nn_results():
    datasets = ['SkyServer']
    algorithms = ['random_hill_climb', 'simulated_anneal', 'genetic_alg' , 'gradient_descent']

    for dataset_name in datasets:
        all_data = load_datasets(dataset_name, algorithms)
        for alg_name in all_data.keys():
            analyze_complexity(all_data[alg_name], dataset_name, alg_name)

        analyze_iterations(all_data, dataset_name)


if __name__ == '__main__':
    plot_all_nn_results()