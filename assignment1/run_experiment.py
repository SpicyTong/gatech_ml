import argparse
from datetime import datetime
import numpy as np
import sys

import experiments
from data import loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some SL experiments')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--ann', action='store_true', help='Run the ANN experiment')
    parser.add_argument('--boosting', action='store_true', help='Run the Boosting experiment')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--knn', action='store_true', help='Run the KNN experiment')
    parser.add_argument('--svm', action='store_true', help='Run the SVM experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--no_set1', action='store_true', help='Disables set 1')
    parser.add_argument('--no_set2', action='store_true', help='Disables set 2')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        print("Using seed {}".format(seed))

    print("Loading data")
    print("----------")

    if not args.no_set1:
        ds1_data = loader.SteelPlateData(verbose=verbose, seed=seed, binarize=False)
        ds1_name = 'SteelPlate'
        ds1_readable_name = 'SteelPlate'
        ds1_data.load_and_process()

    if not args.no_set2:
        ds2_data = loader.AusWeather(verbose=verbose, seed=seed)
        ds2_name = 'AusWeather'
        ds2_readable_name = 'AusWeather'
        ds2_data.load_and_process()

    if verbose:
        print("----------")
    print("Running experiments")

    timings = {}
    experiments_to_run = []

    # sys.exit()

    if not args.no_set1:
        experiment_details_ds1 = experiments.ExperimentDetails(
            ds1_data, ds1_name, ds1_readable_name,
            threads=threads,
            seed=seed
        )
        experiments_to_run.append(experiment_details_ds1)


    if not args.no_set2:
        experiment_details_ds2 = experiments.ExperimentDetails(
            ds2_data, ds2_name, ds2_readable_name,
            threads=threads,
            seed=seed
        )
        experiments_to_run.append(experiment_details_ds2)

    if args.ann or args.all:
        t = datetime.now()
        for details in experiments_to_run:
            experiment = experiments.ANNExperiment(details, verbose=verbose)
            experiment.perform()
        t_d = datetime.now() - t
        timings['ANN'] = t_d.seconds

    if args.boosting or args.all:
        t = datetime.now()
        for details in experiments_to_run:
            experiment = experiments.BoostingExperiment(details, verbose=verbose)
            experiment.perform()
        t_d = datetime.now() - t
        timings['Boost'] = t_d.seconds

    if args.dt or args.all:
        t = datetime.now()
        for details in experiments_to_run:
            experiment = experiments.DTExperiment(details, verbose=verbose)
            experiment.perform()
        t_d = datetime.now() - t
        timings['DT'] = t_d.seconds

    if args.knn or args.all:
        t = datetime.now()
        for details in experiments_to_run:
            experiment = experiments.KNNExperiment(details, verbose=verbose)
            experiment.perform()
        t_d = datetime.now() - t
        timings['KNN'] = t_d.seconds

    if args.svm or args.all:
        t = datetime.now()
        for details in experiments_to_run:
            experiment = experiments.SVMExperiment(details, verbose=verbose)
            experiment.perform()
        t_d = datetime.now() - t
        timings['SVM'] = t_d.seconds

    print(timings)
