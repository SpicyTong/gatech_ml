import argparse
import logging
import sys

import random as rand
import numpy as np

import experiments
from experiments import plotting
from datetime import datetime
from data import loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, dim, skiprerun, verbose, timings):
    t = datetime.now()
    for details in experiment_details:
        exp = experiment(details, verbose=verbose)

        if not skiprerun:
            logger.info("Running {} experiment: {} ({})".format(timing_key, details.ds_readable_name, dim))
            exp.perform()

        if dim is not None:
            logger.info("Running with dimension {}".format(dim))
            exp.perform_cluster(dim)
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some UL and DR')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--dim', type=int, help='The dim parameter to use for clustering with a specific experiment '
                                                '(This MUST be used with a specific experiment)')
    parser.add_argument('--skiprerun', action='store_true',
                        help='If true, do not re-run the main experiment before clustering '
                             '(This MUST be used with --dim and a specific experiment)')
    parser.add_argument('--skyserver', action='store_true', help='Run only SkyServer')
    parser.add_argument('--ausweather', action='store_true', help='Run only Ausweather')
    parser.add_argument('--benchmark', action='store_true', help='Run the benchmark experiments')
    parser.add_argument('--ica', action='store_true', help='Run the ICA experiments')
    parser.add_argument('--pca', action='store_true', help='Run the PCA experiments')
    parser.add_argument('--lda', action='store_true', help='Run the LDA experiments')
    parser.add_argument('--svd', action='store_true', help='Run the SVD experiments')
    parser.add_argument('--rf', action='store_true', help='Run the RF experiments')
    parser.add_argument('--rp', action='store_true', help='Run the RP experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    if args.dim or args.skiprerun:
        if not args.ica and not args.pca and not args.rf and not args.rp and not args.lda and not args.svd:
            logger.error("Cannot specify dimension/skiprerun without specifying a specific experiment")
            parser.print_help()
            sys.exit(1)

    if args.skyserver and not args.dim:
        logger.error("Cannot specify skiprerun without specifying a specific experiment")
        parser.print_help()
        sys.exit(1)

    if args.skyserver and args.ausweather:
        logger.error("Can only specify one of '--skyserver' or '--ausweather', not both")
        parser.print_help()
        sys.exit(1)

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Loading data")
    logger.info("----------")

    datasets = []
    skyserver_details = {
            'data': loader.SkyServerData(verbose=verbose, seed=seed),
            'name': 'skyserver',
            'readable_name': 'SkyServer',
        }
    ausweather_details = {
            'data': loader.AusWeather(verbose=verbose, seed=seed),
            'name': 'AusWeather',
            'readable_name': 'AusWeather',
        }
    if args.skyserver:
        datasets.append(skyserver_details)
    elif args.ausweather:
        datasets.append(ausweather_details)
    elif not args.skyserver and not args.ausweather:
        datasets.append(skyserver_details)
        datasets.append(ausweather_details)

    experiment_details = []
    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        data.build_train_test_split()
        data.scale_standard()
        experiment_details.append(experiments.ExperimentDetails(
            data, ds['name'], ds['readable_name'],
            threads=threads,
            seed=seed
        ))

    if args.all or args.benchmark or args.ica or args.pca or args.lda or args.svd or args.rf or args.rp:
        if verbose:
            logger.info("----------")

        logger.info("Running experiments")

        timings = {}

        if args.benchmark or args.all:
            run_experiment(experiment_details, experiments.BenchmarkExperiment, 'Benchmark', args.dim, args.skiprerun,
                           verbose, timings)
        if args.ica or args.all:
            run_experiment(experiment_details, experiments.ICAExperiment, 'ICA', args.dim, args.skiprerun,
                           verbose, timings)
        if args.pca or args.all:
            run_experiment(experiment_details, experiments.PCAExperiment, 'PCA', args.dim, args.skiprerun,
                           verbose, timings)
        # if args.lda or args.all:
        #     run_experiment(experiment_details, experiments.LDAExperiment, 'LDA', args.dim, args.skiprerun,
        #                    verbose, timings)
        # if args.svd or args.all:
        #     run_experiment(experiment_details, experiments.SVDExperiment, 'SVD', args.dim, args.skiprerun,
        #                    verbose, timings)
        if args.rf or args.all:
            run_experiment(experiment_details, experiments.RFExperiment, 'RF', args.dim, args.skiprerun,
                           verbose, timings)
        if args.rp or args.all:
            run_experiment(experiment_details, experiments.RPExperiment, 'RP', args.dim, args.skiprerun,
                           verbose, timings)

        logger.info("Timings: {}".format(timings))

    if args.plot:
        if verbose:
            logger.info("----------")

        logger.info("Plotting results")
        plotting.plot_results()
