import mlrose
import numpy as np

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from alg_runner import sim_annealing_runner, rhc_runner, ga_runner, mimic_runner
from plotting import plot_montecarlo_sensitivity

import os
import pickle
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


np.random.seed(1)


def run_flipflop():

    # If the output/Cpeaks directory doesn't exist, create it.
    if not os.path.exists('./output/FlipFlop/'):
        os.mkdir('./output/FlipFlop/')

    # TODO Write state regeneration functions as lamdas
    problem_size = 50
    logger = logging.getLogger(__name__)
    flip_fit = mlrose.FlipFlop()
    flop_state_gen = lambda: np.random.randint(2, size=problem_size)
    init_state = flop_state_gen()
    problem = mlrose.DiscreteOpt(length=problem_size, fitness_fn=flip_fit, maximize=True, max_val=2)

    all_results = {}

    print("Running simulated annealing montecarlos")
    sa_results, sa_timing = sim_annealing_runner(problem, init_state, state_regenerator=flop_state_gen)
    plot_montecarlo_sensitivity('FlipFlop', 'sim_anneal', sa_results)
    plot_montecarlo_sensitivity('FlipFlop', 'sim_anneal_timing', sa_timing)
    all_results['SA'] = [sa_results, sa_timing]


    print("Running random hill montecarlos")
    rhc_results, rhc_timing = rhc_runner(problem, init_state, state_regenerator=flop_state_gen)
    plot_montecarlo_sensitivity('FlipFlop', 'rhc', rhc_results)
    plot_montecarlo_sensitivity('FlipFlop', 'rhc_timing', sa_timing)
    all_results['RHC'] = [rhc_results, rhc_timing]

    print("Running genetic algorithm montecarlos")
    ga_results, ga_timing = ga_runner(problem, init_state, state_regenerator=flop_state_gen)
    plot_montecarlo_sensitivity('FlipFlop', 'ga', ga_results)
    plot_montecarlo_sensitivity('FlipFlop', 'ga_timing', ga_timing)
    all_results['GA'] = [ga_results, ga_timing]

    print("Running MIMIC montecarlos")
    mimic_results, mimic_timing = mimic_runner(problem, init_state, state_regenerator=flop_state_gen)
    plot_montecarlo_sensitivity('FlipFlop', 'mimic', mimic_results)
    plot_montecarlo_sensitivity('FlipFlop', 'mimic_timing', mimic_timing)
    all_results['MIMIC'] = [mimic_results, mimic_timing]


    with open('./output/FlipFlop/flipflip_data.pickle', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    problem_size_space = np.linspace(10, 125, 20, dtype=int)

    best_fit_dict = {}
    best_fit_dict['Problem Size'] = problem_size_space
    best_fit_dict['Random Hill Climbing'] = []
    best_fit_dict['Simulated Annealing'] = []
    best_fit_dict['Genetic Algorithm'] = []
    best_fit_dict['MIMIC'] = []


    times = {}
    times['Problem Size'] = problem_size_space
    times['Random Hill Climbing'] = []
    times['Simulated Annealing'] = []
    times['Genetic Algorithm'] = []
    times['MIMIC'] = []


    fits_per_iteration = {}
    fits_per_iteration['Random Hill Climbing'] = []
    fits_per_iteration['Simulated Annealing'] = []
    fits_per_iteration['Genetic Algorithm'] = []
    fits_per_iteration['MIMIC'] = []


    for prob_size in problem_size_space:
        logger.info("---- Problem size: " + str(prob_size) + " ----")
        prob_size_int = int(prob_size)
        flip_fit = mlrose.FlipFlop()
        flop_state_gen = lambda: np.random.randint(2, size=prob_size_int)
        init_state = flop_state_gen()
        problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=flip_fit, maximize=True, max_val=2)

        start = datetime.now()
        _, best_fitness_sa, fit_array_sa = mlrose.simulated_annealing(problem,
                                                                   schedule=mlrose.ExpDecay(exp_const=.001, init_temp=2),
                                                                   max_attempts=20, 
                                                                   max_iters=10000, init_state=init_state, track_fits=True)
        best_fit_dict['Simulated Annealing'].append(best_fitness_sa)
        end = datetime.now()
        times['Simulated Annealing'].append((end-start).total_seconds())


        start = datetime.now()
        _, best_fitness_rhc, fit_array_rhc = mlrose.random_hill_climb(problem, max_attempts=200, max_iters=10000,
                                                                  restarts=20, track_fits=True)
        best_fit_dict['Random Hill Climbing'].append(best_fitness_rhc)
        end = datetime.now()
        times['Random Hill Climbing'].append((end-start).total_seconds())


        start = datetime.now()
        _, best_fitness_ga, fit_array_ga =  mlrose.genetic_alg(problem, pop_size=prob_size_int*5,
                                                            mutation_prob=.025, max_attempts=20, track_fits=True, max_iters=2000)
        best_fit_dict['Genetic Algorithm'].append(best_fitness_ga)
        end = datetime.now()
        times['Genetic Algorithm'].append((end-start).total_seconds())


        start = datetime.now()
        _, best_fitness_mimic, fit_array_mimic = mlrose.mimic(problem, pop_size=prob_size_int*3,
                                                        keep_pct=.25, max_attempts=20, track_fits=True, max_iters=500)
        best_fit_dict['MIMIC'].append(best_fitness_mimic)
        end = datetime.now()
        times['MIMIC'].append((end-start).total_seconds())


    # For the last fit that occurs, save off the fit arrays that are generated. We will plot fitness/iteration.
    fits_per_iteration['Random Hill Climbing'] = fit_array_rhc
    fits_per_iteration['Simulated Annealing'] = fit_array_sa
    fits_per_iteration['Genetic Algorithm'] = fit_array_ga
    fits_per_iteration['MIMIC'] = fit_array_mimic

    fit_frame = pd.DataFrame.from_dict(best_fit_dict, orient='index').transpose()
    # fit_frame.pop('Unnamed: 0') # idk why this shows up.
    time_frame = pd.DataFrame.from_dict(times, orient='index').transpose()
    # time_frame.pop('Unnamed: 0') # idk why this shows up.
    fit_iteration_frame = pd.DataFrame.from_dict(fits_per_iteration, orient='index').transpose()

    fit_frame.to_csv('./output/FlipFlop/problem_size_fit.csv')
    time_frame.to_csv('./output/FlipFlop/problem_size_time.csv')
    fit_iteration_frame.to_csv('./output/FlipFlop/fit_per_iteration.csv')

if __name__ == "__main__":
    run_flipflop()