import mlrose
import numpy as np
import pickle
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import os

from alg_runner import sim_annealing_runner, rhc_runner, ga_runner, mimic_runner
from plotting import plot_montecarlo_sensitivity

from datetime import datetime
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(1)


def run_cpeaks():
    
    
    # If the output/Cpeaks directory doesn't exist, create it.
    if not os.path.exists('./output/CPeaks/'):
        os.mkdir('./output/CPeaks/')

    problem_size = 50
    peaks_fit = mlrose.ContinuousPeaks(t_pct=.1)
    cpeaks_state_gen = lambda: np.random.randint(2, size=problem_size)
    init_state = cpeaks_state_gen()
    problem = mlrose.DiscreteOpt(length=problem_size, fitness_fn=peaks_fit, maximize=True, max_val=2)

    all_results = {}

    print("Running simulated annealing montecarlos")
    sa_results, sa_timing = sim_annealing_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'sim_anneal', sa_results)
    plot_montecarlo_sensitivity('CPeaks', 'sim_anneal_timing', sa_timing)
    all_results['SA'] = [sa_results, sa_timing]


    print("Running random hill montecarlos")
    rhc_results, rhc_timing = rhc_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'rhc', rhc_results)
    plot_montecarlo_sensitivity('CPeaks', 'rhc_timing', sa_timing)
    all_results['RHC'] = [rhc_results, rhc_timing]

    print("Running genetic algorithm montecarlos")
    ga_results, ga_timing = ga_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'ga', ga_results)
    plot_montecarlo_sensitivity('CPeaks', 'ga_timing', ga_timing)
    all_results['GA'] = [ga_results, ga_timing]

    print("Running MIMIC montecarlos")
    mimic_results, mimic_timing = mimic_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'mimic', mimic_results)
    plot_montecarlo_sensitivity('CPeaks', 'mimic_timing', mimic_timing)
    all_results['MIMIC'] = [mimic_results, mimic_timing]


    with open('./output/CPeaks/cpeaks_data.pickle', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)



    problem_size_space = np.linspace(10, 100, 20, dtype=int)

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

    for prob_size in problem_size_space:
        logger.info("---- Problem size: " + str(prob_size) + " ----")
        prob_size_int = int(prob_size)
        peaks_fit = mlrose.ContinuousPeaks(t_pct=.2)
        problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=peaks_fit, maximize=True, max_val=2)
        cpeaks_state_gen = lambda: np.random.randint(2, size=prob_size_int)
        init_state = cpeaks_state_gen()

        start = datetime.now()
        _, best_fitness_sa, fit_array_sa = mlrose.simulated_annealing(problem,
                                                                   schedule=mlrose.ExpDecay(exp_const=.001, init_temp=5,
                                                                   min_temp=.01),
                                                                   max_attempts=50, 
                                                                   max_iters=20000, init_state=init_state, track_fits=True)
        best_fit_dict['Simulated Annealing'].append(best_fitness_sa)
        end = datetime.now()
        times['Simulated Annealing'].append((end-start).total_seconds())


        start = datetime.now()
        _, best_fitness_rhc, fit_array_rhc = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=3000,
                                                                  restarts=50, track_fits=True)
        best_fit_dict['Random Hill Climbing'].append(best_fitness_rhc)
        end = datetime.now()
        times['Random Hill Climbing'].append((end-start).total_seconds())


        start = datetime.now()
        _, best_fitness_ga, fit_array_ga =  mlrose.genetic_alg(problem, pop_size=prob_size_int*10,
                                                            mutation_prob=.025, max_attempts=30, track_fits=True, max_iters=1000)
        best_fit_dict['Genetic Algorithm'].append(best_fitness_ga)
        end = datetime.now()
        times['Genetic Algorithm'].append((end-start).total_seconds())


        start = datetime.now()
        _, best_fitness_mimic, fit_array_mimic = mlrose.mimic(problem, pop_size=prob_size_int*10,
                                                        keep_pct=.1, max_attempts=30, track_fits=True, max_iters=2000)
        best_fit_dict['MIMIC'].append(best_fitness_mimic)
        end = datetime.now()
        times['MIMIC'].append((end-start).total_seconds())

    fits_per_iteration = {}
    fits_per_iteration['Random Hill Climbing'] = fit_array_rhc
    fits_per_iteration['Simulated Annealing'] = fit_array_sa
    fits_per_iteration['Genetic Algorithm'] = fit_array_ga
    fits_per_iteration['MIMIC'] = fit_array_mimic

    fit_frame = pd.DataFrame.from_dict(best_fit_dict, orient='index').transpose()
    # fit_frame.pop('Unnamed: 0') # idk why this shows up.
    time_frame = pd.DataFrame.from_dict(times, orient='index').transpose()
    # time_frame.pop('Unnamed: 0') # idk why this shows up.
    fit_iteration_frame = pd.DataFrame.from_dict(fits_per_iteration, orient='index').transpose()



    fit_frame.to_csv('./output/CPeaks/problem_size_fit.csv')
    time_frame.to_csv('./output/CPeaks/problem_size_time.csv')
    fit_iteration_frame.to_csv('./output/CPeaks/fit_per_iteration.csv')

if __name__ == "__main__":
    run_cpeaks()
    


    # Run fitness at each iteration for a large sample size. 



        

