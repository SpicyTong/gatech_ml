import mlrose

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime

MONTECARLO_COUNT = 7
MC_RUNS = range(MONTECARLO_COUNT)

def sim_annealing_runner(problem, state, state_regenerator=None):
    """
    Runs the simulated annealing experiment on a given problem. 

    Free variables are:
    Initial temperature
    Final temperature
    Decay rate
    # of attempts
    # of iterations
    """
    problem_size = len(state)
    initial_temps = np.arange(.1, 5, .1)
    final_temps = np.arange(.001, 1, .1)
    decay_rates = np.arange(.001, 1, .1)
    attempts = np.arange(10, 500, 10).astype(int)
    iterations = np.arange(10, 500, 10).astype(int)

    scoring_dict = {}
    timing_dict = {}

    init_temp_scores = []
    final_temp_scores = []

    for i, temp in enumerate(initial_temps):
        anneal_schedule = mlrose.ExpDecay(init_temp=temp)
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.simulated_annealing(problem, schedule=anneal_schedule, max_attempts=100, 
                                                            max_iters=800, init_state=init_state)
            best_fits.append(best_fitness)

        init_temp_scores.append(best_fits)
    scoring_dict['Initial Temperature'] = pd.DataFrame(init_temp_scores,
                                                        columns=MC_RUNS,
                                                        index=initial_temps)

    for i, temp in enumerate(final_temps):
        anneal_schedule = mlrose.ExpDecay(min_temp=temp)
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.simulated_annealing(problem, schedule=anneal_schedule, max_attempts=100, 
                                                                    max_iters=800, init_state=init_state)
            best_fits.append(best_fitness)

        final_temp_scores.append(best_fits)
    scoring_dict['Ending Temperature'] = pd.DataFrame(final_temp_scores, index=final_temps, columns=MC_RUNS)

    decay_scores = []
    for i, rate in enumerate(decay_rates):
        anneal_schedule = mlrose.ExpDecay(exp_const=rate)
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.simulated_annealing(problem, schedule=anneal_schedule, max_attempts=100, 
                                                            max_iters=800, init_state=init_state)
            best_fits.append(best_fitness)

        decay_scores.append(best_fits)
    scoring_dict['Decay Rate'] = pd.DataFrame(decay_scores, columns=MC_RUNS, index=decay_rates)



    attempts_scores = []
    attempts_timing = []
    for i, att in enumerate(attempts):
        anneal_schedule = mlrose.ExpDecay()
        best_fits = []
        times = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            start = datetime.now()
            _, best_fitness = mlrose.simulated_annealing(problem, schedule=anneal_schedule, max_attempts=int(att), 
                                                          max_iters=800, init_state=init_state)
            end = datetime.now()
            best_fits.append(best_fitness)
            times.append((end-start).total_seconds())

        attempts_scores.append(best_fits)
        attempts_timing.append(times)
    scoring_dict['Number of Attempts'] = pd.DataFrame(attempts_scores, columns=MC_RUNS, index=attempts)
    timing_dict['Number of Attempts'] = pd.DataFrame(attempts_timing, columns=MC_RUNS, index=attempts)

    iteration_scores = []
    for i, iteration in enumerate(iterations):
        anneal_schedule = mlrose.ExpDecay()
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.simulated_annealing(problem, schedule=anneal_schedule, max_attempts=100, 
                                                            max_iters=int(iteration), init_state=init_state)
            best_fits.append(best_fitness)
        iteration_scores.append(best_fits)
    scoring_dict['Max Iterations'] = pd.DataFrame(iteration_scores, columns=MC_RUNS, index=iterations)

        
    return scoring_dict, timing_dict



def rhc_runner(problem, state, gen_plots=True, state_regenerator=None):
    """
    Runs the simulated annealing experiment on a given problem. 

    Free variables are:
    Initial temperature
    Final temperature
    Decay rate
    # of attempts
    # of iterations
    """
    problem_size = len(state)
    attempts = np.arange(10, 500, 10).astype(int)
    iterations = np.arange(10, 500, 10).astype(int)
    restarts = np.arange(0, 100, 10)

    scoring_dict = {}
    timing_dict = {}

    attempts_scores = []
    for i, att in enumerate(attempts):
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.random_hill_climb(problem, max_attempts=int(att), max_iters=500)
            best_fits.append(best_fitness)

        attempts_scores.append(best_fits)
    scoring_dict['Number of Attempts'] = pd.DataFrame(attempts_scores, columns=MC_RUNS, index=attempts)


    iteration_scores = []
    for i, iteration in enumerate(iterations):

        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=int(iteration))
            best_fits.append(best_fitness)
        iteration_scores.append(best_fits)
    scoring_dict['Max Iterations'] = pd.DataFrame(iteration_scores, columns=MC_RUNS, index=iterations)


    iteration_scores = []
    iteration_timing = []
    for i, rst in enumerate(restarts):
        best_fits = []
        times = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            start = datetime.now()
            _, best_fitness = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=500, restarts=int(rst))
            end = datetime.now()
            best_fits.append(best_fitness)
            times.append((end-start).total_seconds())
        iteration_scores.append(best_fits)
        iteration_timing.append(times)
    scoring_dict['Random Restarts'] = pd.DataFrame(iteration_scores, columns=MC_RUNS, index=restarts)
    timing_dict['Random Restarts'] = pd.DataFrame(iteration_timing, columns=MC_RUNS, index=restarts)

    return scoring_dict, timing_dict


def ga_runner(problem, state, gen_plots=True, state_regenerator=None):
    """
    Runs the simulated annealing experiment on a given problem. 

    Free variables are:
    Initial temperature
    Final temperature
    Decay rate
    # of attempts
    # of iterations
    """
    problem_size = len(state)
    attempts = np.arange(10, 500, 50).astype(int)
    pop_size = np.arange(problem_size//4, problem_size * 4, problem_size//8).astype(int)
    mutation_probs = np.arange(.01, .5, .05)

    scoring_dict = {}
    timing_dict = {}

    attempts_scores = []
    for i, att in enumerate(attempts):
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness =  mlrose.genetic_alg(problem, pop_size=int(problem_size), mutation_prob=.1, max_attempts=int(att))
            best_fits.append(best_fitness)

        attempts_scores.append(best_fits)
    scoring_dict['Number of Attempts'] = pd.DataFrame(attempts_scores, columns=MC_RUNS, index=attempts)


    population_scores = []
    population_timing = []
    for i, psize in enumerate(pop_size):
        times = []
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            start = datetime.now()
            _, best_fitness = mlrose.genetic_alg(problem, pop_size=int(psize), mutation_prob=.1, max_attempts=100)
            end = datetime.now()
            best_fits.append(best_fitness)
            times.append((end-start).total_seconds())
        population_scores.append(best_fits)
        population_timing.append(times)
    scoring_dict['Population Size'] = pd.DataFrame(population_scores, columns=MC_RUNS, index=pop_size)
    timing_dict['Population Size'] = pd.DataFrame(population_timing, columns=MC_RUNS, index=pop_size)


    mutation_scores = []
    for i, mprob in enumerate(mutation_probs):

        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            _, best_fitness = mlrose.genetic_alg(problem, pop_size=int(problem_size), mutation_prob=mprob, max_attempts=100)
            best_fits.append(best_fitness)
        mutation_scores.append(best_fits)
    scoring_dict['Mutation Probability'] = pd.DataFrame(mutation_scores, columns=MC_RUNS, index=mutation_probs)

    return scoring_dict, timing_dict



def mimic_runner(problem, state, gen_plots=True, state_regenerator=None):
    """
    Runs the simulated annealing experiment on a given problem. 

    Free variables are:
    Initial temperature
    Final temperature
    Decay rate
    # of attempts
    # of iterations
    """
    problem_size = len(state)
    attempts = np.arange(25, 25, 1).astype(int)
    pop_size = np.arange(problem_size//4, problem_size * 6, problem_size//4).astype(int)
    percents = np.arange(.005, .3, .01)

    scoring_dict = {}
    timing_dict = {}


    attempts_scores = []
    attempt_timing = []
    # print("Running attempt sweep")
    for i, att in enumerate(attempts):
        best_fits = []
        times = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            print("Running attempt sweep " + str(att) + ": " + str(i))
            start = datetime.now()
            _, best_fitness =  mlrose.mimic(problem, pop_size=problem_size, keep_pct=.3, max_attempts=int(att))
            end = datetime.now()
            best_fits.append(best_fitness)
            times.append((end-start).total_seconds())
        attempt_timing.append(times)
        attempts_scores.append(best_fits)
    scoring_dict['NumberofAttempts'] = pd.DataFrame(attempts_scores, columns=MC_RUNS, index=attempts)
    timing_dict['NumberofAttempts'] = pd.DataFrame(attempt_timing, columns=MC_RUNS, index=attempts)


    population_scores = []
    population_timing = []
    print("Running population size sweep")
    for i, psize in enumerate(pop_size):
        times = []
        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            print("Running population size sweep " + str(psize) + ": " + str(i))
            start = datetime.now()
            _, best_fitness = mlrose.mimic(problem, pop_size=int(psize), keep_pct=.3, max_attempts=10)
            end = datetime.now()
            best_fits.append(best_fitness)
            times.append((end-start).total_seconds())
        population_scores.append(best_fits)
        population_timing.append(times)
    scoring_dict['PopulationSize'] = pd.DataFrame(population_scores, columns=MC_RUNS, index=pop_size)
    timing_dict['PopulationSize'] = pd.DataFrame(population_timing, columns=MC_RUNS, index=pop_size)

    mutation_scores = []
    print("Running mutation rate sweep")
    for i, prcnt in enumerate(percents):

        best_fits = []
        for i in MC_RUNS:
            init_state = state_regenerator()
            print("Running percentage kept sweep " + str(prcnt) + ": " + str(i))
            _, best_fitness = mlrose.mimic(problem, pop_size=problem_size*2, keep_pct=prcnt, max_attempts=20)
            best_fits.append(best_fitness)
        mutation_scores.append(best_fits)
    scoring_dict['PercentageKept'] = pd.DataFrame(mutation_scores, columns=MC_RUNS, index=percents)

    return scoring_dict, timing_dict