import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from alg_runner import sim_annealing_runner, rhc_runner
from plotting import *

np.random.seed(1)
problem_size = 100


if __name__ == "__main__":

    # TODO Write state regeneration functions as lamdas
    peaks_fit = mlrose.ContinuousPeaks(t_pct=.1)
    init_state = np.random.randint(2, size=problem_size)
    problem = mlrose.DiscreteOpt(length=problem_size, fitness_fn=peaks_fit, maximize=True, max_val=2)

    print("Running simulated annealing montecarlos")
    results = sim_annealing_runner(problem, init_state)
    plot_simulated_annealing('CPeaks', 'sim_anneal', results)

    print("Running random hill montecarlos")
    results = rhc_runner(problem, init_state)
    plot_simulated_annealing('CPeaks', 'rhc', results)

    print("Running genetic algorithm montecarlos")
    results = rhc_runner(problem, init_state)
    plot_simulated_annealing('CPeaks', 'ga', results)

    print("Running MIMIC montecarlos")
    results = rhc_runner(problem, init_state)
    plot_simulated_annealing('CPeaks', 'mimic', results)

