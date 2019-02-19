import mlrose
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from alg_runner import sim_annealing_runner, rhc_runner, ga_runner, mimic_runner
from plotting import plot_montecarlo_sensitivity

np.random.seed(1)
problem_size = 50


if __name__ == "__main__":

    # TODO Write state regeneration functions as lamdas
    peaks_fit = mlrose.ContinuousPeaks(t_pct=.1)
    cpeaks_state_gen = lambda: np.random.randint(2, size=problem_size)
    init_state = cpeaks_state_gen()
    problem = mlrose.DiscreteOpt(length=problem_size, fitness_fn=peaks_fit, maximize=True, max_val=2)

    print("Running simulated annealing montecarlos")
    results, timing = sim_annealing_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'sim_anneal', results)
    plot_montecarlo_sensitivity('CPeaks', 'sim_anneal_timing', timing)

    print("Running random hill montecarlos")
    results, timing = rhc_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'rhc', results)
    plot_montecarlo_sensitivity('CPeaks', 'rhc_timing', timing)

    print("Running genetic algorithm montecarlos")
    results, timing = ga_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'ga', results)
    plot_montecarlo_sensitivity('CPeaks', 'ga_timing', timing)

    print("Running MIMIC montecarlos")
    results, timing = mimic_runner(problem, init_state, state_regenerator=cpeaks_state_gen)
    plot_montecarlo_sensitivity('CPeaks', 'mimic', results)
    plot_montecarlo_sensitivity('CPeaks', 'mimic_timing', timing)






    # Travelling sales?

