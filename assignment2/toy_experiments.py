# import cpeaks
# import knapsack
# import flipflop
import os
import pandas as pd
from plotting import plot_problem_size_time, plot_iteration_fit, plot_problem_size_scores

OUTPUT = './output'

def plot_toy_problems(problem_list, plot_time=True, plot_iterations=True, plot_scores=True):
    # For each problem, read the relevant CSVs and call the plot method.
    for problem in problem_list:
        if plot_time:
            # Load this problems time data
            path = os.path.join(OUTPUT, problem, "problem_size_time.csv")
            if not os.path.exists(path):
                print("Specified path " + path + " does not exist. Skipping.")
                continue
            time_frame = pd.read_csv(path)
            plot_problem_size_time(time_frame, problem)


        if plot_iterations:
            path = os.path.join(OUTPUT, problem, "fit_per_iteration.csv")
            if not os.path.exists(path):
                print("Specified path " + path + " does not exist. Skipping.")
                continue
            iteration_frame = pd.read_csv(path)
            plot_iteration_fit(iteration_frame, problem)

        if plot_iterations:
            path = os.path.join(OUTPUT, problem, "problem_size_fit.csv")
            if not os.path.exists(path):
                print("Specified path " + path + " does not exist. Skipping.")
                continue
            iteration_frame = pd.read_csv(path)
            plot_problem_size_scores(iteration_frame, problem)

if __name__ == '__main__':
    problems = ['FlipFlop', 'CPeaks', 'Knapsack', 'TSP']
    plot_toy_problems(problems)