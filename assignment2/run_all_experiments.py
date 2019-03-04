import os

from neuralnet import run_nn_experiment
from plot_nn_results import plot_all_nn_results

from cpeaks import run_cpeaks
from flipflop import run_flipflop
from knapsack import run_knapsack

from plot_toy_results import plot_toy_problems


if __name__ == '__main__':
    # Run the neural net stuff
    if not os.path.exists('./output/'):
        os.mkdir('./output/')
    
    run_nn_experiment()
    # Plot NN
    plot_all_nn_results()

    # Run toy problems
    run_cpeaks()
    run_flipflop()
    run_knapsack()

    # Plot toy problems
    problems = ['FlipFlop', 'CPeaks', 'Knapsack']
    plot_toy_problems(problems)
