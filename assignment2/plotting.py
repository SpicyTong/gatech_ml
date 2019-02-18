import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

import os

sns.set_style("darkgrid")


def plot_simulated_annealing(problem, alg, sweep_dict={}):
    """
    For a given problem, plot the scores and timing information in the dictionary.

    The sweep dictionary should be pairs of <parameter name>: (x, y) array 
    for a parameter sweep of values x and resulting values y.
    """
    outputdir = 'output/' + problem + '/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    prefix = problem + '_' + alg + '_'

    for param in sweep_dict.keys():
        plt.figure()
        plt.title(prefix + param)
        means = sweep_dict[param].mean(axis=1)
        devs = sweep_dict[param].std(axis=1)
        axis = sns.lineplot(data=means, ci='sd', err_style="band")
        plt.fill_between(sweep_dict[param].index, means + devs, means - devs, color='red', alpha=.2)
        plt.xlabel(param)
        axis.figure.savefig(outputdir + prefix + param + '.png', dpi=150)



