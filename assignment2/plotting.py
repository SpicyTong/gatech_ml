import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os

sns.set_style("darkgrid")

def plot_montecarlo_sensitivity(problem, alg, sweep_dict={}):
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
        plt.close()

    # dataframe = pd.DataFrame(sweep_dict)
    # dataframe.to_csv(outputdir + prefix + ".csv")



def plot_complexity(data, complexity_param, plot_prefix, pretty_name=None, plot_time=True, plot_score=True):
    x_values = data[complexity_param]
    if pretty_name is None:
        pretty_name = "Complexity analysis"

    if plot_time:
        fit_time_mean = data['mean_fit_time']
        fit_time_std = data['std_fit_time']
        axis = sns.lineplot(x=x_values, y='mean_fit_time', data=data, ci='sd', err_style="band", label='Fit time (s)')
        plt.fill_between(x_values, fit_time_mean + fit_time_std, fit_time_mean - fit_time_std, color='purple', alpha=.2)
        plt.xlabel(complexity_param)
        plt.ylabel("Training run-time (s)")
        plt.title(pretty_name + ": " + complexity_param + " Timing Analysis")
        axis.figure.savefig(plot_prefix + '_time_' + complexity_param + '.png', dpi=150)

        plt.close()

    if plot_score:
        mean_test = data['mean_test_score']
        std_test = data['std_test_score']
        mean_train = data['mean_train_score']
        std_train = data['std_train_score']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x_values, mean_test, color='b', label='Test Score')
        ax1.fill_between(x_values, mean_test + std_test, mean_test - std_test, color='blue', alpha=.2)
        ax1.set_xscale('log')
        plt.title(pretty_name + ": " + complexity_param + " acccuracy")
        plt.xlabel(complexity_param)
        plt.ylabel("Classification Score")

        # ax2 = ax1.twinx()
        ax1.plot(x_values, mean_train, color='k', label='Train Score')
        ax1.fill_between(x_values, mean_train + std_train, mean_train - std_train, color='grey', alpha=.2)
        plt.xlabel(complexity_param)
        ax1.legend()
        # ax2.legend()
        ax1.figure.savefig(plot_prefix + '_score_' + complexity_param + '.png', dpi=150)
        

        plt.close()


def plot_problem_size_scores(problem_dataframe, problem_name, output_dir='./output'):
    # We have a line for each of the algorithms. Use seaborn to draw them super easily.
    if 'Unnamed: 0' in problem_dataframe.keys():
        problem_dataframe.pop('Unnamed: 0')

    problem_dataframe = problem_dataframe.set_index('Problem Size')

    axis = sns.lineplot(data=problem_dataframe)
    axis.set_ylabel('Fit')
    axis.legend(loc='lower right')
    axis.set_xlabel('Problem size')
    axis.set_title('Achievable fit score vs problem size: ' + problem_name)
    axis.figure.savefig(os.path.join(output_dir, problem_name, 'fit_vs_size.png'), dpi=300)
    plt.close()


def plot_problem_size_time(problem_dataframe, problem_name, output_dir='./output'):

    if 'Unnamed: 0' in problem_dataframe.keys():
        problem_dataframe.pop('Unnamed: 0')

    problem_dataframe = problem_dataframe.set_index('Problem Size')

    axis = sns.lineplot(data=problem_dataframe)
    axis.legend(loc='lower right')
    axis.set_yscale('log')
    axis.set_ylabel('Time (log seconds)')
    axis.set_xlabel('Problem size')
    axis.set_title('Problem size vs optimization time on ' + problem_name)
    axis.figure.savefig(os.path.join(output_dir, problem_name, 'time_vs_size.png'), dpi=300)
    plt.close()

    

def plot_iteration_fit(problem_dataframe, problem_name, output_dir='./output'):

    if 'Unnamed: 0' in problem_dataframe.keys():
        problem_dataframe.pop('Unnamed: 0')

    axis = sns.lineplot(data=problem_dataframe)
    axis.set_xscale('log')
    axis.legend(loc='lower right')
    axis.set_ylabel('Fit')
    axis.set_xlim(1, 1e5)
    axis.set_xlabel('Iteration') 
    axis.set_title('Fit vs Iteration For All Optimizers on ' + problem_name)
    path = os.path.join(output_dir, problem_name, 'fit_vs_iter.png')       
    axis.figure.savefig(path, dpi=300)
    plt.close()


def plot_neural_net_analysis(dataframe, dataset_name, output_dir='./output'):
    # Make a plot of iterations vs score for each.

    # Collect all values from a bunch of dataframes that have param_max_iter in them. 
    # Make a new dataframe with a column for each alg and iterations as the index. (need to match?)
    axis = sns.lineplot(data=dataframe)
    # axis.set_xscale('log')
    axis.legend(loc='lower right')
    axis.set_xscale('log')
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Maximum Allowed iterations') 
    axis.set_title('Accuracy vs Number of Allowed Iterations:' + dataset_name)
    path = os.path.join(output_dir, "images", 'accuracy_vs_iter.png')       
    axis.figure.savefig(path, dpi=300)
    plt.close()