import mlrose
import numpy as np
import argparse
import logging
import random as rand
from datetime import datetime
from data import loader
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = './output/Neural/'



def run_nn_gridsearch(initial_model, params, algname, dataset, data_dict):
    x_train = data_dict['xtrain']
    y_train = data_dict['ytrain']
    x_test = data_dict['xtest']
    y_test = data_dict['ytest']
    clf = GridSearchCV(initial_model, params, cv=5, scoring='accuracy', n_jobs=3, verbose=10, refit=True)
    clf.fit(x_train, y_train)
    reg_table = pd.DataFrame(clf.cv_results_)
    grid_table = pd.DataFrame(clf.grid_scores_)
    reg_table.to_csv('{}/{}_{}_reg.csv'.format(OUTPUT_DIRECTORY, algname, dataset), index=False)
    grid_table.to_csv('{}/{}_{}_grid.csv'.format(OUTPUT_DIRECTORY, algname, dataset), index=False)

    y_test_pred = clf.predict(x_test)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    logger.info(dataset + " :: " + algname + " :: highest score: " + str(y_test_accuracy))
    logger.info(algname + " :: " + dataset + " :: " + " Best params: " + str(clf.best_params_))
    logger.info(algname + " :: " + dataset + " :: " + " Best score: " + str(clf.best_score_))
    confusemat = confusion_matrix(y_test.argmax(1), y_test_pred.argmax(1))
    heatmap = sns.heatmap(confusemat, annot=True)
    heatmap.set(xlabel='Prediction', ylabel='Actual')
    plot = heatmap.get_figure()
    plt.savefig('{}/images/{}_{}_CM.png'.format(OUTPUT_DIRECTORY, algname, dataset), format='png', dpi=150,
                bbox_inches='tight')
    plt.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for Randomized Optimization')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--dump_data', action='store_true', help='Build train/test/validate splits '
                                                                 'and save to the data folder '
                                                                 '(should only need to be done once)')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    
    logger.info("Loading data")
    logger.info("----------")

    datasets = [
    # {
    #     'data': loader.StatlogVehicleData(verbose=verbose, seed=seed),
    #     'name': 'statlog_vehicle',
    #     'readable_name': 'Statlog Vehicle',
    # },
    # {
    #     'data': loader.HTRU2Data(verbose=verbose, seed=seed),
    #     'name': 'htru2',
    #     'readable_name': 'HTRU2',
    # },
    {
        'data': loader.SkyServerData(verbose=verbose, seed=seed),
        'name': 'SkyServer',
        'readable_name': 'SkyServer',
    },
    # {
    #     'data': loader.AusWeather(verbose=verbose, seed=seed),
    #     'name': 'AusWeather',
    #     'readable_name': 'AusWeather',
    # }
    # {
    #     'data': loader.SpamData(verbose=verbose, seed=seed),
    #     'name': 'spam',
    #     'readable_name': 'Spam',
    # },
    # {
    #     'data': loader.CreditDefaultData(verbose=verbose, seed=seed),
    #     'name': 'credit_default',
    #     'readable_name': 'Credit Default',
    # }
    ]




    for ds in datasets:
        logger.info('Processing dataset ' + ds['name'])
        data = ds['data']
        data.load_and_process()
        data._data.pop(data._data.columns[-1])
        # Generate test/train splits.
        X_train, X_test, y_train, y_test = train_test_split(data._data, data.classes, test_size = 0.3)
        n_features = X_train.shape[1]
        print("Total dataset size: " + str(len(data._data)))
        print("Total test size: " + str(len(X_test)))
        print("Number of features: " + str(n_features))

        scaler = MinMaxScaler()

        x_train_scaled = scaler.fit_transform(X_train)
        x_test_scaled = scaler.transform(X_test)

        one_hot = OneHotEncoder()

        y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

        data_dict = {'xtrain': x_train_scaled, 'ytrain': y_train_hot, 'xtest': x_test_scaled, 'ytest': y_test_hot}
        # TODO match pipeline from before
        iterations =  np.logspace(0, 4, 3).astype(int)

        rhc_params = [ {'max_iters': iterations}]

        rhcmodel = mlrose.NeuralNetwork(hidden_nodes = [n_features, n_features], activation ='relu', 
                                        algorithm ='random_hill_climb', 
                                        max_iters = 4000, bias = False, is_classifier = True, 
                                        lr=0.005, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100)

        run_nn_gridsearch(rhcmodel, rhc_params, 'random_hill_climb', ds['name'], data_dict)



        sa_params = {'max_iters': iterations}
        samodel = mlrose.NeuralNetwork(hidden_nodes = [n_features, n_features], activation ='relu', 
                                        algorithm ='simulated_annealing', 
                                        max_iters = 4000, bias = False, is_classifier = True, 
                                        lr=0.005, early_stopping = True, 
                                        clip_max = 5, max_attempts = 300)

        run_nn_gridsearch(samodel, sa_params, 'simulated_anneal', ds['name'], data_dict)


        genetic_params = [ {'max_iters': iterations}]
        gamodel = mlrose.NeuralNetwork(hidden_nodes = [n_features], activation ='relu', 
                                        algorithm ='genetic_alg', 
                                        max_iters = 3000, bias = False, is_classifier = True, 
                                        lr=0.005, early_stopping = True,  mutation_prob=.05,
                                        clip_max = 5, max_attempts = 100, pop_size=300)

        run_nn_gridsearch(gamodel, genetic_params, 'genetic_alg', ds['name'], data_dict)



        gd_params = [{'max_iters': iterations}]
        gdmodel = mlrose.NeuralNetwork(hidden_nodes = [n_features], activation ='relu', 
                                        algorithm ='gradient_descent', 
                                        max_iters = 3000, bias = False, is_classifier = True, 
                                        lr=0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100, pop_size=300)

        run_nn_gridsearch(gdmodel, gd_params, 'gradient_descent', ds['name'], data_dict)