import mlrose
import numpy as np
import argparse
import logging
import os
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

def run_nn_gridsearch(initial_model, params, algname, dataset, data_dict, outdir='./output/Neural/'):
    x_train = data_dict['xtrain']
    y_train = data_dict['ytrain']
    x_test = data_dict['xtest']
    y_test = data_dict['ytest']
    clf = GridSearchCV(initial_model, params, cv=3, scoring='accuracy', n_jobs=3, verbose=3, refit=True)
    clf.fit(x_train, y_train)
    reg_table = pd.DataFrame(clf.cv_results_)
    grid_table = pd.DataFrame(clf.grid_scores_)
    reg_table.to_csv('{}/{}_{}_reg.csv'.format(outdir, algname, dataset), index=False)
    grid_table.to_csv('{}/{}_{}_grid.csv'.format(outdir, algname, dataset), index=False)

    y_test_pred = clf.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)

    logger.info(dataset + " :: " + algname + " :: highest score: " + str(y_test_accuracy))
    logger.info(algname + " :: " + dataset + " :: " + " Best params: " + str(clf.best_params_))
    logger.info(algname + " :: " + dataset + " :: " + " Best score: " + str(clf.best_score_))
    confusemat = confusion_matrix(y_test.argmax(1), y_test_pred.argmax(1))
    confusemat = confusemat.astype('float') / confusemat.sum(axis=1)[:, np.newaxis]
    heatmap = sns.heatmap(confusemat, annot=True)
    heatmap.set(xlabel='Prediction', ylabel='Actual')
    plot = heatmap.get_figure()
    plt.savefig('{}/images/{}_{}_CM.png'.format(outdir, algname, dataset), format='png', dpi=150,
                bbox_inches='tight')
    plt.close()
    return



def run_nn_experiment():
    seed = np.random.randint(0, (2 ** 32) - 1)
    logger.info("Using seed {}".format(seed))
    np.random.seed(seed)
    rand.seed(seed)

    OUTPUT_DIRECTORY = './output/Neural/'

    if not os.path.exists('./output/Neural/'):
        os.mkdir('./output/Neural/')
    if not os.path.exists('./output/Neural/images'):
        os.mkdir('./output/Neural/images')

    
    logger.info("Loading data")
    logger.info("----------")

    datasets = [
    {
        'data': loader.SkyServerData(verbose=True, seed=seed),
        'name': 'SkyServer',
        'readable_name': 'SkyServer',
    },
    # {
    #     'data': loader.AusWeather(verbose=verbose, seed=seed),
    #     'name': 'AusWeather',
    #     'readable_name': 'AusWeather',
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
        iterations =  np.logspace(0, 4, 20).astype(int) * 2


        rhc_params = [ {'max_iters': iterations}]
                    #    {'max_attempts': np.arange(50, 400, 200)} ]

        rhcmodel = mlrose.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid', 
                                        algorithm ='random_hill_climb', 
                                        max_iters = 4000, bias = False, is_classifier = True, 
                                        lr=0.001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 50)

        run_nn_gridsearch(rhcmodel, rhc_params, 'random_hill_climb', ds['name'], data_dict)



        sa_params = [{'max_iters': iterations}]
                    # {'lr': np.arange(.00001, .0001, .0005) } ]
        samodel = mlrose.NeuralNetwork(hidden_nodes = [3*n_features//2],
                                        activation ='sigmoid', 
                                        algorithm ='simulated_annealing', 
                                        max_iters = 4000, bias = False, is_classifier = True, 
                                        lr=0.0001, early_stopping = True, schedule=mlrose.GeomDecay(15., .99, .1),
                                        clip_max=5, max_attempts = 25)

        run_nn_gridsearch(samodel, sa_params, 'simulated_anneal', ds['name'], data_dict)


        genetic_params = [ {'max_iters': iterations}]
                        #    {'mutation_prob': np.arange(.001, .1, .01).astype(float)},
                        #    {'pop_size': np.arange(50, 300, 200).astype(int)} ]
        gamodel = mlrose.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid', 
                                        algorithm ='genetic_alg', 
                                        max_iters = 100, bias = False, is_classifier = True, 
                                        lr=0.0001, early_stopping = True, mutation_prob=.025,
                                        clip_max=5, max_attempts = 50, pop_size=1000)

        run_nn_gridsearch(gamodel, genetic_params, 'genetic_alg', ds['name'], data_dict)



        gd_params = [{'max_iters': iterations}]
                    #  {'lr': np.arange(.00001, .0001, .0005) } ]
        gdmodel = mlrose.NeuralNetwork(hidden_nodes = [3*n_features//2], activation ='sigmoid', 
                                        algorithm ='gradient_descent', 
                                        max_iters = 3000, bias = False, is_classifier = True, 
                                        lr=0.0002, early_stopping = True, 
                                        clip_max=5, max_attempts = 25, pop_size=100)

        run_nn_gridsearch(gdmodel, gd_params, 'gradient_descent', ds['name'], data_dict)


if __name__ == '__main__':
    run_nn_experiment()