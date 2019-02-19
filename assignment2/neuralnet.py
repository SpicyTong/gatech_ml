import mlrose
import numpy as np
import argparse
import logging
import random as rand
from datetime import datetime
from data import loader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    # {
    #     'data': loader.SkyServerData(verbose=verbose, seed=seed),
    #     'name': 'SkyServer',
    #     'readable_name': 'SkyServer',
    # },
    {
        'data': loader.AusWeather(verbose=verbose, seed=seed),
        'name': 'AusWeather',
        'readable_name': 'AusWeather',
    }
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
        data = ds['data']
        data.load_and_process()
        # Generate test/train splits.
        X_train, X_test, y_train, y_test = train_test_split(data._data, data.classes, test_size = 0.3)

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        one_hot = OneHotEncoder()

        y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
        # TODO match pipeline from before

        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                        algorithm ='random_hill_climb', 
                                        max_iters = 2000, bias = True, is_classifier = True, 
                                        learning_rate = 0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 1000)

        nn_model1.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model1.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        logger.info(ds['name'] + ": RHC accuracy --> " + str(y_test_accuracy))


        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                        algorithm ='simulated_annealing', 
                                        max_iters = 1000, bias = True, is_classifier = True, 
                                        learning_rate = 0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100)

        nn_model1.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model1.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        logger.info(ds['name'] + ": SA accuracy --> " + str(y_test_accuracy))

        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                        algorithm ='genetic_alg', 
                                        max_iters = 1000, bias = True, is_classifier = True, 
                                        learning_rate = 0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100)

        nn_model1.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model1.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        logger.info(ds['name'] + ": GA accuracy --> " + str(y_test_accuracy))

        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                        algorithm ='gradient_descent', 
                                        max_iters = 1000, bias = True, is_classifier = True, 
                                        learning_rate = 0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100)

        nn_model1.fit(X_train_scaled, y_train_hot)
        y_test_pred = nn_model1.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        logger.info(ds['name'] + ": GD accuracy --> " + str(y_test_accuracy))