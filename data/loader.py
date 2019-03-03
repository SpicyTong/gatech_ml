import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import seaborn as sns

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
    return H/np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()
        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.2):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.3, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(self.features, self.classes,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           stratify=self.classes)
        pipe = Pipeline([('Scale', preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
                                                                       test_size=test_size, random_state=random_state,
                                                                       stratify=train_y)
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv('data/{}_test.csv'.format(self.data_name()), index=False, header=False)
        trg.to_csv('data/{}_train.csv'.format(self.data_name()), index=False, header=False)
        val.to_csv('data/{}_validate.csv'.format(self.data_name()), index=False, header=False)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class CreditDefaultData(DataLoader):

    def __init__(self, path='data/default of credit card clients.xls', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_excel(self._path, header=1, index_col=0)

    def data_name(self):
        return 'CreditDefaultData'

    def class_column_name(self):
        return 'default payment next month'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes

class SteelPlateData(DataLoader):

    def __init__(self, path='data/Faults.NAA', verbose=False, seed=1, binarize=False):
        super().__init__(path, verbose, seed)
        self.binarize = binarize

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None, delimiter=r"\s+")


    def data_name(self):
        return 'SteelPlateData'

    def class_column_name(self):
        return '33'

    def _preprocess_data(self):
        class_cols = [27, 28, 29, 30, 31, 32, 33]
        # Collapse to integer encoded data
        self.classes = np.where(self._data[class_cols] == 1)[1]
        if self.binarize:
            known_items = self.classes < 6
            unknown_items = self.classes == 6
            self.classes[known_items] = 0
            self.classes[unknown_items] = 1
        self._data.drop(self._data.columns[27:34], axis=1, inplace=True)
        self._data['33'] = self.classes
        self._data.columns = self._data.columns.astype(str)

    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes


class AusWeather(DataLoader):

    def __init__(self, path='data/weatherAUS.csv', verbose=False, seed=1, binarize=True):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path)


    def data_name(self):
        return 'AusWeather'

    def class_column_name(self):
        return 'RainTomorrow'

    def _preprocess_data(self):
        # Select on location
        print(self._data.Location.unique())
        keep_locations = ['Sydney', 'SydneyAirport', 'Brisbane',
                          'Melbourne', 'MelbourneAirport', 'Canberra']
        new_sets = []
        for loc in keep_locations:
            new_sets.append(self._data[self._data['Location'] == loc])

        self._data = pd.concat(new_sets)
        self._data.pop('Location')

        # Convert dates
        # We want to convert dates to a year, month, and day feature column. Could be useful (more ran in fall, or 2008, etc.)
        dates = pd.to_datetime(self._data.Date)
        year = [d.day for d in dates]
        month = [d.month for d in dates]
        day = [d.day for d in dates]
        self._data.pop('Date')
        self._data['Year'] = year
        self._data['Month'] = month
        # self._data['Day'] = day

        # Convert cardinal directions
        directions_dict = {'E': 0., 'ENE': 22.5, 'NE': 45., 'NNE': 67.5, 'N': 90., 'NNW': 112.5, 'NW': 135., 'WNW': 157.5,
                          'W': 180., 'WSW': 202.5, 'SW': 225., 'SSW': 247.5, 'S': 270., 'SSE': 292.5, 'SE': 315., 'ESE': 337.5}
        self._data.WindDir9am = self._data.WindDir9am.map(directions_dict)
        self._data.WindDir3pm = self._data.WindDir3pm.map(directions_dict)

        # convert labels
        self._data.replace({'No':0, 'Yes':1}, inplace=True)
        # RainToday has some nans in it. replace with zeros.
        self._data['RainToday'].fillna(0, inplace=True)

        # Drop columns with lots of missing data
        self._data.pop('WindGustDir')
        self._data.pop('WindGustSpeed')
        self._data.pop('Cloud9am')
        self._data.pop('Cloud3pm')
        self._data.pop("Evaporation")

        # Drop attributes to make the data more interesting (maybe)
        # self._data.pop('Humidity9am')
        # self._data.pop('Humidity3pm')
        # self._data.pop('RainToday')
        # self._data.pop('WindDir9am')
        # self._data.pop('WindDir3pm')
        # self._data.pop("Pressure3pm")
        # self._data.pop("Pressure9am")
        # self._data.pop("WindSpeed9am")
        # self._data.pop("WindSpeed3pm")
        # self._data.pop("Sunshine")
        
        # Remove Risk_MM which can leak as per dataset description. Direct predictor of rain
        self._data.pop("RISK_MM")

        # Fill means for each column. Basic analysis shows not a ton of nans are left.
        # for col in self._data.keys():
        #     self._data[col].fillna((self._data[col].mean()), inplace=True)

        self._data.dropna(how='any', inplace=True)

        # Put labels at end
        classes = self._data.pop('RainTomorrow')
        self._data['RainTomorrow'] = classes
        self._data.reset_index()

        num_positive = self._data[self._data.RainTomorrow == 1].RainTomorrow.sum()
        df1 = self._data[self._data.RainTomorrow == 0].sample(num_positive, random_state=13)
        df2 = self._data[self._data.RainTomorrow == 1]

        self._data = pd.concat([df1, df2]).sample(frac=1)


    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes


class CreditApprovalData(DataLoader):

    def __init__(self, path='data/crx.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'CreditApprovalData'

    def class_column_name(self):
        return '12'

    def _preprocess_data(self):
        # https://www.ritchieng.com/machinelearning-one-hot-encoding/
        to_encode = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15]
        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        # https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, vec_data], axis=1)

        # Clean any ?'s from the unencoded columns
        self._data = self._data[( self._data[[1, 2, 7]] != '?').all(axis=1)]

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class PenDigitData(DataLoader):
    def __init__(self, path='data/pendigits.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def class_column_name(self):
        return '16'

    def data_name(self):
        return 'PendDigitData'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class AbaloneData(DataLoader):
    def __init__(self, path='data/abalone.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'AbaloneData'

    def class_column_name(self):
        return '8'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class HTRU2Data(DataLoader):
    def __init__(self, path='data/HTRU_2.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'HTRU2Data'

    def class_column_name(self):
        return '8'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class SpamData(DataLoader):
    def __init__(self, path='data/spambase.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'SpamData'

    def class_column_name(self):
        return '57'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class SkyServerData(DataLoader):
    def __init__(self, path='data/skyserver_sql2.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path)

    def data_name(self):
        return 'SkyServer'

    def class_column_name(self):
        return 'class'

    def _preprocess_data(self):
        # Convert class categories to codes.
        # https://stackoverflow.com/questions/30510562/get-mapping-of-categorical-variables-in-pandas
        self._data['class'] = self._data['class'].astype('category').cat.codes
        # Also push classes to the end.
        classes = self._data.pop('class')
        self._data['class'] = classes

        self._data.pop('objid')
        self._data.pop('specobjid')
        self._data.pop('rerun')
        self._data.pop('run')
        self._data.pop('camcol')


        num_fewest = self._data[self._data['class'] == 1]['class'].sum()
        num_to_take = int(np.round(num_fewest * 1.5))
        df1 = self._data[self._data['class'] == 0].sample(num_to_take, random_state=13)
        df2 = self._data[self._data['class'] == 2].sample(num_to_take, random_state=13)
        df3 = self._data[self._data['class'] == 1]

        self._data = pd.concat([df1, df2, df3]).sample(frac=1)


    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes



class StatlogVehicleData(DataLoader):
    def __init__(self, path='data/statlog.vehicle.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'StatlogVehicleData'

    def class_column_name(self):
        return '18'

    def _preprocess_data(self):
        to_encode = [18]
        label_encoder = preprocessing.LabelEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, df], axis=1)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


if __name__ == '__main__':
    cd_data = CreditDefaultData(verbose=True)
    cd_data.load_and_process()

    ca_data = CreditApprovalData(verbose=True)
    ca_data.load_and_process()
