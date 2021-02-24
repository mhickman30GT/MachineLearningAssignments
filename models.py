import copy
import json
import multiprocessing
import os
import time

from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree

from mlxtend import evaluate

import numpy as np
import pandas as pd

import plot

# GLOBAL VARIABLES
RANDOM_SEED = 14
CORE_COUNT_PERCENTAGE = .75  # NOTE: Any increase past this and the comp is unusable
ONE_HOT_ENCODING = True


class DataSet:
    """ Class holding values for dataset """

    def __init__(self, data_name, file, data_dict):
        """ Constructor for Dataset """
        self.data_name = data_name
        self.file = file
        self.config = data_dict
        self.csv = pd.read_csv(self.file)
        self.label = data_dict["label"]
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        self.y_predict = pd.Series()

    def process(self, label):
        """ Processes data set """
        # Separate classification labels
        x = self.csv.drop(label, 1)
        y = self.csv[label]

        # Default to one hot for all sets
        if ONE_HOT_ENCODING:
            x = pd.get_dummies(x, columns=x.select_dtypes(include=[object]).columns)

        # Split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
            x, y, stratify=y, test_size=0.25)

        # Scale data using standardizer
        standardize = preprocessing.StandardScaler()
        self.x_train = standardize.fit_transform(self.x_train)
        self.x_test = standardize.transform(self.x_test)


def generate_classes(data, model):
    """ Generate model classes for all models requested """
    models = list()
    # For each model, create class instance
    if model == 'DT':
        model_inst = SupervisedClassifier(model, tree.DecisionTreeClassifier, data)
        models.append(model_inst)
    elif model == 'NN':
        model_inst = SupervisedClassifier(model, neural_network.MLPClassifier, data)
        models.append(model_inst)
    elif model == "BOOST":
        model_inst = SupervisedClassifier(model, ensemble.AdaBoostClassifier, data)
        models.append(model_inst)
    elif model == 'SVM':
        model_inst = SupervisedClassifier(model, svm.SVC, data)
        models.append(model_inst)
    elif model == 'KNN':
        model_inst = SupervisedClassifier(model, neighbors.KNeighborsClassifier, data)
        models.append(model_inst)

    return models


class SupervisedClassifier:
    """ Class for all models """

    def __init__(self, name, model_inst, dataset):
        """Constructor for SupervisedClassifier"""
        self.name = name
        self.inst = model_inst
        self.data = dataset
        self.times = dict()
        self.output = list()
        self.grid_search = None
        self.multi_param = False
        self.title = self.get_algorithm(f'{self.name}')
        self.total_params = self.get_params()
        self.grid_results = dict()
        self.jobs = round(multiprocessing.cpu_count() * CORE_COUNT_PERCENTAGE)
        # The model is jenky. Some have seeds, some dont
        try:
            self.model = self.inst(random_state=RANDOM_SEED, **self.get_single_params())
        except TypeError:
            self.model = self.inst(**self.get_single_params())

    def __repr__(self):
        """Simple __repr__"""
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def __str__(self):
        """Simple __str___"""
        return self.__repr__()

    @staticmethod
    def get_algorithm(i):
        switcher = {
            'DT': 'Decision Tree Classifier',
            'BOOST': 'Boosting Classifier',
            'KNN': 'KNeighbors Classifier',
            'NN': 'Neural Networks Classifier',
            'SVM': 'SVM Classifier',
        }
        return switcher.get(i, "INVALID")

    def get_params(self):
        """ Find and process all params from config """
        return_params = dict()
        config_params = self.data.config["params"][self.name]
        for key, param in config_params.items():
            # Boosting has the stupid internal depth
            if self.name == "BOOST" and key == "max_depth":
                return_params["base_estimator"] = [
                    tree.DecisionTreeClassifier(max_depth=iter_val)
                    for iter_val in param["value"]
                ]
            else:
                return_params[key] = param["value"]
        return return_params

    def get_single_params(self):
        """ Finds the parameters that are single values in config """
        single_params = dict()
        for key, value in self.total_params.items():
            if len(value) == 1:
                single_params[key] = value[0]
        return single_params

    def update_params(self):
        """ Update model with all process parameters """
        parameters = dict()
        # Take the first value for all parameters
        for key, value in self.total_params.items():
            parameters[key] = value[0]
        # Update model
        self.model = self.inst(random_state=RANDOM_SEED, **parameters)

    def grid_search_cv(self):
        """ Create and run grid search over range """
        # Save time and print to indicate search start
        print(f"{self.name}: Running grid search")
        start_time = time.time()
        # Multiple parameters run
        self.multi_param = True

        # Create grid search
        self.grid_search = model_selection.GridSearchCV(
            self.model,
            param_grid=self.total_params,
            n_jobs=self.jobs,
        )

        # Run grid search
        self.grid_search.fit(self.data.x_train, self.data.y_train)

        # Save off results
        self.times["grid_search"] = round(time.time() - start_time, 2)
        self.grid_results = self.grid_search.best_params_
        self.model = self.grid_search.best_estimator_

    def learning_curve(self):
        """ Generate Learning Curve Data """
        # Generate LC Data
        train_sizes, train_scores, test_scores, fit_times, score_times = model_selection.learning_curve(
            self.model,
            self.data.x_train,
            self.data.y_train,
            n_jobs=self.jobs,
            return_times=True,
        )
        self.output.append(
            {
                "type": "learning_curve",
                "data": {
                    "train_sizes": train_sizes.tolist(),
                    "train_scores": train_scores.tolist(),
                    "test_scores": test_scores.tolist(),
                    "fit_times": fit_times.tolist(),
                    "score_times": score_times.tolist(),
                },
            }
        )

    def validation_curve(self, scoring="accuracy"):
        """ Generate Validation Curve Data """
        # Find all params with multiple values
        multi_params = dict()
        for key, value in self.total_params.items():
            if len(value) > 1:
                multi_params[key] = value

        # Loop through ranges of params
        for param, range in multi_params.items():
            # Generate VC data
            train_scores, test_scores = model_selection.validation_curve(
                self.model,
                self.data.x_train,
                self.data.y_train,
                param_name=param,
                param_range=range,
                scoring=scoring,
                n_jobs=self.jobs,
            )
            self.output.append(
                {
                    "type": "validation_curve",
                    "data": {
                        "parameter": param,
                        "range": range,
                        "train_scores": train_scores.tolist(),
                        "test_scores": test_scores.tolist(),
                    },
                }
            )
            print(f'{self.name}: Validation Curve Train Scores for {param} = {train_scores * 100}')
            print(f'{self.name}: Validation Curve Test Scores for {param} = {test_scores * 100}')

    def confusion_matrix(self):
        """ Generate Confusion matrix """
        confusion_matrix = metrics.confusion_matrix(
            self.data.y_test, self.grid_search.predict(self.data.x_test)
        )
        self.output.append(
            {
                "type": "confusion_matrix",
                "data": confusion_matrix.tolist(),
            }
        )

    def calculate_bias_variance(self):
        """ Calculate bias and variance """
        mse, bias, var = evaluate.bias_variance_decomp(
            self.model,
            np.array(self.data.x_train),
            np.array(self.data.y_train),
            np.array(self.data.x_test),
            np.array(self.data.y_test),
            loss="mse",
            num_rounds=200,
            random_seed=RANDOM_SEED,
        )
        self.output.append(
            {
                "type": "bias_variance",
                "data": {
                    "MSE": mse,
                    "BIAS": bias,
                    "VARIANCE": var,
                },
            }
        )
        print(f"{self.name}: Total Error (Means, Bias, Variance) = "
              f"({mse}, {bias}, {var})")
        breakpoint()

    def score(self, generate_plots=False):
        """ Generate Scores """
        # Accuracy score
        accuracy = metrics.accuracy_score(self.data.y_test, self.data.y_predict)
        self.output.append({"type": "accuracy", "data": accuracy})
        print(f"{self.name}: Accuracy = {accuracy * 100}")
        
        # F1 score
        f1 = metrics.f1_score(self.data.y_test, self.data.y_predict, average="weighted")
        self.output.append({"type": "f1", "data": f1})
        print(f"{self.name}: F1 = {f1}")
        
        # Cross Validation score
        cv_score = model_selection.cross_val_score(
            self.model, self.data.x_train, self.data.y_train, n_jobs=self.jobs
        )
        self.output.append({"type": "cross_validation_score", "data": cv_score.tolist()})
        print(f"{self.name}: Cross Validation Score (Mean, Median) = "
              f"({np.mean(cv_score)}, {np.median(cv_score)}) ")
        
        # Bias variance
        self.calculate_bias_variance()

        # Generate curves for plot generation
        if generate_plots:
            self.learning_curve()
            self.validation_curve()

        # Confusion Matrix (only if we've completed a grid search)
        if self.multi_param:
            self.confusion_matrix()

    def run(self, generate_plots=False):
        """ Run models """
        model_repr = str(self.model).replace("\n", "").replace(" ", "").replace(",", ", ")
        print(f'{self.name}: Running model: {model_repr}')

        # Run fit
        print(f"{self.name}: Running fit")
        start_time = time.time()
        self.model.fit(self.data.x_train, self.data.y_train)
        self.times["fit"] = round(time.time() - start_time, 2)

        # Run predict
        print(f"{self.name}: Running predict")
        start_time = time.time()
        self.data.y_predict = self.model.predict(self.data.x_test)
        self.times["predict"] = round(time.time() - start_time, 2)

        # Generate scores
        self.score(generate_plots)

    def output_dict(self, for_json=False):
        """ Convert output data to dict for output """
        # Save off all data
        output_dict = {
            "name": self.name,
            "file": self.data.file,
            "params": copy.deepcopy(self.total_params),
            "grid_results": copy.deepcopy(self.grid_results),
            "times": self.times,
            "scores": self.output,
        }

        # NOTE: JSON CODE BELOW CREDIT GOES TO MY ROOMMATE
        # Was originally using text outputs... ty ty
        if for_json:
            # Check if JSON serializable as is
            try:
                json.dumps(output_dict, indent=4)
            # If auto-generation doesn't work, convert any stubborn fields
            # to dicts/strings manually
            except TypeError:
                for key, value in output_dict.items():
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            try:
                                json.dumps(nested_value)
                            except TypeError:
                                if hasattr(nested_value, "__dict__"):
                                    output_dict[key][nested_key] = nested_value.__dict__
                                elif hasattr(nested_value, "__len__") and hasattr(
                                        nested_value[0], "__dict__"
                                ):
                                    output_dict[key][nested_key] = [
                                        iter_value.__dict__ for iter_value in nested_value
                                    ]
                                else:
                                    output_dict[key][nested_key] = str(nested_value)
                    else:
                        try:
                            json.dumps(value)
                        except TypeError:
                            output_dict[key] = str(value)
        return output_dict

    def to_json(self, filename):
        """Write to JSON file

        Args:
            filename (str): Path to JSON file to write
        """
        # Write to file
        with open(filename, "w") as open_file:
            open_file.write(json.dumps(self.output_dict(for_json=True), indent=4))

    def plot(self, dirname):
        """ Generate plots """
        # Plot learning curve
        print(f"{self.name}: Plotting learning curve")
        learning_curve = [x for x in self.output if x["type"] == "learning_curve"]
        if learning_curve:
            fig = plot.plot_learning_curve(
                self.title,
                learning_curve[0]["data"]["train_sizes"],
                learning_curve[0]["data"]["train_scores"],
                learning_curve[0]["data"]["test_scores"],
                learning_curve[0]["data"]["fit_times"],
            )
            fig.savefig(os.path.join(dirname, f"{self.name}_learning_curve.png"))

        # Plot validation curve(s)
        valdiation_curve = [x for x in self.output if x["type"] == "validation_curve"]
        if valdiation_curve:
            for param in valdiation_curve:
                fig = plot.plot_validation_curve(
                    self.title,
                    param["data"]["parameter"],
                    param["data"]["range"],
                    param["data"]["train_scores"],
                    param["data"]["test_scores"],
                )
                fig.savefig(
                    os.path.join(
                        dirname,
                        f'{self.name}_{param["data"]["parameter"]}_validation_curve.png',
                    )
                )

        # If NN plot loss curve
        if self.title == "Neural Networks Classifier":
            fig = plot.plot_loss_curve(self.model)
            fig.savefig(
                os.path.join(
                    dirname,
                    f'{self.name}_loss_curve.png',
                )
            )

        # Plot confusion matrix
        confusion = [x for x in self.output if x["type"] == "confusion_matrix"]
        if confusion:
            fig = plot.plot_confusion_matrix(self.model, self.data.x_test, self.data.y_test)
            fig.savefig(os.path.join(dirname, f"{self.name}_confusion_matrix.png"))
