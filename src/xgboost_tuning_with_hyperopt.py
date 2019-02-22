# -*- coding: utf-8 -*-

# From: https://www.kaggle.com/yassinealouini/hyperopt-the-xgboost-model
import logging
from sklearn import metrics
from sklearn.metrics import classification_report
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
random_state = 777
np.random.seed(random_state)
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import seaborn as sns


class XGBClassifierTuning():
    """
    Prototype a fast tuned xgboost with hyperopt.
    We try to find the best set of xgboost's hyperparameters that optmizes the 
    [metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.metrics.average_precision_score.html) 
    per default. You can change the optimized metric with the parameter `optmized_metric`.

    XGBClassifierTuning contains an helper to create a mini-report of classification performance 
    (confusion matrix, classification report with precision, recall and f1-score, and proability distribution for each class).

    :param X_train: The training input samples.
    :param y_train: The target values for training set.
    :param X_val: The validation input samples.
    :param y_val: The target values for validation set.
    :param eval_metric: evaluation metrics for validation data, a default metric will be assigned according to objective. 
    Must be an [eval_metric](https://xgboost.readthedocs.io/en/latest/parameter.html) from xgboost.
    :param optmized_metric: should be a [sklearn metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) 
    returning a number (ex: accuracy_score, metrics.average_precision_score, roc_auc_score, etc).
    :param max_evals: try max_evals parameters combination for evaluation before returning best hyerparameters combination.
    :param weight_imbalanced: 
    :param verbose: if `True` xgboost is verbose.
    :param random_state: seed used by the random number generator for reproductibility.
    """

    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 eval_metric="auc",
                 optimized_metric=metrics.average_precision_score,
                 max_evals=10,
                 weight_imbalanced=False,
                 verbose=False,
                 random_state=777):

        self.logger = logging.getLogger(__name__)
        self.weight_imbalanced = weight_imbalanced
        self.optimized_metric = optimized_metric
        self.max_evals = max_evals
        self.verbose = verbose
        self.random_state = random_state
        self.best_hyperparams = None
        cores_number = np.min([8, os.cpu_count()])
        self.logger.info("Use " + str(cores_number) + " cores on " +
                         str(os.cpu_count()) + ".")

        if self.weight_imbalanced:
            scale_pos_weight = np.min([
                9, np.sum(1 - y_train) / np.sum(y_train)
            ])  # https://github.com/dmlc/xgboost/issues/2428
        else:
            scale_pos_weight = 1

        self.fix_hyperparameters = {
            'eval_metric': eval_metric,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': not self.verbose,
            'seed': self.random_state,
            'scale_pos_weight': scale_pos_weight,
            'max_delta_step': 1,
            'nthread': 0  #cores_number
        }
        self.fit_with_best_parameters(X_train, y_train, X_val, y_val)

    def fit_with_best_parameters(self, X_train, y_train, X_val, y_val):
        self.logger.info("Fit with best parameters")
        if self.best_hyperparams is None:
            self.optimize(X_train, y_train, X_val, y_val)
        model = xgb.XGBClassifier(**self.best_hyperparams)
        model.fit(X_train, y_train, verbose=self.verbose)
        self.model = model

    def __get_best_hyperparameters(self, best_hyperparams):
        hyperparams = dict(self.fix_hyperparameters, **best_hyperparams)
        hyperparams['n_estimators'] = int(hyperparams['n_estimators'])
        return hyperparams

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        This is the optimization function that given a space (space here) of 
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """

        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

        def objective_function(params):

            params['n_estimators'] = int(params['n_estimators'])

            self.logger.info("Training with params: ")
            self.logger.info(params)

            watchlist = [(X_train, y_train), (X_val, y_val)]

            model = xgb.XGBClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=watchlist,
                verbose=self.verbose,
                early_stopping_rounds=100)

            predictions = model.predict_proba(
                X_val, ntree_limit=model.best_iteration + 1)[:, 1]

            valid_score = self.optimized_metric(y_val, predictions)

            self.logger.info("\tScore {0}\n\n".format(valid_score))
            self.logger.info("-------- End Iteration --------")

            # The score function should return the loss (1-score)
            # since the optimize function looks for the minimum
            loss = 1 - valid_score

            return {'loss': loss, 'status': STATUS_OK}

        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
            'eta': hp.quniform('eta', 0.01, 0.4, 0.02),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.1, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'lambda': hp.quniform('lambda', 0.1, 1, 0.2),
        }
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best_hyperparams = self.__get_best_hyperparameters(
            fmin(
                objective_function,
                dict(self.fix_hyperparameters, **space),
                algo=tpe.suggest,
                max_evals=self.max_evals))
        self.best_hyperparams = best_hyperparams
        self.logger.info("Best parameters are:\n")
        self.logger.info(best_hyperparams)

    def report(self, X, y):
        """
        Helper to create a mini-report of classification performance (confusion matrix, 
        classification report with precision, recall and f1-score, and proability distribution for each class).
        :param X: features matrix (in general from test set).
        :param y: target vector (in general from test set)
        """
        pred = self.model.predict(X)
        skplt.estimators.plot_feature_importances(
            self.model, x_tick_rotation=90)
        plt.show()
        skplt.metrics.plot_confusion_matrix(y, pred, normalize=True)
        plt.show()
        print("Classification report")
        print(classification_report(y, pred))
        skplt.metrics.plot_precision_recall(y, self.model.predict_proba(X))
        plt.show()
        pd.DataFrame({
            "predict_probability": self.model.predict_proba(X)[:, 1],
            "observed": y
        }).hist(
            "predict_probability", by="observed")
        plt.suptitle("Probability distribution by class")
        plt.show()


class XGBRegressorTuning():
    """
    Prototype a fast tuned xgboost with hyperopt.
    We try to find the best set of xgboost's hyperparameters that optimizes the 
    [metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error) 
    per default. You can change the optimized metric with the parameter `optimized_metric`.

    XGBRegressorTuning contains an helper to create a mini-report of regressioon performance.

    :param X_train: The training input samples.
    :param y_train: The target values for training set.
    :param X_val: The validation input samples.
    :param y_val: The target values for validation set.
    :param eval_metric: evaluation metrics for validation data, a default metric will be assigned according to objective. 
    Must be an [eval_metric](https://xgboost.readthedocs.io/en/latest/parameter.html) from xgboost.
    :param optimized_metric: should be a [sklearn metric](https://scikit-learn.org/stable/modules/classes.html#regression-metrics).
    :param max_evals: try max_evals parameters combination for evaluation before returning best hyerparameters combination.
    :param verbose: if `True` xgboost is verbose.
    :param random_state: seed used by the random number generator for reproductibility.
    """

    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 eval_metric="rmse",
                 optimized_metric=metrics.mean_squared_error,
                 max_evals=10,
                 verbose=False,
                 random_state=777):

        self.logger = logging.getLogger(__name__)
        self.optimized_metric = optimized_metric
        self.max_evals = max_evals
        self.verbose = verbose
        self.random_state = random_state
        self.best_hyperparams = None
        cores_number = np.min([8, os.cpu_count()])
        self.logger.info("Use " + str(cores_number) + " cores on " +
                         str(os.cpu_count()) + ".")

        self.fix_hyperparameters = {
            'eval_metric': eval_metric,
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': not self.verbose,
            'seed': self.random_state,
            'max_delta_step': 1,
            'nthread': 0  #cores_number
        }
        self.fit_with_best_parameters(X_train, y_train, X_val, y_val)

    def fit_with_best_parameters(self, X_train, y_train, X_val, y_val):
        self.logger.info("Fit with best parameters")
        if self.best_hyperparams is None:
            self.optimize(X_train, y_train, X_val, y_val)
        model = xgb.XGBRegressor(**self.best_hyperparams)
        model.fit(X_train, y_train, verbose=self.verbose)
        self.model = model

    def __get_best_hyperparameters(self, best_hyperparams):
        hyperparams = dict(self.fix_hyperparameters, **best_hyperparams)
        hyperparams['n_estimators'] = int(hyperparams['n_estimators'])
        return hyperparams

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        This is the optimization function that given a space (space here) of 
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """

        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

        def objective_function(params):

            params['n_estimators'] = int(params['n_estimators'])

            self.logger.info("Training with params: ")
            self.logger.info(params)

            watchlist = [(X_train, y_train), (X_val, y_val)]

            model = xgb.XGBRegressor(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=watchlist,
                verbose=self.verbose,
                early_stopping_rounds=100)

            predictions = model.predict(
                X_val, ntree_limit=model.best_iteration + 1)

            valid_score = self.optimized_metric(y_val, predictions)

            self.logger.info("\tScore {0}\n\n".format(valid_score))
            self.logger.info("-------- End Iteration --------")

            return {'loss': valid_score, 'status': STATUS_OK}

        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
            'eta': hp.quniform('eta', 0.01, 0.4, 0.02),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.1, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'lambda': hp.quniform('lambda', 0.1, 1, 0.2),
        }
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best_hyperparams = self.__get_best_hyperparameters(
            fmin(
                objective_function,
                dict(self.fix_hyperparameters, **space),
                algo=tpe.suggest,
                max_evals=self.max_evals))
        self.best_hyperparams = best_hyperparams
        self.logger.info("Best parameters are:\n")
        self.logger.info(best_hyperparams)

    def report(self, X, y):
        """
        Helper to create a mini-report of classification performance (confusion matrix, 
        classification report with precision, recall and f1-score, and proability distribution for each class).
        :param X: features matrix (in general from test set).
        :param y: target vector (in general from test set)
        """
        predict = self.model.predict(X)

        skplt.estimators.plot_feature_importances(
            self.model, x_tick_rotation=90)
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(x=y, y=predict)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predict")
        ax.set_title("Predict vs. Observed")
        plt.show()

        residuals = y - predict

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(x=y, y=residuals)
        plt.title("Residuals vs. Observed")
        plt.xlabel("Obserbed")
        plt.ylabel("Residuals")
        plt.show()

        plt.hist(residuals)
        plt.title("Residuals distribution")
        plt.xlabel("Residuals value")
        plt.ylabel("Count")
        plt.show()

        display(
            pd.DataFrame({
                "explained_variance_score":
                metrics.explained_variance_score(y, predict),
                "mean_absolute_error":
                metrics.mean_absolute_error(y, predict),
                "mean_squared_log_error":
                metrics.mean_squared_log_error(y, predict),
                "median_absolute_error":
                metrics.median_absolute_error(y, predict),
                "r2_score":
                metrics.r2_score(y, predict)
            },
                         index=[0]))


def classification_score_metrics(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "accuracy_score": metrics.accuracy_score(y, pred),
        "balanced_accuracy_score": metrics.balanced_accuracy_score(y, pred),
        "metrics.average_precision_score": metrics.average_precision_score(
            y, pred),
        "f1_score": metrics.f1_score(y, pred),
        "precision_score": metrics.precision_score(y, pred),
        "recall_score": metrics.recall_score(y, pred),
        "roc_auc_score": metrics.roc_auc_score(y, pred)
    }


def regression_score_metrics(model, X, y):
    predict = model.predict(X)
    return {
        "explained_variance_score": metrics.explained_variance_score(
            y, predict),
        "mean_absolute_error": metrics.mean_absolute_error(y, predict),
        "mean_squared_log_error": metrics.mean_squared_log_error(y, predict),
        "median_absolute_error": metrics.median_absolute_error(y, predict),
        "r2_score": metrics.r2_score(y, predict)
    }


def train_tuned_xgboost(X,
                        y,
                        regression=False,
                        test_size=0.25,
                        eval_metric="auc",
                        optimized_metric=metrics.average_precision_score,
                        max_evals=10,
                        weight_imbalanced=False,
                        verbose=False,
                        random_state=777):
    """
    Quickly create a tuned xgboost model.
    If categorical feature in dataset, each categorical feature is transformed with labelEncoder (from scikit-learn),
    then we transformed data with one-hot-encoding.
    :param X: whole feature set (matrix pandas DataFrame)
    :param y: target vector (vector pandas Series)
    :param test_size: size of the test set (this value is reused to split the (1-test_size) train set in train and valid set).
    :param max_evals: try max_evals parameters combination for evaluation before returning best hyerparameters combination.
    :param random_state: seed used by the random number generator for reproductibility.
    :return: (X_train: features from train set,
     y_train: target from train set, 
     X_val: feature from validation set,
     y_val: target from validation set,
     X_test: feature from test set,
     y_test: target from test set,  
     xgboost: XGBClassifierTuning with xgboost model in it)
    """

    # Split in train, validation, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state)

    # XGboost model
    if regression:
        xgboost = XGBRegressorTuning(
            X_train,
            y_train,
            X_val,
            y_val,
            eval_metric=eval_metric,
            optimized_metric=optimized_metric,
            max_evals=max_evals,
            verbose=verbose,
            random_state=random_state)
    else:
        xgboost = XGBClassifierTuning(
            X_train,
            y_train,
            X_val,
            y_val,
            eval_metric=eval_metric,
            optimized_metric=optimized_metric,
            max_evals=max_evals,
            weight_imbalanced=weight_imbalanced,
            verbose=verbose,
            random_state=random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test, xgboost


def xgboost_baseline(X,
                     y,
                     regression=False,
                     n_splits=10,
                     test_size=0.25,
                     eval_metric="auc",
                     optimized_metric=metrics.average_precision_score,
                     max_evals=10,
                     weight_imbalanced=False,
                     verbose=False,
                     random_state=777):
    """
    Quickly run benchmark multiple times on a dataset to evalute xgboost model on it.
    At each iteration, we split the dataset in train / valid / test and trained a tuned xgboost model.
    We return different classification performance metrics.

    If categorical feature in dataset, each categorical feature is transformed with labelEncoder (from scikit-learn),
    then we transformed data with one-hot-encoding.
    :param X: whole feature set (matrix pandas DataFrame)
    :param y: target vector (vector pandas Series)
    :param n_splits: number of iteration to repeat.
    :param test_size: size of the test set (this value is reused to split the (1-test_size) train set in train and valid set).
    :param max_evals: try max_evals parameters combination for evaluation before returning best hyerparameters combination.
    :param random_state: seed used by the random number generator for reproductibility.
    :return: classification performance metrics for each iteration:
     f1_score, f2_score, precision_score, recall_score, metrics.average_precision_score, roc_auc_score.
    """

    metrics = []
    rs = ShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)

    for train_index, test_index in tqdm(rs.split(X), total=n_splits):

        # get train and test set
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # Add valid set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state)

        # XGboost model
        if regression:
            xgboost = XGBRegressorTuning(
                X_train,
                y_train,
                X_val,
                y_val,
                eval_metric=eval_metric,
                optimized_metric=optimized_metric,
                max_evals=max_evals,
                verbose=verbose,
                random_state=random_state)
            metrics.append(
                regression_score_metrics(xgboost.model, X_test, y_test))
        else:
            xgboost = XGBClassifierTuning(
                X_train,
                y_train,
                X_val,
                y_val,
                eval_metric=eval_metric,
                optimized_metric=optimized_metric,
                max_evals=max_evals,
                weight_imbalanced=weight_imbalanced,
                verbose=verbose,
                random_state=random_state)
            metrics.append(
                classification_score_metrics(xgboost.model, X_test, y_test))

    return metrics