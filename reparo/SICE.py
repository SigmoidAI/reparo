"""
Created with love by Sigmoid
@Author - Sclearuc Marius - marius.sclearuc@gmail.com
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .MICE import MICE


class SICE(BaseEstimator, TransformerMixin):
    def __init__(self, n_cycle : int = 5):
        """
            Initializing the algorithm.
        :param n_cycle: int, default = 0.05
            The number of cycles the algorithm goes through.
        """
        self.n_cycle = n_cycle

    def fit(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict):
        """
        :param X: np.array of shape (n_samples, n_features)
            The input samples (optional).
        :param y: np.array of shape (n_samples,), default = None
            The target values (optional).
        :param fit_params: dict, default = None
            Additional fit parameters (optional).
        :return: self
        """
        return self

    def transform(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict):
        """
        Fill in the missing values of X.
        :param X: np.array of shape (n_samples, n_features)
            The input samples.
        :param y: np.array of shape (n_samples, ), default = None
            The target values (none for unsupervised transformations).
        :param fit_params: dict
            Optional additional fit parameters.
        :return: np.array of shape (n_samples, n_features)
            The original data, but with the missing values imputed.
        """
        missing_values = np.where(pd.isnull(X))
        missing_values = list(zip(missing_values[0], missing_values[1]))
        mice_computed_values = {location: list() for location in missing_values}

        mice = MICE()

        for i in range(self.n_cycle):
            mice_output = mice.fit_transform(X)
            for location in missing_values:
                mice_computed_values[location].append(mice_output[missing_values])

        X_copy = X.copy()
        for location in missing_values:
            if type(mice_computed_values[location][0]) in [int, float]:
                X_copy[location] = np.mean(mice_computed_values[location])
            else:
                values, counts = np.unique(mice_computed_values[location], return_counts=True)
                X_copy[location] = values[np.argmax(counts)]

        return X_copy

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
        :param X: np.array of shape (n_samples, n_features)
            The input samples.
        :param y: np.array of shape (n_samples, ), default = None
            The target values (none for unsupervised transformations).
        :param fit_params: dict
            Optional additional fit parameters.
        :return: np.array of shape (n_samples, n_features)
            The original data, but with the missing values imputed.
        """
        return self.fit(X).transform(X)

    def apply(self, df, columns):
        """
        Apply the algorithm on a DataFrame.
        :param df: pandas DataFrame
            The DataFrame with possible NaN-values that should be imputed
        :param columns: list of str
            The columns on which to apply the algorithm.
        :return: pandas DataFrame
            The DataFrame with imputed missing-values.
        """
        data = df[columns].copy()
        imputed_data = self.transform(data.values)

        df.update(pd.DataFrame(imputed_data, columns=columns))
