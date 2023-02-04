"""
Created with love by Sigmoid
@Author - Sclearuc Marius - marius.sclearuc@gmail.com
"""

import pandas as pd
import numpy as np
from random import choice
from sklearn.base import BaseEstimator, TransformerMixin

from .errors import NoSuchDistanceMetricError, NonNumericDataError


class HotDeckImputation(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', p: int = 2, eps: float = 1e-10):
        """
            Initializing the algorithm.
        :param n_neighbors: int, default = 0.05
            The number of neighbours to consider for KNN.
        :param metric: str, default = 'minkowski'
            The type of distance metric to use for calculating the distance between data.
            Note: Supported metrics are: chebyshev, euclidean, seuclidean, manhattan, mahalanobis, minkowski.
        :param p: int, default = 2
            The parameter for the Minkowski metric.
            Note: for p = 2, the 'minkowski' metric is equivalent to the 'euclidean' metric.
        :param eps: float, default = 1e-10
            The additional error.
        """
        self.n_neighbors = n_neighbors
        self.p = p
        self.eps = eps

        # checking if metric distance is supported
        if metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']:
            self.metric = metric
        else:
            raise NoSuchDistanceMetricError(
                "X metric isn't supported right now, choose on of the following: euclidean, manhattan, "
                "chebyshev, minkowski, wminkowski, seuclidean, mahalanobis.")

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

    def __get_distance(self, a: 'np.array', b: 'np.array'):
        """
        Function to calculate the distance according to the metric.
        :param a: np.array
            The first array.
        :param b: np.array
            The second array.
        :return: float
            The computed distance.
        """
        if a.shape != b.shape:
            raise ValueError("Mismatch in __get_distance.")
        elif self.metric == 'manhattan':
            return sum([abs(a[i] - b[i]) for i in range(len(a))])
        elif self.metric == 'euclidean':
            return sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5
        elif self.metric == 'chebyshev':
            return max([(a[i] - b[i]) for i in range(len(a))])
        elif self.metric == 'minkowski':
            return sum([abs(a[i] - b[i]) ** self.p for i in range(len(a))]) ** (1/self.p)
        elif self.metric == 'wminkowski':
            return -1
        elif self.metric == 'seuclidean':
            return sum([(a[i] - b[i]) ** 2 for i in range(len(a))])
        cov_matrix = np.cov(np.stack((a, b), axis=1))
        if np.linalg.det(cov_matrix) == 0:
            raise ValueError("Can't compute covariance matrix.")
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return np.matmul(np.matmul((a - b).T, inv_cov_matrix), (a - b))

    def transform(self, X, y=None, **fit_params):
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
        # checking if received data is ok
        if not np.issubdtype(X.dtype, np.number):
            raise NonNumericDataError("The given DataFrame contains non-numeric values.")

        # checking if there are enough rows to find at least self.n_neighbors values
        if X.shape[0] < self.n_neighbors + 1:
            raise ValueError("The number of specified neighbors exceeds the shape number of samples.")

        # creating the working copy and making a dict mapping id (i.e. rows) to the columns of the missing values
        # i.e. if for row nr.2 there are missing values in columns 4, 10, then dict[2] = [4, 10]
        X_copy = X.copy()
        missing_values = np.where(np.isnan(X))
        missing_values = list(zip(missing_values[0], missing_values[1]))
        id_to_missing_value_columns_map = {i: list() for i in range(X.shape[0])}
        for id, column in missing_values:
            id_to_missing_value_columns_map[id].append(int(column))

        for id, column in missing_values:
            # taking a record/row and eliminating all its missing columns
            current_record = np.delete(X[id], id_to_missing_value_columns_map[id])
            # making a copy of all the dataset less the current record
            records = np.delete(X, id, axis=0)
            distances = list()
            for record in range(records.shape[0]):
                # checking if it's possible to compare the distances of the 2 records
                if np.isnan(records[record][column]) or \
                        (not set(id_to_missing_value_columns_map[record]).issubset(set(id_to_missing_value_columns_map[id]))):
                    continue
                # computing the distance, then appending it to the list of all distances
                distance = self.__get_distance(
                    current_record,
                    np.delete(
                        records[record],
                        id_to_missing_value_columns_map[id]
                    )
                )
                distances.append((records[record][column], distance))

            # checking once again if there are enough records to impute a missing value
            if len(distances) < 1:
                raise ValueError("Too many lacking spaces, can't properly fill the data.")
            distances.sort(key=lambda elem: elem[1])

            # choosing a random near neighbour and imputing its value
            nearest = distances[:min(len(distances), self.n_neighbors)]
            randomly_chosen_value = choice(nearest)
            X_copy[id, column] = randomly_chosen_value[0] + self.eps

        return X_copy

    def fit_transform(self, X: 'np.array', y: 'np.array' = None, **fit_params):
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

    def apply(self, df: 'pd.DataFrame', columns: list):
        """
        Apply the algorithm on a DataFrame.
        :param df: pandas DataFrame
            The DataFrame with possible NaN-values that should be imputed
        :param columns: list of str
            The columns on which to apply the algorithm.
        :return: pandas DataFrame
            The DataFrame with imputed missing-values.
        """
        if not set(columns).issubset(set(df.columns.values)):
            raise ValueError("Columns not found in the DataFrame.")

        imputed_data = self.fit_transform(df[columns].to_numpy())

        df.update(pd.DataFrame(imputed_data, columns=columns))
