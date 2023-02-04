'''
Created with love by Sigmoid
@Author - Denis Smocvin - denissmocvin@gmail.com
'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

from .errors import NonNumericDataError, NoSuchDistanceMetricError, AllSamplesHaveNaNsError, SampleFullOfNaNs

from scipy.spatial.distance import minkowski

class FRNNI(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', p: int =2):
        '''
        Initialize the algorithm.
        :param n_neighbors: the number of neighbors to find for every sample with missing values
        :param metric: the distance metric to use for finding the nearest neighbors
        :param p: the parameter for the Minkowski metric
        '''

        self.distances = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

        if metric.lower() not in self.distances:
            raise NoSuchDistanceMetricError(f'{metric} metric isn\'t supported right now, choose one of the following: euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis.')

    def fit(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """
        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples,)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
        :return: self
        """

        return self

    def transform(self, X: 'np.array', y: 'np.array' = None, **fit_params):
        """
        fill the missing values using Fuzzy-Rough Nearest Neighbor algorithm for imputation
        :param X: array-like of shape (n_samples, n_features)
            2d matrix representing the data that should be filled
        :param y: array-like of shape (n_samples,)
            1d array representing the target vector
        :param fit_params: dict
            the fit parameteres of the function
        :return: ndarray array of shape (n_samples, n_features_new)
            transformed array
        """

        # check if data is numeric
        if not np.issubdtype(X.dtype, np.number):
            raise NonNumericDataError('The given array contains non-numeric values !')

        X_new = X.copy()

        for i, sample in enumerate(X_new):
            if np.isnan(sample).all():
                error_text = f'Sample {i} is full of NaNs. Cannot operate with such samples. Consider removing the row.'
                raise SampleFullOfNaNs(error_text)

            if np.isnan(sample).any():
                # find neighbors
                d, N = self.__getNearestNeighbors(sample, X, n_neighbors=self.n_neighbors, metric=self.metric)

                for j, a in enumerate(sample):
                    if np.isnan(a):
                        tau1 = tau2 = 0

                        # create a copy of the original array with deleted rows that contained nans
                        X_temp = X_new.copy()
                        X_temp = np.array([x for x in X_temp if not np.isnan(x).any()])

                        if len(X_temp) == 0:
                            error_text = 'All samples contain NaN values. FRNNI isn\'t fitted for this configuration.'
                            raise AllSamplesHaveNaNsError(error_text)

                        # print(N)
                        for k, z in enumerate(N):
                            for l, t in enumerate(N):
                                # print(z, t)
                                if np.array_equal(t, z):
                                    continue

                                dist = (d[k], d[l])
                                # print(d, dist)

                                lower = self.__approximation(t, z, np.array(X_new).T[j], dist, N, j, type='lower')
                                upper = self.__approximation(t, z, np.array(X_new).T[j], dist, N, j, type='upper')

                                M = (lower + upper) / 2

                                tau1 += M * z[j]
                                tau2 += M

                        sample[j] = tau1 / tau2 if tau2 > 0 else sum([neighbor[j] for neighbor in N]) / len(N)

        return X_new

    def fit_transform(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """
        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
        :param X: array-like of shape (n_samples, n_features)
            2d matrix representing the data that should be filled
        :param y: array-like of shape (n_samples,)
            1d array representing the target vector
        :param fit_params: dict
            the fit parameteres of the function
        :return: ndarray array of shape (n_samples, n_features_new)
            transformed array
        """

        return self.fit(X).transform(X)

    def apply(self, df: 'pandas DataFrame', columns: 'list'):
        """
        Apply the algorithm on a DataFrame
        :param df: pandas DataFrame
            the DataFrame with the possible NaN-values that should be imputed
        :param columns: list of strings
            the column that should be taken into account during the NaN-values imputation process
        :return:
        """

        # check if the given columns are in the DataFrame
        cols_not_in_df = [col for col in columns if col not in df]
        if cols_not_in_df:
            raise ValueError(f"{', '.join(cols_not_in_df)} are not present in the DataFrame")

        # select the desired columns
        sub_df = df[columns]

        # check if data is numeric
        for col in sub_df.columns:
            if not pd.api.types.is_numeric_dtype(sub_df[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values !')

        # convert DataFrame to numpy array
        sub_df = sub_df.to_numpy()

        # apply the transformation
        sub_df_new = self.fit_transform(sub_df)

        # save the transformation in the DataFrame
        df.update(pd.DataFrame(sub_df_new, columns=columns))

    def __getNearestNeighbors(self, sample, X, n_neighbors, metric):
        # make a copy and drop nans
        X_temp = X.copy()
        X_temp = np.array([x for x in X_temp if not np.isnan(x).any()])

        # the algorithm will try to find as many neighbors as possible, the limit being the given number of neighbors
        if n_neighbors > len(X_temp):
            n_neighbors = len(X_temp)

        # put at the start of the array the sample requiring imputation with means temporarily in place of nans
        temp = sample.copy()
        means = [col.mean() for col in X_temp.T]

        for j, _ in enumerate(temp):
            if np.isnan(temp[j]):
                temp[j] = means[j]

        X_temp = np.insert(X_temp, 0, temp, axis=0)

        # find the nearest neighbors
        if metric == 'minkowski':
            knn = NearestNeighbors(metric=metric, p=self.p)
        else:
            knn = NearestNeighbors(metric=metric)
        knn.fit(X_temp)
        dist, neighbors = knn.kneighbors(n_neighbors=n_neighbors)

        return dist[0], X_temp[neighbors[0]]

    def __R_a(self, y, t, a):
        return 1 - np.abs(y - t) / (max(a) - min(a))

    def __R(self, y, t, a, d):
        equal = 1 if np.array_equal(d[0], d[1]) else 0

        return min(self.__R_a(y, t, a), 1 - equal)

    def __implication(self, a, b):
        return max(1 - a, b)

    def __t_norm(self, a, b):
        return min(a, b)

    def __approximation(self, y: list, z: list, a: list, d: tuple, N: 'np.array', i: int, type: str):
        values = []

        for t in N:
            r = self.__R(y[i], t[i], a, d)
            r_a = self.__R_a(y[i], z[i], a)

            if type == 'lower':
                values.append(self.__implication(r, r_a))
            elif type == 'upper':
                values.append(self.__t_norm(r, r_a))

        return min(values) if type == 'lower' else max(values)