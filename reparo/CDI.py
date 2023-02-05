'''
Created with love by Sigmoid
@Author - Denis Smocvin - denissmocvin@gmail.com
'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np

from .errors import NonNumericDataError, NoSuchDistanceMetricError, AllSamplesHaveNaNsError, SampleFullOfNaNs


class CDI(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', p: int = 2, n_select: int = 1):
        '''
        Initialize the algorithm.
        :param n_neighbors: the number of neighbors to find for every sample with missing values
        :param metric: the distance metric to use for finding the nearest neighbor
        :param p: the parameter for the Minkowski metric
        :param n_select: the index of the selected sample
        '''

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.n_select = n_select

        self.DISTANCES = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']

        if metric.lower() not in self.DISTANCES:
            error_text = f"{metric} metric isn\'t supported right now, choose one of the following: {', '.join(self.DISTANCES)}"
            raise NoSuchDistanceMetricError(error_text)

        if  n_select > n_neighbors:
            error_text = f'n_select ({n_select}) > n_neighbors ({n_neighbors}) : The selected sample index is higher than the n neighbors used by the algorithm, but should be lower.'
            raise ValueError(error_text)

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

    def transform(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """
        Impute according to the Cold Deck Imputation algorithm. It will try to find as many neighbors as possible up to
        the given n_neighbors number of neighbors and choose the n_select-th one for imputation.

        Note: there should be at least on row without NaN values.
        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples,)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
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
                # create a copy of the original array with deleted rows that contained nans
                X_temp = X_new.copy()
                X_temp = np.array([x for x in X_temp if not np.isnan(x).any()])

                if len(X_temp) == 0:
                    error_text = 'All samples contain NaN values. Cold Deck Imputation isn\'t fitted for this configuration. Consider using other imputation algorithms.'
                    raise AllSamplesHaveNaNsError(error_text)

                # the algorithm will try to find as many neighbors as possible, the limit being the given number of neighbors
                n_neighbors = self.n_neighbors
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
                knn = NearestNeighbors()
                knn.fit(X_temp)
                neighbors = knn.kneighbors(return_distance=False, n_neighbors=n_neighbors)

                # impute
                neighbor = X_temp[neighbors[0][self.n_select - 1]]

                for j, _ in enumerate(sample):
                    if np.isnan(sample[j]):
                        sample[j] = neighbor[j]

        return X_new

    def fit_transform(self, X: 'np.array', y: 'np.array'=None, **fit_params):
        """
        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples,)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
        :return: ndarray array of shape (n_samples, n_features_new)
            transformed array
        """

        return self.fit(X).transform(X)

    def apply(self, df: 'pd.DataFrame', columns: list):
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