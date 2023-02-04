'''
Created with love by Sigmoid

@Author - Stojoc Vladimir - vladimir.stojoc@iis.utm.md
'''
# Importing all needed libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .errors import NonNumericDataError


class KNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors = 5, epsilon = 1e-10) -> None:
        '''
            The constructor of the KNNImputer class.
        :param n_neighbors: int, default = 5
            The number of neighbors to find for every sample with missing values.
        :param metric: str, default = 'minkowski'
            The distance metric to use for finding the nearest neighbors.
        :param epsilon: int, default = 1e-10
            The parameter do add to inverse distance weight in case the distance is 0 to nearest neighbors
        '''
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon


    def fit(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict):
        '''
            The fit function of the KNNImputer, fits up the model.
        :param X: 2-d numpy array or pd.DataFrame
            The 2-d numpy array or pd.DataFrame that represents the feature matrix.
        :param y: 1-d numpy array or pd.DataFrame
            The 1-d numpy array or pd.DataFrame that represents the target array.
        :param fit_params: dict
            The fit parameters of the function, they are ignored in the body of the function.
        :return: KNNImputer
            The fitter KNNImputer object.
        '''
        return self

    def transform(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict) -> 'np.array':
        '''
            The transform function of the KNNImputer, transforms the passed data..
        :param X: 2-d numpy array
            The 2d matrix representing the data that should be filled.
        :param y: 1-d numpy array, default = None
            The 1d array representing the target vector, it is not used in this function, ignored.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''

        if not np.issubdtype(X.dtype, np.number):
            raise NonNumericDataError('Not all passed columns of the dataframe represent numeric data.')

        # Creating a copy of array
        new_X = X.copy()

        # Looping through all rows
        for x_idx, x in enumerate(X):
            check_nan = np.isnan(x)
            # Check for Nans in rows
            if check_nan.any():
                nan_indexes = np.where(check_nan == True)[0]
                # Go through columns with NaNs
                for i in nan_indexes:
                    # select all rows from X where the ith column in not nan
                    nan_col = X[:, i]
                    X_not_nan_flags = np.isnan(nan_col)
                    # Get all rows where this column is not NaN
                    X_not_nan_indexes = np.where(X_not_nan_flags == False)[0]
                    X_not_nan = X[X_not_nan_indexes, :]

                    # Get closest k neighbours
                    nearest_neighbors, distances = self.__get_k_neighbours(x, X_not_nan)
                    # Select the needed column from nearest neighbors
                    nearest_neighbors = nearest_neighbors[:, i]
                    # Calculating a weighted average
                    weighted_average = np.average(nearest_neighbors, weights=1/distances)

                    # Imputing calculated value
                    new_X[x_idx, i] = weighted_average

        return new_X


    def fit_transform(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict):
        '''
            The transform function of the KNNImputer, transforms the passed data..
        :param X: 2-d numpy array
            The 2d matrix representing the data that should be filled.
        :param y: 1-d numpy array, default = None
            The 1d array representing the target vector, it is not used in this function, ignored.
        :param fit_params: dict
            The fit parameters that control the fitting process.
        :return: np.array
            The transformed data.
        '''
        return self.fit(X).transform(X)

    def apply(self, df: 'pd.DataFrame', columns: list = None) -> 'pd.DataFrame':
        '''
            This function allows applying the transformer on certain columns of a data frame.
        :param df: pandas DataFrame
            The data frame with the possible NaN-values that should be inputed.
        :param columns: list
            The column that should be taken into account during the NaN value imputation process.
        :return: pandas DataFrame
            The new pandas DataFrame with transformed columns.
        '''

        cols_not_in_df = [col for col in columns if col not in df]
        if cols_not_in_df:
            raise ValueError(f"{', '.join(cols_not_in_df)} are not present in the DataFrame")

        X = df[columns].values

        if not np.issubdtype(X.dtype, np.number):
            raise NonNumericDataError('Not all passed columns of the dataframe represent numeric data.')
        else:
            df_new = self.fit_transform(X)
            df.update(pd.DataFrame(df_new, columns=columns))


    def __get_k_neighbours(self, x, X_not_nan):
        '''
            KNN, getting nearest neighbors
        :param x: Numpy array
            the row from df to get neighbours from
        :param X_not_nan: Numpy.ndarray
            Samples from where we find neighbours
        '''

        # Calculating distance to all other samples
        dist = np.sqrt(np.nansum((X_not_nan - x) ** 2, axis=1))
        dist += self.epsilon
        dist_rows = list(enumerate(dist))
        # Sorting distances and first k neighbors
        dist_rows.sort(key=lambda x: x[1])
        dist_rows = dist_rows[:self.n_neighbors]

        return X_not_nan[np.array([x[0] for x in dist_rows])], np.array([x[1] for x in dist_rows])