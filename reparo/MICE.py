'''
Created with love by Sigmoid
@Author - Basoc Nicoleta-Nina - nicoleta.basoc28@gmail.com
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from .errors import OnlyOneColumnSelectedError, NonNumericDataError


class MICE(BaseEstimator, TransformerMixin):
    
    def fit(self, X: 'np.array', y: 'np.array'= None, **fit_params):
        """
        :param X: array-like of shape (n_samples, n_features)
            input samples
        :param y: array-like of shape (n_samples)
            target values (None for unsupervised transformations)
        :param fit_params: dict
            additional fit parameters
        :return: self
        """
        return self

    def transform(self, X: 'np.array', y: 'np.array' = None, **fit_params):

        if not np.issubdtype(X.dtype, np.number):
            raise NonNumericDataError('The given array contains non-numeric values !')

        df_copy = pd.DataFrame(X).copy()
        new_df = []

        for column in df_copy.columns:
            if df_copy[column].isna().any():
                # Select columns all the other columns except the one we want to impute right now
                columns_imputed = df_copy.loc[:, df_copy.columns != column].columns
                # Select the indexes of NaN elements
                nan_indexes = np.where(df_copy[column].isna())[0]
                # Perform mean imputation and create the zeroth dataset
                zeroth_df = pd.DataFrame(np.array([df_copy[column_zeroth].fillna(np.mean(df_copy[column_zeroth]))
                                                   for column_zeroth in columns_imputed]).transpose())
                # Select train and test indicies
                train_indicies = df_copy.loc[pd.isna(df_copy[column]) == False].index
                test_indicies = df_copy.loc[pd.isna(df_copy[column])].index
                # Train linear regression
                linear_regression = LinearRegression()
                linear_regression.fit(zeroth_df.iloc[train_indicies], df_copy[column].iloc[train_indicies])

                predictions = linear_regression.predict(zeroth_df.iloc[test_indicies])
                # Impute the predictions to the given positions
                for idx, prediction in zip(test_indicies, predictions):
                    df_copy[column].iloc[idx] = prediction
            else:
                continue

        return df_copy.values
    
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
        #Only one column selected
        if len(df[columns].columns)<=1:
            raise OnlyOneColumnSelectedError('The given algorithm cannot be applied only on one column')
        
        # check if the given columns are in the Dataframe
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

        # apply the transformation on the DataFrame
        sub_df_new = self.fit_transform(sub_df)

        # save the transformation in the DataFrame to return
        df.update(pd.DataFrame(sub_df_new, columns=columns))
