import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .errors import NonNumericDataError, SampleFullOfNaNs
import statsmodels.api as sm


class PMM(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initializing the algorithm.
        """
        return

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

    def __get_chi_squared_value(self, X: 'np.array'):
        '''
        Function to calculate the chi-value of an array.
        :param X: np.array of shape (n_samples, n_features)
            The input samples (required).
        :return: the chi-value of the x_observed array.
        '''
        X_chi = np.multiply.outer(X.sum(1), X.sum(0)) / X.sum()
        X_chi = ((X_chi - X) ** 2) / X_chi
        return X_chi.sum()

    def transform(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict):
        """
        Filling in the missing values of X.
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

        # making the working copy of X and its missing values
        X_copy = X.copy()
        missing_values = np.where(pd.isnull(X_copy))

        if len(set(missing_values[1])) >= X_copy.shape[1] - 1:
            raise SampleFullOfNaNs("All columns contain missing values. Can't impute using pmm. ")

        # a buffer where the columns with missing values are kept
        columns_still_with_missing_values = list(set(missing_values[1]))

        while len(columns_still_with_missing_values) > 0:
            x = np.delete(X_copy, columns_still_with_missing_values, axis = 1)
            column = columns_still_with_missing_values.pop()
            y = X_copy[:, column]

            missing_values_rows = missing_values[0][np.where(missing_values[1] == column)]
            x_missing = x[missing_values_rows]
            x_observed = np.delete(x, missing_values_rows, axis=0)
            y_observed = np.delete(y, missing_values_rows, axis=0)

            ols = sm.OLS(y_observed, sm.add_constant(x_observed)).fit()
            beta_shapka = np.delete(ols.params, 0, axis = 0)
            # sigma_shapka = ols.params[0]
            epsilon_shapka = abs(ols.predict(sm.add_constant((x_observed))) - y_observed)
            std_dev = np.matmul(epsilon_shapka.T, epsilon_shapka) / self.__get_chi_squared_value(x_observed)
            beta_zviozdacika = np.random.multivariate_normal(
                beta_shapka,
                std_dev * np.linalg.inv(np.matmul(x_observed.T, x_observed))
            )

            y_observed_pred = np.matmul(x_observed, beta_shapka)
            y_missing_pred = np.matmul(x_missing, beta_zviozdacika)
            for i in range(len(y_missing_pred)):
                deltas = np.vstack(
                    (abs(y_observed_pred - y_missing_pred[i]), y_observed_pred)
                ).T
                deltas = deltas[deltas[:, 0].argsort()]
                y[missing_values_rows[i]] = np.random.choice(deltas[:3, 1])

        return X_copy

    def fit_transform(self, X: 'np.array', y: 'np.array' = None, **fit_params: dict):
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
        if not set(columns).issubset(set(df.columns)):
            raise ValueError("Columns not found in the DataFrame.")

        data = df[columns].copy()
        imputed_data = self.fit_transform(data.to_numpy())

        df.update(pd.DataFrame(imputed_data, columns=columns))
