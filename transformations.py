# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:14:24 2024

@author: Pedro
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class NoneScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

class SNVScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        self.stds_[self.stds_ < 1] = 1  # Threshold for small stds
        # self.stds_[self.stds_ == 0] = 0.001
        return self
    
    def transform(self, X):
        X_out = (X - self.means_) / self.stds_
        return X_out
    
    def __repr__(self):
        return "Standard\nNormal Variate"

# class LogShift(BaseEstimator, TransformerMixin):
#     def __init__(self, shift=1.0):
#         self.shift = shift

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return np.log(X + self.shift)

#     def inverse_transform(self, X):
#         return np.exp(X) - self.shift
    
    
class LogShift(BaseEstimator, TransformerMixin):
    def __init__(self, shift=0.1, apply_to="both"):
        """
        Parameters:
        - shift: float, the value to add before taking the log
        - apply_to: str, one of "X", "y", or "both" to specify where to apply the transformation
        """
        self.shift = shift
        self.apply_to = apply_to

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Apply the log shift transformation.
        
        Parameters:
        - X: array-like, input features
        - y: array-like, target values (optional)
        
        Returns:
        - tuple of transformed (X, y) depending on the apply_to parameter
        """
        X_transformed = X
        y_transformed = y

        if self.apply_to in ["X", "both"]:
            X_transformed = np.log(X + self.shift)

        if self.apply_to in ["y", "both"] and y is not None:
            y_transformed = np.log(y + self.shift)

        return X_transformed if y is None else (X_transformed, y_transformed)

    def inverse_transform(self, X, y=None):
        """
        Apply the inverse of the log shift transformation.
        
        Parameters:
        - X: array-like, input features
        - y: array-like, target values (optional)
        
        Returns:
        - tuple of inverse-transformed (X, y) depending on the apply_to parameter
        """
        X_inverse = X
        y_inverse = y

        if self.apply_to in ["X", "both"]:
            X_inverse = np.exp(X) - self.shift

        if self.apply_to in ["y", "both"] and y is not None:
            y_inverse = np.exp(y) - self.shift

        return X_inverse if y is None else (X_inverse, y_inverse)
    
    def __repr__(self):
        return f"Log({self.apply_to}+{self.shift})"
    
class CLRscaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Fit method doesn't need to do anything for this transformer
        return self

    def transform(self, X):
        return self._clr_transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _clr_transform(self, compositional_data):
        compositional_data += 1e-9  # Add small value to avoid log(0)
        geometric_mean = np.exp(np.mean(np.log(compositional_data), axis=1))
        geometric_mean = np.asarray(geometric_mean)  # Ensure it is a NumPy array
        clr_data = np.log(compositional_data / geometric_mean[:, np.newaxis])
        return clr_data
    

from scipy.signal import savgol_filter
from scipy import signal

# Define custom transformations
class DeTrend:
    def __init__(self, type='linear'):
        self.type = type
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return signal.detrend(X, type = self.type)
    
class SavGol(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = savgol_filter(X, self.window_length, self.polyorder, axis=0)

        # Ensure output remains a DataFrame with original index and column names
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_transformed, index=X.index, columns=X.columns)
        else:
            return X_transformed  # Fallback for NumPy arrays
    
class Derivative(BaseEstimator, TransformerMixin):
    def __init__(self, order=1):
        self.order = order
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.gradient(X, axis=0, edge_order=self.order)
    
    def get_params(self, deep=True):
        return {"order": self.order}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
import numpy as np

class MSCWrapper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # Compute the reference spectrum if not provided
        self.reference_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        X_corrected = np.zeros_like(X)
        for i in range(X.shape[0]):
            # Linear fit: X_i = a + b * reference
            fit = np.polyfit(self.reference_, X[i, :], 1)
            a, b = fit
            X_corrected[i, :] = (X[i, :] - a) / b
        return X_corrected
    
from sklearn.base import BaseEstimator, TransformerMixin

class EmptyScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X  
    
def apply_dilution_factor(df,assay,absorbance_columns, days, assays, dilution_factor,
                          multi_index = False):
    """
    Applies a dilution factor to specified days and assays in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - days: List of days to apply the dilution factor to.
    - assays: List of assays to apply the dilution factor to.
    - dilution_factor: The dilution factor to apply.

    Returns:
    - A pandas DataFrame with the dilution factor applied to the specified days and assays.
    """

    if multi_index:
        # Filter the DataFrame for the specified days and assays
        filtered_df = df[(df.index.get_level_values('Day').isin(days)) &
                         (df.index.get_level_values(assay).isin(assays))]

    else:   
        # Filter the DataFrame for the specified days and assays
        filtered_df = df[(df['Day'].isin(days)) & (df[assay].isin(assays))]
    
    # Columns that contain absorbance values
    # absorbance_columns = [str(i) for i in range(300, 800)]  # Assuming columns are named as strings
    
    # Apply the dilution factor to the absorbance values
    df.loc[filtered_df.index, absorbance_columns] = df.loc[filtered_df.index, absorbance_columns].apply(lambda x: x * dilution_factor)
    return df