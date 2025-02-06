import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from keras.models import Sequential
from keras.layers import Dense





def rmsle_calculation(y_true, y_pred):
    """
    Calculate the Root Mean Squared Logarithmic Error (RMSLE).

    The RMSLE is a metric used to measure the accuracy of a model's predictions. It is particularly useful when the target variable has a wide range or when predicting values that can be exponentially large. The logarithmic transformation helps to manage large differences and to penalize underestimations more than overestimations.

    Args:
    - y_true (array-like): True values. These should be non-negative.
    - y_pred (array-like): Predicted values. These should also be non-negative.

    Returns:
    - float: The RMSLE value, which indicates the average magnitude of the log-transformed prediction errors.

    Note:
    - The function uses `np.log1p` to compute the natural logarithm of (1 + value) for better numerical stability, especially when dealing with small values.
    """

    # Convert inputs to numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Compute the logarithms of (1 + values)
    log_true = np.log1p(y_true)  # log1p is used for better numerical stability with small values
    log_pred = np.log1p(y_pred)

    # Calculate the squared logarithmic errors
    squared_log_error = np.square(log_pred - log_true)

    # Compute the mean of the squared logarithmic errors
    mean_squared_log_error = np.mean(squared_log_error)

    # Return the square root of the mean squared logarithmic error
    return np.sqrt(mean_squared_log_error)