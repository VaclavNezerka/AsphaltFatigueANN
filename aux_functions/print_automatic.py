import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint


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

def plot_ann_performance_dec(y_true, y_pred, dataset_name, color, marker, s=20):
    """
    Plot the performance of an ANN model on given data.

    This function creates a scatter plot to compare the true output values with the predicted output values from an Artificial Neural Network (ANN) model. The plot uses a logarithmic scale for both axes and includes a diagonal line representing perfect prediction.

    Args:
    - y_true (array-like): True output values.
    - y_pred (array-like): Predicted output values.
    - dataset_name (str): Name of the dataset (e.g., 'Train', 'Test'). This will be used in the plot legend.
    - color (str): Color for the scatter plot points.
    - marker (str): Marker style for the scatter plot points.
    - s (int, optional): Size of the markers. Default is 20.
    """

    # Ensure y_true and y_pred are numpy arrays and squeeze them to 1D
    y_true, y_pred = np.squeeze(np.array(y_true)), np.squeeze(np.array(y_pred))

    # Determine the range for the plot
    min_value, max_value = np.min([y_true, y_pred]), np.max([y_true, y_pred])

    # Create the scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, color=color, label=dataset_name, marker=marker, s=s)

    # Set the axis to scientific notation format
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Set the axis to logarithmic scale
    plt.yscale('log')
    plt.xscale('log')

    # Add a legend
    plt.legend(loc='upper left')

    # Label the axes
    plt.xlabel(r'$N_{\mathrm{f}}^{\mathrm{true}}$')
    plt.ylabel(r'$N_{\mathrm{f}}^{\mathrm{predicted}}$')

    # Plot the diagonal line representing perfect prediction
    plt.plot([min_value, max_value], [min_value, max_value], color='black')

    # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Set the axis limits
    plt.axis([min_value, max_value, min_value, max_value])

    # Add a grid with specific style
    plt.grid(True, linestyle='--', alpha=0.7)


def ann_performance_report_separatly_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions,
                                         validate_outputs, validate_predictions):
    """
    Generate and save plots for the performance of an ANN model on train, test, and validation datasets.

    This function creates scatter plots for the true vs. predicted values of the ANN model on different datasets (train, test, and validation). It also calculates and displays performance metrics including R^2, RMSE, and RMSLE on each plot. The plots are saved as both PDF and JPG files in the specified directory.

    Args:
    - dir_name (str): Directory where the plots will be saved.
    - train_outputs (array-like): True output values for the training set.
    - train_predictions (array-like): Predicted output values for the training set.
    - test_outputs (array-like): True output values for the test set.
    - test_predictions (array-like): Predicted output values for the test set.
    - validate_outputs (array-like): True output values for the validation set.
    - validate_predictions (array-like): Predicted output values for the validation set.
    """

    # Define the datasets with their corresponding color and marker for plotting
    datasets = {
        "Test": (test_outputs, test_predictions, 'blue', 'o'),
        "Train": (train_outputs, train_predictions, 'green', 'x'),
        "Validate": (validate_outputs, validate_predictions, 'red', '*')
    }

    for name, (outputs, predictions, color, marker) in datasets.items():
        # Plot the performance
        plot_ann_performance_dec(outputs, predictions, name, color, marker)

        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(outputs, predictions))
        r2 = r2_score(outputs, predictions)
        rmsle = rmsle_calculation(outputs, predictions)

        # Display the performance metrics on the plot
        plt.text(0.05, 0.90, f'$R^2 = {r2:.2f}$\n$RMSE = {rmse:.0f}$\n$RMSLE = {rmsle:.4f}$',
                 transform=plt.gca().transAxes, verticalalignment='top')

        # Save the plots as PDF and JPG
        plt.savefig(f"{dir_name}/{name.lower()}.pdf", format='pdf')
        plt.savefig(f"{dir_name}/{name.lower()}.jpg", format='jpg')
        plt.close()

def ann_performance_report_all_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions, validate_outputs, validate_predictions):
    """
    Generate and save plots for the performance of an ANN model on train, test, and validation datasets.

    This function creates scatter plots to compare the true and predicted values for the ANN model on different datasets (train, test, and validation). It also combines all data together to create an overall performance plot. The plots include performance metrics such as R^2, RMSE, and RMSLE, which are displayed on the plots and saved in both PDF and JPG formats.

    Args:
    - dir_name (str): Directory where the plots will be saved.
    - train_outputs (array-like): True output values for the training set.
    - train_predictions (array-like): Predicted output values for the training set.
    - test_outputs (array-like): True output values for the test set.
    - test_predictions (array-like): Predicted output values for the test set.
    - validate_outputs (array-like): True output values for the validation set.
    - validate_predictions (array-like): Predicted output values for the validation set.
    """

    # Combine all data together
    all_outputs = np.concatenate([train_outputs, test_outputs, validate_outputs])
    all_predictions = np.concatenate([train_predictions, test_predictions, validate_predictions])

    # Plot all data separately
    plot_ann_performance_dec(train_outputs, train_predictions, 'Train', 'green', 'x', 20)
    plot_ann_performance_dec(test_outputs, test_predictions, 'Test', 'blue', 'o', 20)
    plot_ann_performance_dec(validate_outputs, validate_predictions, 'Validate', 'red', '*', 20)

    # Calculate accuracy metrics for the combined data
    rmse_all = np.sqrt(mean_squared_error(all_outputs, all_predictions))
    r2_all = r2_score(all_outputs, all_predictions)
    rmsle_all = rmsle_calculation(all_outputs, all_predictions)

    # Display the performance metrics on the combined plot
    plt.text(0.05, 0.78, f'$R^2 = {r2_all:.2f}$\n$RMSE = {rmse_all:.0f}$\n$RMSLE = {rmsle_all:.4f}$',
             transform=plt.gca().transAxes, verticalalignment='top')

    # Save the combined plot as PDF and JPG
    plt.savefig(f"{dir_name}/all.pdf", format='pdf')
    plt.savefig(f"{dir_name}/all.jpg", format='jpg')
    plt.close()