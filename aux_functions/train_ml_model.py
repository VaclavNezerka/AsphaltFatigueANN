import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Nadam, RMSprop
from keras import initializers
from keras.callbacks import Callback
import random


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, *args, **kwargs):
        """
        Custom Model Checkpoint that starts saving the model only after a specified epoch.

        Args:
        - start_epoch (int): The epoch after which the model should start being saved.
        - *args, **kwargs: Arguments to be passed to the parent ModelCheckpoint class.
        """
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        """
        Check if the current epoch is greater than or equal to the start_epoch.
        If so, proceed to save the model.

        Args:
        - epoch (int): The current epoch number.
        - logs (dict): The metrics and losses at the end of the current epoch.
        """
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)

class EpochLogger(Callback):
    def __init__(self, log_file='training_log.xlsx'):
        """
        Custom callback to log training and validation loss at the end of each epoch and save the logs to an Excel file.

        Args:
        - log_file (str): The path to the Excel file where logs will be saved. Default is 'training_log.xlsx'.
        """
        super().__init__()
        self.log_file = log_file
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of each epoch, log the training and validation loss.

        Args:
        - epoch (int): The current epoch number.
        - logs (dict): The metrics and losses at the end of the current epoch.
        """
        logs = logs or {}
        self.logs.append({
            'epoch': epoch + 1,
            'train_loss': logs.get('loss'),
            'val_loss': logs.get('val_loss')
        })

    def on_train_end(self, logs=None):
        """
        At the end of training, save all logs to the specified Excel file.

        Args:
        - logs (dict): The metrics and losses at the end of training.
        """
        log_df = pd.DataFrame(self.logs)
        log_df.to_excel(self.log_file, index=False)

def train_ann_lin_cv(kf_index, actual_opt, actual_temp, train_inputs, test_inputs,
                     validate_inputs, train_outputs, test_outputs, validate_outputs,
                     num_layers, n_epochs, dense, activation_function, optimizer,
                     plot_iterations=True):
    """
    Train a sequential ANN model and return predictions for train, test, and validate datasets.

    Args:
    - kf_index (int): K-Fold index for cross-validation.
    - actual_opt (str): Actual optimization method used.
    - actual_temp (str): Actual temperature used in the experiment.
    - train_inputs (np.array): Training input features.
    - test_inputs (np.array): Testing input features.
    - validate_inputs (np.array): Validation input features.
    - train_outputs (np.array): Training output labels.
    - test_outputs (np.array): Testing output labels.
    - validate_outputs (np.array): Validation output labels.
    - num_layers (int): Number of hidden layers in the model.
    - n_epochs (int): Number of epochs for training.
    - dense (int): Number of neurons in each hidden layer.
    - activation_function (str): Activation function for the neurons.
    - optimizer (str): Optimizer to use for training the model.
    - plot_iterations (bool): If True, plots the training progress.

    Returns:
    - Tuple containing directory name, predictions and true values for train, test, and validate datasets, along with training statistics and the model.
    """

    # Define the model
    model = Sequential()
    model.add(Dense(dense, activation=activation_function, kernel_initializer='he_normal', input_shape=(train_inputs.shape[1],)))

    for _ in range(num_layers - 1):
        model.add(Dense(dense, activation=activation_function))

    model.add(Dense(1))

    # Choose the optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'nadam':
        opt = Nadam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss='mean_squared_error')

    # Create directory for saving outputs
    dir_name = f'output/{actual_opt}/{actual_temp}/{kf_index}kf_{num_layers}layers_{dense}dense_{n_epochs}epochs_{optimizer}_optimizer_{activation_function}_activation'
    os.makedirs(dir_name, exist_ok=True)

    # Define the EpochLogger and ModelCheckpoint callbacks
    epoch_logger = EpochLogger(log_file=f'{dir_name}/epoch_logs.xlsx')
    start_epoch = int(n_epochs / 10)
    model_checkpoint = CustomModelCheckpoint(start_epoch, filepath=f'{dir_name}/best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)

    # Train the model
    history = model.fit(train_inputs, train_outputs, epochs=n_epochs, verbose=0,
                        validation_data=(validate_inputs, validate_outputs),
                        callbacks=[model_checkpoint, epoch_logger], shuffle=True)

    train_predictions = model.predict(train_inputs).flatten()
    test_predictions = model.predict(test_inputs).flatten()
    validate_predictions = model.predict(validate_inputs).flatten()

    # Clip predictions
    train_predictions = np.clip(train_predictions, a_min=770, a_max=20000000)
    test_predictions = np.clip(test_predictions, a_min=770, a_max=20000000)
    validate_predictions = np.clip(validate_predictions, a_min=770, a_max=20000000)

    # Find the best epochs for validation and training loss
    val_loss_after_start = history.history['val_loss'][start_epoch:]
    min_val_loss = min(val_loss_after_start)
    best_val_epoch = val_loss_after_start.index(min_val_loss) + 1 + start_epoch

    train_loss_after_start = history.history['loss'][start_epoch:]
    min_train_loss = min(train_loss_after_start)
    best_train_epoch = train_loss_after_start.index(min_train_loss) + 1 + start_epoch

    # Plot model architecture
    plot_model(model, to_file=f'{dir_name}/model_architecture.pdf', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file=f'{dir_name}/model_architecture.jpg', show_shapes=True, show_layer_names=True)

    if plot_iterations:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        epoch_range = int(n_epochs * 0.1)
        max_loss = max(max(history.history['loss'][epoch_range:]), max(history.history['val_loss'][epoch_range:]))
        min_loss = min(min(history.history['loss'][epoch_range:]), min(history.history['val_loss'][epoch_range:]))
        plt.ylim(min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss))
        plt.xlim((n_epochs * 0.1), n_epochs)

        train_legend = mlines.Line2D([], [], color='C0', marker='None', markersize=10,
                                     label=f'Min Train Loss: {min_train_loss:.2e}\nEpoch: {best_train_epoch}')
        val_legend = mlines.Line2D([], [], color='C1', marker='None', markersize=10,
                                   label=f'Min Val Loss: {min_val_loss:.2e}\nEpoch: {best_val_epoch}')
        plt.legend(handles=[train_legend, val_legend])
        plt.savefig(f"{dir_name}/iteration_progress.pdf")
        plt.savefig(f"{dir_name}/iteration_progress.jpg")
        plt.close()

    model.load_weights(f'{dir_name}/best_model.h5')

    return dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
        validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model

def train_ann_log_cv(kf_index, actual_opt, actual_temp, train_inputs, test_inputs,
                     validate_inputs, train_outputs, test_outputs, validate_outputs,
                     num_layers, n_epochs, dense, activation_function, optimizer,
                     plot_iterations=True):
    """
    Train a sequential ANN model and return predictions for train, test, and validate datasets.

    Args:
    - kf_index (int): K-Fold index for cross-validation.
    - actual_opt (str): Actual optimization method used.
    - actual_temp (str): Actual temperature used in the experiment.
    - train_inputs (np.array): Training input features.
    - test_inputs (np.array): Testing input features.
    - validate_inputs (np.array): Validation input features.
    - train_outputs (np.array): Training output labels.
    - test_outputs (np.array): Testing output labels.
    - validate_outputs (np.array): Validation output labels.
    - num_layers (int): Number of hidden layers in the model.
    - n_epochs (int): Number of epochs for training.
    - dense (int): Number of neurons in each hidden layer.
    - activation_function (str): Activation function for the neurons.
    - optimizer (str): Optimizer to use for training the model.
    - plot_iterations (bool): If True, plots the training progress.

    Returns:
    - Tuple containing directory name, predictions and true values for train, test, and validate datasets, along with training statistics and the model.
    """

    # Define the model
    model = Sequential()
    model.add(Dense(dense, activation=activation_function, kernel_initializer='he_normal', input_shape=(train_inputs.shape[1],)))

    for _ in range(num_layers - 1):
        model.add(Dense(dense, activation=activation_function))

    model.add(Dense(1))

    # Choose the optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'nadam':
        opt = Nadam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss='mean_squared_logarithmic_error')

    # Create directory for saving outputs
    dir_name = f'output/{actual_opt}/{actual_temp}/{kf_index}kf_{num_layers}layers_{dense}dense_{n_epochs}epochs_{optimizer}_optimizer_{activation_function}_activation'
    os.makedirs(dir_name, exist_ok=True)

    # Define the EpochLogger and ModelCheckpoint callbacks
    epoch_logger = EpochLogger(log_file=f'{dir_name}/epoch_logs.xlsx')
    start_epoch = int(n_epochs / 10)
    model_checkpoint = CustomModelCheckpoint(start_epoch, filepath=f'{dir_name}/best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)

    # Train the model
    history = model.fit(train_inputs, train_outputs, epochs=n_epochs, verbose=0,
                        validation_data=(validate_inputs, validate_outputs),
                        callbacks=[model_checkpoint, epoch_logger], shuffle=True)

    train_predictions = model.predict(train_inputs).flatten()
    test_predictions = model.predict(test_inputs).flatten()
    validate_predictions = model.predict(validate_inputs).flatten()

    # Clip predictions
    train_predictions = np.clip(train_predictions, a_min=770, a_max=20000000)
    test_predictions = np.clip(test_predictions, a_min=770, a_max=20000000)
    validate_predictions = np.clip(validate_predictions, a_min=770, a_max=20000000)

    # Find the best epochs for validation and training loss
    val_loss_after_start = history.history['val_loss'][start_epoch:]
    min_val_loss = min(val_loss_after_start)
    best_val_epoch = val_loss_after_start.index(min_val_loss) + 1 + start_epoch

    train_loss_after_start = history.history['loss'][start_epoch:]
    min_train_loss = min(train_loss_after_start)
    best_train_epoch = train_loss_after_start.index(min_train_loss) + 1 + start_epoch

    # Plot model architecture
    plot_model(model, to_file=f'{dir_name}/model_architecture.pdf', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file=f'{dir_name}/model_architecture.jpg', show_shapes=True, show_layer_names=True)

    if plot_iterations:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        epoch_range = int(n_epochs * 0.1)
        max_loss = max(max(history.history['loss'][epoch_range:]), max(history.history['val_loss'][epoch_range:]))
        min_loss = min(min(history.history['loss'][epoch_range:]), min(history.history['val_loss'][epoch_range:]))
        plt.ylim(min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss))
        plt.xlim((n_epochs * 0.1), n_epochs)

        train_legend = mlines.Line2D([], [], color='C0', marker='None', markersize=10,
                                     label=f'Min Train Loss: {min_train_loss:.2e}\nEpoch: {best_train_epoch}')
        val_legend = mlines.Line2D([], [], color='C1', marker='None', markersize=10,
                                   label=f'Min Val Loss: {min_val_loss:.2e}\nEpoch: {best_val_epoch}')
        plt.legend(handles=[train_legend, val_legend])
        plt.savefig(f"{dir_name}/iteration_progress.pdf")
        plt.savefig(f"{dir_name}/iteration_progress.jpg")
        plt.close()

    model.load_weights(f'{dir_name}/best_model.h5')

    return dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
        validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model


def reset_random_seeds(seed_value=42):
   os.environ['PYTHONHASHSEED']=str(seed_value)
   tf.random.set_seed(seed_value)
   np.random.seed(seed_value)
   random.seed(seed_value)