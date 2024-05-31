This repository contains a set of scripts and data files for training
and optimizing Artificial Neural Network (ANN) models to predict asphalt
fatigue life. The main components are divided into data files, auxiliary
functions, and scripts for hyperparameter optimization and model
training. The results of the model training are visualized using
additional scripts. The scripts are tested on data taken from a 4PB
testing machine with a temperature of the test equal to 20°C and a
frequency of loading the tested specimen equal to 10Hz.

**Data Files**

-   **excel_spreadsheets/asphalt_20_deg.xlsx**: This dataset contains
    all the initial data. It is crucial to keep the formatting unchanged
    and only modify the data.

-   **excel_spreadsheets/hyperparameters.xlsx**: This file contains
    hyperparameter settings, divided into two lists for linear and
    logarithmic loss functions.

**Auxiliary Functions (non-runnable)**

-   **aux_functions/excel_modifications.py**: Contains functions for
    modifying, filtering, and handling data from **asphalt_20_deg.xlsx**
    to prepare it for machine learning models.

-   **aux_functions/print_automatic.py**: Script for printing parts of
    the results.

-   **aux_functions/train_ml_model.py**: Defines functions for training
    ANN models.

**Hyperparameter Optimization Scripts**

-   **hyp_optimalization_linear.py**: Uses auxiliary functions to
    determine a linear loss function\'s best combination of
    hyperparameters.

-   **hyp_optimalization_logarithmic.py**: Uses auxiliary functions to
    determine the best combination of hyperparameters for a logarithmic
    loss function.

After running these scripts, results are saved in:

-   **excel_spreadsheets/lin20_deg_hyperparameter_tuning_results.xlsx**
    (for linear loss function)

-   **excel_spreadsheets/log20_deg_hyperparameter_tuning_results.xlsx**
    (for logarithmic loss function)

Hyperparameter sets are evaluated by the R2 score. The best combination
of hyperparameters can be determined manually or by using the following
graphical scripts in the **printing_outputs** directory:

-   **hyperparameters-activation_and_optimizer.py**

-   **best_model_plot.py**

**Model Training Scripts**

-   **train_model_linear.py**: Uses the best hyperparameters to train
    the model with a linear loss function.

-   **train_model_logarithmic.py**: Uses the best hyperparameters to
    train the model with a logarithmic loss function.

**Note:**

Set the exact hyperparameter values on line 68 of these scripts before
running the training.

**Output**

The output of the final model training is saved in the **output**
directory. The files generated depend on the loss function and
hyperparameters used during training.

**Visualization and Interpretation**

To get the results, these functions must be started. Ensure that each
function uses the correct input files for the best model.

-   **printing_outputs/iteration_process.py**: Prints the learning
    process and trains the model side by side for linear and logarithmic
    loss errors.

-   **printing_outputs/pdp_plot.py**: Generates a partial dependence
    plot with a colormap indicating the predicted number of cycles and
    white triangles representing input data. It plots two graphs side by
    side for different levels of used strain.

-   **printing_outputs/print_by_authors.py**: Plots the ability to
    predict true vs. predicted values for linear and logarithmic models.
    Points on the diagonal are predicted correctly.

**\
**

**Step-by-Step Instructions**

1.  **Prepare Data:**

    -   Edit your own **excel_spreadsheets/asphalt_20_deg.xlsx** file
        filled with data.

    -   Update **excel_spreadsheets/hyperparameters.xlsx** with your
        desired combinations of hyperparameters for lists labelled as
        \'lin\' (linear) or \'log\' (logarithmic).

2.  **Hyperparameter Optimization:**

    -   Run either **hyp_optimalization_linear.py** or
        **hyp_optimalization_logarithmic.py**. Depending on the size of
        your dataset, adjust the number of folds, line 75, **n_splits**
        (the number of data subsets used for testing and training). This
        division into folds ensures a robust evaluation by mitigating
        the effect of a lucky dataset split.

    -   After completing the hyperparameter optimization, the results
        will be saved in
        **excel_spreadsheets/lin20_deg_hyperparameter_tuning_results.xlsx**
        and
        **excel_spreadsheets/log20_deg_hyperparameter_tuning_results.xlsx**.

3.  **Evaluate Models:**

    -   In these result files, you will find the performance metrics of
        all trained models. The key metric used for evaluation is the R2
        score; a higher R2 score indicates a more accurate model.

    -   Choose the best model manually by considering the average R2
        score across all folds for each set of hyperparameters.
        Alternatively, use the scripts
        **printing_outputs/best_model_plot.py** and
        **printing_outputs/hyperparameters-activation_and_optimizer.py**
        to visualize and identify the best combination of activation
        functions, optimizers, number of layers, and neuron densities.

4.  **Update Visualization Scripts:**

    -   In **printing_outputs/best_model_plot.py**, set the variable
        **actual_opt** on line 19 to \'lin\' or \'log\', depending on
        the model you trained.

    -   In
        **printing_outputs/hyperparameters-activation_and_optimizer.py**,
        manually update the values on lines 9 and 10 (activation
        functions and optimizers from **hyperparameters.xlsx**) and
        lines 13 and 14 (average R2 scores corresponding to activation
        functions and optimizers, from previous step of optimalization).

5.  **Train the Final Model:**

    -   Use the best hyperparameters to train your model by updating
        line 68 in **train_model_linear.py** or
        **train_model_logarithmical.py** with the chosen values.

    -   After training, the results will be saved in
        **output/lin/20_deg** or **output/log/20_deg** with a filename
        format of
        **Xlayers_Xdense_Xepochs_XXX_optimizer_XXX_activation**.

6.  **Interpret and Visualize Results:**

    -   Use the generated files for the next steps:

        -   Update the file path in
            **printing_outputs/iteration_process.py** on lines 14 and 15
            to visualize the learning process of the models. This script
            compares linear and logarithmic models.

        -   Update the file path in **printing_outputs/pdp_plot.py** on
            lines 11 and 16 to generate partial dependence plots.

        -   Update the file path in
            **printing_outputs/print_by_authors.py** on lines 9 and 10
            to plot true vs. predicted values.

7.  **Graphical Outputs:**

    -   All graphical outputs will be saved in the **output/figures**
        folder.

Following these steps, you can efficiently optimize, train, evaluate,
and visualize machine learning models for predicting asphalt fatigue
loading cycles.
