This repository contains a set of scripts and data files for training and optimizing Artificial Neural Network (ANN) models to predict asphalt fatigue life. The main components are divided into data files, auxiliary functions, and scripts for hyperparameter optimization and model training. The results of the model training are visualized using additional scripts. The scripts are tested on data taken from a 4PB testing machine with a temperature of the test equal to 20°C and a frequency of loading the tested specimen equal to 10Hz.

Data Files

excel_spreadsheets/asphalt_20_deg.xlsx: This dataset contains all the initial data. It is crucial to keep the formatting unchanged and only modify the data.

excel_spreadsheets/hyperparameters.xlsx: This file contains hyperparameter settings, divided into two lists for linear and logarithmic loss functions.

Auxiliary Functions (non-runnable)

aux_functions/excel_modifications.py: Contains functions for modifying, transforming, filtering, and handling data from asphalt_20_deg.xlsx to prepare it for machine learning models.

aux_functions/metrics_calculation.py: Defined functions for metrics evaluation.

aux_functions/train_ml_model.py: Defines functions for training ANN models.

Hyperparameter Optimization Scripts

hyp_optimalization_linear.py: Uses auxiliary functions to determine a linear loss function's best combination of hyperparameters.

hyp_optimalization_logarithmic.py: Uses auxiliary functions to determine the best combination of hyperparameters for a logarithmic loss function.

After running these scripts, results are saved in:

excel_spreadsheets/lin20_deg_hyperparameter_tuning_results.xlsx (for linear loss function)

excel_spreadsheets/log20_deg_hyperparameter_tuning_results.xlsx (for logarithmic loss function)

Hyperparameter sets are evaluated by the R2 score. The best combination of hyperparameters can be determined manually or by using the following graphical scripts in the printing_outputs directory:

hyperparameters-activation_and_optimizer.py

best_model_plot.py

Model Training Scripts

After discovery of proper hyperparameters for individual dataset The model should be trained again with just this hyperparameter or use results from hyperparameter optimalization process.  

Output

The output of the final model training is saved in the output directory. The files generated depend on the loss function and hyperparameters used during training.

Visualization and Interpretation

To get the results, these functions must be started. Ensure that each function uses the correct input files for the best model.

printing_outputs/iteration_process.py: Prints the learning process and trains the model side by side for linear and logarithmic loss errors.

printing_outputs/pdp.py: Generates a partial dependence plot with a colormap indicating the predicted number of cycles and white triangles representing input data. Printing area is a convex hull around training data. It plots two graphs side by side for different levels of used strain.

printing_outputs/pdp_2d.py: Generates a partial dependence plot in 2D scale with a colored lines indicating the predicted number of cycles for each Air Void content. Printing area is a convex hull around training data.

printing_outputs/true_vs_predicted.py: Plots the ability to predict true vs. predicted values for chosen model. Points show only the testing dataset. Each color is assigned for author of the data. Points on the diagonal are predicted correctly.

**
**

Step-by-Step Instructions

Prepare Data:

Edit your own excel_spreadsheets/asphalt_20_deg.xlsx file filled with data.
  
Test normality by printing_outputs/normality_test.py and printing_outputs/pair_plot_feature_analysis.py. These outcomes shows if the data are normally distributed and if is it possible to filter it by using z-score function. (before this process hyp_optimalization_linear.py or hyp_optimalization_logarithmic.py needs to be started to get data preprocesing steps. After these steps it is possible to make an analysis.)

Update excel_spreadsheets/hyperparameters.xlsx with your desired combinations of hyperparameters for lists labelled as 'lin' (linear) or 'log' (logarithmic).

Hyperparameter Optimization:

Run either hyp_optimalization_linear.py or hyp_optimalization_logarithmic.py. Depending on the size of your dataset, adjust the number of folds, line 108, n_splits (the number of data subsets used for testing and training). This division into folds ensures a robust evaluation by mitigating the effect of a lucky dataset split.


After completing the hyperparameter optimization, the results will be saved in excel_spreadsheets/lin20_deg_hyperparameter_tuning_results.xlsx and excel_spreadsheets/log20_deg_hyperparameter_tuning_results.xlsx.

Evaluate Models:

In these result files, you will find the performance metrics of all trained models. The key metric used for evaluation is the R2 score; a higher R2 score indicates a more accurate model.

Choose the best model manually by considering the average R2 score across all folds for each set of hyperparameters. Alternatively, use the scripts printing_outputs/best_model_plot.py and printing_outputs/hyperparameters-activation_and_optimizer.py to visualize and identify the best combination of activation functions, optimizers, number of layers, and neuron densities.

Update Visualization Scripts:

In printing_outputs/best_model_plot.py, set the variable actual_opt on line 19 to 'lin' or 'log', depending on the model you trained.

In printing_outputs/hyperparameters-activation_and_optimizer.py, manually update the values on lines 10 and 11 (activation functions and optimizers from hyperparameters.xlsx) and lines 14 and 15 (average R2 scores corresponding to activation functions and optimizers, from previous step of optimalization).

Train the Final Model:

After training and finding best haperparameters, the results will be saved in output/lin/20_deg or output/log/20_deg with a filename format of Xlayers_Xdense_Xepochs_XXX_optimizer_XXX_activation.

Interpret and Visualize Results:

Use the generated files for the next steps:

Update the file path in printing_outputs/iteration_process.py on lines 31 and 33 to visualize the learning process of the models. This script compares linear and logarithmic models.

Update the file path in printing_outputs/pdp.py on lines 17 and 20 to generate partial dependence plots.

Update the file path in printing_outputs/pdp_2d.py on lines 13 and 16 to generate partial dependence plots. Aditionally update lines 23-35 to set propper binder content ranges for desired air voids levels. 

Update the file path in printing_outputs/true_vs_prediction.py on lines 2 to plot true vs. predicted values.

Graphical Outputs:

All graphical outputs will be saved in the output/figures folder.
Following these steps, you can efficiently optimize, train, evaluate, and visualize machine learning models for predicting asphalt fatigue loading cycles.