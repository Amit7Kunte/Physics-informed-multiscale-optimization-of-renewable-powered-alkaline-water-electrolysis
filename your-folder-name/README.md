This folder contains data and source code for generation of data in section 3.0 of the manuscript

Folder 'DNN training' contains code to generate training data and surrogate model training.

ANN_predict_AWE: python script for comparison of CFD output data vs DNN predictions (Fig.4e)

Data_section3.0.xls : consolidated results used in plots in main text Fig.3

iModel_Wgde_1.2_P_5_vin_0.025_dpore_500_dpored_0_Wsep_*_Ls_2000_datan_ANN_valid_inputs: Input .mat files for 'ANN_predict_AWE' containing data extracted from CFD simulations.

train_ann_R2_RMSE_saveCSV: python code for DNN training

preparedData_BoxCox_scaled.mat : MATLAB Data file with DNN training data
trainedANN_seed7084_boxcox_3.keras : Trained AWE CFD model surrogate in python

fitness_unscaled_output*_seed7084: Parity plot data for all 18 outputs from DNN model (Fig. 4d)

DNNval_{output} : excel output file from 'ANN_predict_AWE' with DNN prediction data vs CFD Data
