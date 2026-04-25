This folder contains data and source code for generation of Table 3 and Fig. S6 to Fig. S11

GA_MOOptim_param_sensitivity_lcoh : python script for global optimization using NSGA2 type genetic algorithm for fixed input parameter values.

preparedData_BoxCox_scaled.mat : MATLAB Data file with DNN training data used by GA optimizer code for unscaling output predictions from trained surrogate model

trainedANN_seed7084_boxcox_3.keras : Trained AWE CFD model surrogate in python

ga_sens_{parameter}{value}_cdeta0.28_m0.0010_nj0.5_PV_rated75.0_gr3.0_lcoh_job{ID}_SECsys_.mat : MATLAB data file with GA optimizer output with one fixed parameter (currdens: current density, poresize: electrode pore size, pressure : operating pressure, inletvel: inlet velocity, sepwidth : separator width, elecwidth : electrode width) value.

combined_paramSens_{parameter}{value}_cdeta0.280_m0.0010_nj0.50_PV75.0_gr3.0.mat : combined matlab data file with fixed parameter value GA optimizer data

GA_2D_param_sens_minmax_lcoh : MATLAB code to process data from combined... data files.


D_metrics_vs_load_{parameter}_m0.001_PV75.0_gr3.0.xlsx : Output excel result file with optimum TEA metrics data calculated using 'GA_2D_param_sens_minmax_lcoh' MATLAB  code.

inputparam_vs_load_{parameter}_m0.001_PV75.0_gr3.0: Output excel result file with optimum input parameters (other than fixed param) data calculated using 'GA_2D_param_sens_minmax_lcoh' MATLAB code.

Data_parameter_senstivity.xls : consolidated results used in plots in main text and S6 to S11

overall_sens: MATLAB code to calculate the results represented in Table3..

percent_change_TEA_metrics : MATLAB code to calculate the results represented in S6 to S11

windgen_timeseries : Timseries data of wind power generation

pvgen_timeseries : Timseries data of PV power generation