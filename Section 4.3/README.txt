This folder contains data and source code for generation of data in section 4.3


GA_MOOptim_const_sensitivity_lcoh : python script for global optimization using NSGA2 type genetic algorithm for partial load limit sweep while degradation rate, rated stack efficiency and RES oversize ratio are fixed parameters.

preparedData_BoxCox_scaled.mat : MATLAB Data file with DNN training data used by GA optimizer code for unscaling output predictions from trained surrogate model

trainedANN_seed7084_boxcox_3.keras : Trained AWE CFD model surrogate in python

Data_section4.3.xls : consolidated results used in plots in main text Fig.5

windgen_timeseries : Timseries data of wind power generation

pvgen_timeseries : Timseries data of PV power generation


ga_sens_cdeta{value}_m{value}_nj0.5_PV_rated{value}_gr{value}_lcoh_heater0.05_job{ID}_SECsys_ : MATLAB data file with GA optimizer output with 'cdeta': part-load, 'm':degradation rate param, 'PV_rated' : rated stack efficiency, 'gr': RES oversize ratio parameter values.

combined_cdeta{value}_m{value}_nj0.50_PV{value}_gr{value}.mat : combined matlab data file with fixed parameter value GA optimizer data

pareto_combine : MATLAB code to combine and organize  data from ga_sens_cdeta..... files.


D_metrics_vs_load_m{value}_PV{value}_gr{value}.xlsx : Output excel result file with optimum TEA metrics data calculated using 'GA_2D_param_sens_minmax_lcoh' MATLAB  code.

inputparam_vs_load_m{value}_PV{value}_gr{value}: Output excel result file with optimum input parameters (other than fixed param) data calculated using 'GA_2D_param_sens_minmax_lcoh' MATLAB code.

LCOH_vs_load_m{value}_PV{value}_gr{value}: Output excel result file with the LCOH component data for fixed parameters across the part-load sweep calculated using 'GA_2D_param_sens_minmax_lcoh' MATLAB code.


