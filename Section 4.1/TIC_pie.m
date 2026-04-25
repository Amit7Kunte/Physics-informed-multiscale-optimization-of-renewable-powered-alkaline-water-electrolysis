% Load data
clc;clear; close all;
load('preparedData_TIC_pie.mat');  % adjust path if needed

% 1) Construct data vector (in the order you want wedges)
pieVals = [
    IC_comp(1)
    IC_comp(2)
    IC_comp(3)
];

% 2) Construct corresponding labels (short, Origin-friendly)
pieLabels = {
    'Contingency'
    'Aux_cost'
    'BM'

};

% 3) Quick check pie in MATLAB (optional)
figure;
pie(pieVals, pieLabels);
title('CAPEX breakdown (check only, final plot in Origin)');

% 4) Export values + labels to CSV for Origin
T = table(pieLabels, pieVals, 'VariableNames', {'Label','Value'});
writetable(T, 'TIC_pie_data.csv');  % Origin can import this directly
