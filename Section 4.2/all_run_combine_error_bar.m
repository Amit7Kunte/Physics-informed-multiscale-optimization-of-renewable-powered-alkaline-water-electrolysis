clc; clear;
format long g

baseDir = pwd;
runFolders = {'run1','run2','run3','run4','run5'};
nRuns = numel(runFolders);

LCOH_all = {};
MET_all = {};
INP_all = {};

for r = 1:nRuns
    runPath = fullfile(baseDir, runFolders{r});
    if ~isfolder(runPath)
        error('Folder not found: %s', runPath);
    end

    fL = dir(fullfile(runPath, 'LCOH_vs_load*.xlsx'));
    fM = dir(fullfile(runPath, 'D_metrics_vs_load*.xlsx'));
    fI = dir(fullfile(runPath, ' inputparam_vs_load*.xlsx'));

    if isempty(fL) || isempty(fM) || isempty(fI)
        error('Missing one or more result files in folder: %s', runPath);
    end

    [~,ix] = sort({fL.name}); fL = fL(ix);
    [~,ix] = sort({fM.name}); fM = fM(ix);
    [~,ix] = sort({fI.name}); fI = fI(ix);

    LCOH_all{r} = readtable(fullfile(runPath, fL(1).name));
    MET_all{r}  = readtable(fullfile(runPath, fM(1).name));
    INP_all{r}  = readtable(fullfile(runPath, fI(1).name));
end

baseL = LCOH_all{1};
baseM = MET_all{1};
baseI = INP_all{1};

for r = 2:nRuns
    if ~isequal(baseL.Properties.VariableNames, LCOH_all{r}.Properties.VariableNames)
        error('LCOH headers differ in run %d', r);
    end
    if ~isequal(baseM.Properties.VariableNames, MET_all{r}.Properties.VariableNames)
        error('Metric headers differ in run %d', r);
    end
    if ~isequal(baseI.Properties.VariableNames, INP_all{r}.Properties.VariableNames)
        error('Input headers differ in run %d', r);
    end
    if height(baseL) ~= height(LCOH_all{r}) || height(baseM) ~= height(MET_all{r}) || height(baseI) ~= height(INP_all{r})
        error('Row count mismatch in run %d', r);
    end
end

numColsL = width(baseL) - 1;
numColsM = width(baseM) - 1;
numColsI = width(baseI) - 1;

LCOH_mat = nan(height(baseL), numColsL, nRuns);
MET_mat  = nan(height(baseM), numColsM, nRuns);
INP_mat  = nan(height(baseI), numColsI, nRuns);

for r = 1:nRuns
    LCOH_mat(:,:,r) = table2array(LCOH_all{r}(:,2:end));
    MET_mat(:,:,r)  = table2array(MET_all{r}(:,2:end));
    INP_mat(:,:,r)  = table2array(INP_all{r}(:,2:end));
end

LCOH_mean = mean(LCOH_mat, 3, 'omitnan');
MET_mean  = mean(MET_mat, 3, 'omitnan');
INP_mean  = mean(INP_mat, 3, 'omitnan');

LCOH_mean_tbl = baseL(:,1);
LCOH_mean_tbl{:,2:width(baseL)} = LCOH_mean;
MET_mean_tbl = baseM(:,1);
MET_mean_tbl{:,2:width(baseM)} = MET_mean;
INP_mean_tbl = baseI(:,1);
INP_mean_tbl{:,2:width(baseI)} = INP_mean;

[~, idxMin] = min(LCOH_mean(:,1));
minRow_LCOH = LCOH_mean_tbl(idxMin,:);

minRow_INP  = INP_mean_tbl(idxMin,:);
optRow_MET = MET_mean_tbl(idxMin, :);
MET_dev_tbl = MET_mean_tbl;



MET_dev_tbl{:, 2:end} = (MET_mean_tbl{:, 2:end} ./ optRow_MET{1, 2:end} - 1) * 100;
MET_dev_tbl.Properties.VariableNames = {'min_load','M_H2','TIC','repl_cost','LF','SEC_sys','LCOE'};
INP_mean_tbl.Properties.VariableNames = {'min_load','On-Off cycles*0.01','Delta_MaxT (K)','elec_width (mm)','Pressure (bar)','Inlet velocity*100 m/s','Pore size*0.01 um','separator width*5 mm','current density * 0.001 A/m2'};
LCOH_mean_tbl.Properties.VariableNames = {'min_load','LCOH_mean','Spec. Energy_mean','Spec. capex_mean'};
%MET_dev_tbl.Properties.VariableNames(2:end) = strcat(MET_mean_tbl.Properties.VariableNames(2:end), '_devPct');

LCOH_vec = squeeze(LCOH_mat(:,1,:));   % rows x runs
LCOH_min = min(LCOH_vec, [], 2, 'omitnan');
LCOH_max = max(LCOH_vec, [], 2, 'omitnan');
LCOH_range_tbl = table(LCOH_mean_tbl{:,1}, LCOH_mean_tbl{:,2}, LCOH_min, LCOH_max, ...
    'VariableNames', {'min_load','LCOH_mean','LCOH_min','LCOH_max'});


outDir = fullfile(pwd, 'consolidated');
if ~isfolder(outDir), mkdir(outDir); end

writetable(MET_dev_tbl, fullfile(outDir, 'TE_metrics_percent_deviation_from_opt.xlsx'));
writetable(LCOH_range_tbl, fullfile(outDir, 'LCOH_min_max_across_runs.xlsx'));
writetable(INP_mean_tbl, fullfile(outDir, 'INput_param_mean_runs.xlsx'));
writetable(LCOH_mean_tbl, fullfile(outDir, 'LCOH_components_mean_tbl.xlsx'));

fprintf('Done. Minimum mean LCOH at row %d, min_load = %.6g\n', idxMin, minRow_LCOH{1,1});