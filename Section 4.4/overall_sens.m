
% Batch sensitivity analysis for all D_metrics files with parameter keywords
% Writes one consolidated Excel sheet with one row per parameter keyword

folderPath = pwd;   % change if needed
keywords = {'sepwidth','pressure','poresize','inletvel','elecwidth','currdens'};
outputFile = 'Table3_sensitivity_results_allparams.xlsx';

allResults = table();

for k = 1:numel(keywords)
    key = keywords{k};
    pattern = fullfile(folderPath, ['D_metrics_vs_load_' key '_m*.xlsx']);
    files = dir(pattern);
    
    if isempty(files)
        warning('No file found for keyword: %s', key);
        continue;
    elseif numel(files) > 1
        warning('Multiple files found for keyword: %s. Using first file: %s', key, files(1).name);
    end
    
    filename = fullfile(files(1).folder, files(1).name);
    T = readtable(filename, 'VariableNamingRule', 'preserve');
    
    ml = T{:,1};
    varNames = T.Properties.VariableNames(2:end-1);
    ml_min = min(ml);
    ml_max = max(ml);
    idx_min = find(ml == ml_min, 1, 'first');
    idx_max = find(ml == ml_max, 1, 'first');
    ml_avg = mean(ml, 'omitnan');
    denom = (ml_max - ml_min) / ml_avg;
    
    n = numel(varNames);
    sensVals = nan(1,n);
    
    for i = 1:n
        y = T{:,i+1};
        y_min = y(idx_min);
        y_max = y(idx_max);
        y_avg = mean(y, 'omitnan');
        sensVals(i) = ((y_max - y_min) / y_avg) / denom;
    end
    
    rowTable = array2table(sensVals, 'VariableNames', matlab.lang.makeValidName(varNames));
    rowTable = addvars(rowTable, string(key), 'Before', 1, 'NewVariableNames', 'Parameter');
    
    allResults = [allResults; rowTable]; %#ok<AGROW>
end

disp(allResults)
writetable(allResults, outputFile, 'Sheet', 1);