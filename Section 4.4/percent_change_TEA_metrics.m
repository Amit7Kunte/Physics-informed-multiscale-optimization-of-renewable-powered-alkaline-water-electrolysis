% Combined sensitivity table across all parameter files
% Output: one sheet with a parameter-name column + wide result columns

folderPath = pwd;
outputFile = 'Global_%change_results_combined.xlsx';

paramMap = struct();
paramMap.sepwidth = struct('selected',[0.1 0.25 0.4 0.485],'opt',0.1);
paramMap.pressure = struct('selected',[1 3 5.571 9.3],'opt',5.571);
paramMap.poresize   = struct('selected',[200 360 500 800],'opt',360);
paramMap.inletvel  = struct('selected',[0.007 0.011 0.03 0.05],'opt',0.011);
paramMap.elecwidth  = struct('selected',[0.7 1 1.3 1.6],'opt',0.7);
paramMap.currdens  = struct('selected',[3000 9000 11460  12500],'opt',11460);

keys = fieldnames(paramMap);
allOut = table();

for k = 1:numel(keys)
    key = keys{k};
    pattern = fullfile(folderPath, ['D_metrics_vs_load_' key '_m*.xlsx']);
    files = dir(pattern);
    if isempty(files)
        warning('No file found for %s', key);
        continue;
    end
    filename = fullfile(files(1).folder, files(1).name);
    T = readtable(filename, 'VariableNamingRule', 'preserve');
    ml = T{:,1};
    varNames = T.Properties.VariableNames(2:end-1);
    idx_opt = find(ml == paramMap.(key).opt, 1, 'first');
    if isempty(idx_opt)
        error('Optimum value not found in file: %s', filename);
    end
    selVals = paramMap.(key).selected(:);
    for s = 1:numel(selVals)
        sel = selVals(s);
        idx_sel = find(ml == sel, 1, 'first');
        if isempty(idx_sel)
            warning('Selected value %g not found in %s', sel, filename);
            continue;
        end
        row = table(string(key), sel, 'VariableNames', {'Parameter','SelectedParam'});
        for i = 1:numel(varNames)
            y = T{:,i+1};
            pct_change = (y(idx_sel) / y(idx_opt) - 1) * 100;
            row.(matlab.lang.makeValidName(varNames{i})) = pct_change;
        end
        allOut = [allOut; row];
    end
end

disp(allOut)
writetable(allOut, outputFile, 'Sheet', 1);