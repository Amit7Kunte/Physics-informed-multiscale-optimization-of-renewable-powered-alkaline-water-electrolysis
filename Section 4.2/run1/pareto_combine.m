clc; clear;
format long g

 %files = dir('ga_sens_cdeta0.*_m0.0000_nj0.5_PV_rated75.0_gr*_lcoh_jo*eta10.mat');
 %files = dir('ga_sens_cdeta0.*_m0.*_nj0.5_PV_rated*_gr*_lcoh_jo*.mat');

 %files = dir('ga_sens_cdeta0.*_m0.0010_nj0.5_PV_rated75.0_gr3.5_lcoh_job9778642_SECsys.mat');
  files = dir('ga_sens_*.mat');                 % all GA sensitivity files


  

  names = {files.name};
% 
% logical index of files that DO NOT contain 'eta10'
keep = cellfun(@(s) isempty(strfind(s,'eta10')), names);

files_use = files(keep);                      % struct array without eta10 files

for k = 1:numel(files_use)
   fname = fullfile(files_use(k).folder, files_use(k).name);
   %fprintf('Loading file: %s\n', fname);
try
    S = load(fname);

    n = size(S.pareto_F,1);   % reference per file
    
    lenOK = [ ...
        size(S.pareto_G,1)            == n, ...
        size(S.pareto_X,1)           == n, ...
        numel(S.pareto_W_sys75)      == n, ...
        numel(S.pareto_LCOE_awe)     == n, ...
        numel(S.pareto_SEC_sta)      == n ...
        % add all other pareto_* used
    ];

if ~all(lenOK)
    warning('Skipping %s: inconsistent pareto_* lengths in this file.', fname);
    continue
end

catch ME
    warning('Skipping file (cannot load): %s\n  Reason: %s', fname, ME.message);
    continue
end

    % your processing here
end

nF = numel(files_use);
params = struct('cdeta',[],'m',[],'nj',[],'PV',[],'gr',[]);
for i = 1:nF
    name = files_use(i).name;
    t = regexp(name,'_cdeta([0-9\.]+)_','tokens','once');  params(i).cdeta = str2double(t{1});
    t = regexp(name,'_m([0-9\.]+)_','tokens','once');      params(i).m     = str2double(t{1});
    t = regexp(name,'_nj([0-9\.]+)_','tokens','once');     params(i).nj    = str2double(t{1});
    t = regexp(name,'_PV_rated([0-9\.]+)_','tokens','once'); params(i).PV  = str2double(t{1});
    t = regexp(name,'_gr([0-9\.]+)_','tokens','once');     params(i).gr    = str2double(t{1});
end

cdeta_v = [params.cdeta]';
m_v     = [params.m]';
nj_v    = [params.nj]';
PV_v    = [params.PV]';
gr_v    = [params.gr]';
Tpar = table(cdeta_v,m_v,nj_v,PV_v,gr_v, ...
    'VariableNames',{'cdeta','m','nj','PV','gr'});

[Tu,~,idxGroup] = unique(Tpar,'rows');
nGroups = height(Tu);

for g = 1:nGroups
    idxFiles = find(idxGroup == g);

    % concatenated arrays
    all_F   = [];
    all_G   = [];
    all_X   = [];
    all_W   = [];
    all_LCOE = [];
    all_curt = [];
    all_LF   = [];
    all_maxT = [];
    all_vapH2 = [];
    all_Nrep = [];
    all_degrate_eff = [];
    all_n_starts =[];
    all_standby_time = [];
    all_BM_refrig = [];
    all_BM_pumps =[];
    all_BM_lyecool = [];
    all_BM_heaters = [];
    all_BM_h2purif = [];
    all_BM_h2compression = [];
    all_BM_gasliqsep = [];
    all_BM_pow = [];
    all_SEC_sta =[];
    all_cd_eta75 =[];
 %   all_LCOE_awe= [];
    all_frac_heating_limited= [];
    all_hours_prod= [];
    all_hours_standby= [];
    all_hours_idle= [];
    all_therm_tau= [];
    all_p_min_standby =[];
    all_frac_standby_ener = [];
    all_SEC_avg_stack =[];    
    all_T_hist=[];
    for k = idxFiles'
    fname = fullfile(files_use(k).folder, files_use(k).name);
    try
        S = load(fname);
    catch ME
        warning('Skipping file (cannot load): %s\nReason: %s\n', fname, ME.message);
        continue
    end

        all_F      = [all_F;      S.pareto_F];
        all_G      = [all_G;      S.pareto_G];
        all_X      = [all_X;      S.pareto_X];
        all_W      = [all_W;      S.pareto_W_sys75(:)];
        all_LCOE   = [all_LCOE;   S.pareto_LCOE_awe(:)];
        all_curt   = [all_curt;   S.pareto_curtail_factor(:)];
        all_LF     = [all_LF;     S.pareto_load_factor(:)];
        all_maxT   = [all_maxT;   S.pareto_maxT(:)];
        all_vapH2  = [all_vapH2;  S.pareto_vap_h2_pdt(:)];
        all_Nrep = [all_Nrep; S.pareto_Nrep(:)];
        all_degrate_eff = [all_degrate_eff; S.pareto_degrate_eff(:)];
        %all_standby_time = [all_standby_time; S.standby_time(:)];
        all_n_starts =[all_n_starts; S.pareto_n_starts(:)];
        all_BM_refrig = [all_BM_refrig;S.BM_refrig(:)];
        all_BM_pumps =[all_BM_pumps;S.BM_pumps(:)];
        all_BM_lyecool = [all_BM_lyecool;S.BM_lyecool(:)];
        all_BM_heaters = [all_BM_heaters;S.BM_heaters(:) ];
        all_BM_h2purif = [all_BM_h2purif;S.BM_h2purif(:)];
        all_BM_h2compression = [all_BM_h2compression;S.BM_h2compression(:)];
        all_BM_gasliqsep = [all_BM_gasliqsep ; S.BM_gasliqsep(:)];
        all_BM_pow = [all_BM_pow; S.BM_pow(:)];
        all_SEC_sta = [all_SEC_sta; S.pareto_SEC_sta(:)];
        all_cd_eta75 = [all_cd_eta75; S.pareto_cd_eta75(:)];
        all_frac_heating_limited = [all_frac_heating_limited; S.pareto_frac_heating_limited(:) ];
        all_hours_prod = [all_hours_prod ; S.pareto_hours_prod(:)];
        all_hours_standby= [all_hours_standby ;S.pareto_hours_standby(:)];
        all_hours_idle = [all_hours_idle ; S.pareto_hours_idle(:) ];
        all_therm_tau = [all_therm_tau ; S.pareto_therm_tau(:) ];
        all_p_min_standby =[all_p_min_standby ; S.pareto_p_min_standby(:)];
        all_frac_standby_ener = [all_frac_standby_ener ; S.pareto_frac_standby_ener(:)];
        all_T_hist = [all_T_hist ; S.pareto_T_hist]; 
        %if isfield(S, 'pareto_SEC_avg_stack')
        try
            all_SEC_avg_stack = [all_SEC_avg_stack ; S.pareto_SEC_avg_stack(:)];
        catch ME
             warning('Could not append pareto_SEC_avg_stack: %s', ME.message);
        end

        %end

    end

    % global Pareto indices
    idxP = find_pareto(all_F);

    % select combined-front data
    pareto_F             = all_F(idxP,:);
    pareto_G             = all_G(idxP,:);
    pareto_X             = all_X(idxP,:);
    pareto_W_sys75       = all_W(idxP);
    pareto_LCOE_awe      = all_LCOE(idxP);
    pareto_curtail_factor= all_curt(idxP);
    pareto_load_factor   = all_LF(idxP);
    pareto_maxT          = all_maxT(idxP);
    pareto_vap_h2_pdt    = all_vapH2(idxP);
    pareto_Nrep          = all_Nrep(idxP);
    pareto_degrate_eff   = all_degrate_eff(idxP);

    %pareto_standby_time  = all_standby_time(idxP);
    pareto_n_starts         = all_n_starts(idxP);
    pareto_BM_refrig        = all_BM_refrig(idxP);
    pareto_BM_pumps         = all_BM_pumps(idxP);
    pareto_BM_lyecool       = all_BM_lyecool(idxP);
    pareto_BM_heaters       = all_BM_heaters(idxP);
    pareto_BM_h2purif       = all_BM_h2purif(idxP);
    pareto_BM_h2compression     = all_BM_h2compression(idxP);
    pareto_BM_gasliqsep         = all_BM_gasliqsep(idxP);
    pareto_BM_pow               = all_BM_pow(idxP);
    pareto_SEC_sta              = all_SEC_sta(idxP);
    pareto_cd_eta75             = all_cd_eta75(idxP);
    pareto_frac_heating_limited = all_frac_heating_limited(idxP);
    pareto_hours_prod           = all_hours_prod(idxP);
    pareto_hours_standby        = all_hours_standby(idxP);
    pareto_hours_idle           = all_hours_idle(idxP);
    pareto_therm_tau            = all_therm_tau(idxP);
    pareto_p_min_standby        = all_p_min_standby(idxP);
    pareto_frac_standby_ener    = all_frac_standby_ener(idxP);
    % if isfield(S, 'pareto_SEC_avg_stack')
    pareto_SEC_avg_stack    = all_SEC_avg_stack(idxP);  
    pareto_T_hist = all_T_hist(idxP, :);
    % end

   % k
    outName = sprintf('combined_cdeta%.3f_m%.4f_nj%.2f_PV%.1f_gr%.1f.mat', ...
                      Tu.cdeta(g), Tu.m(g), Tu.nj(g), Tu.PV(g), Tu.gr(g));
vars = { ...
    'pareto_F','pareto_G','pareto_X', ...
    'pareto_W_sys75','pareto_LCOE_awe', ...
    'pareto_curtail_factor','pareto_load_factor', ...
    'pareto_maxT','pareto_vap_h2_pdt','Tu', ...
    'pareto_Nrep','pareto_degrate_eff','pareto_n_starts', ...
    'pareto_BM_refrig','pareto_BM_pumps','pareto_BM_lyecool', ...
    'pareto_BM_heaters','pareto_BM_h2purif','pareto_BM_h2compression', ...
    'pareto_BM_gasliqsep','pareto_BM_pow','pareto_SEC_sta', ...
    'pareto_cd_eta75', ...
    'pareto_frac_heating_limited','pareto_hours_prod', ...
    'pareto_hours_standby','pareto_hours_idle','pareto_therm_tau', ...
    'pareto_p_min_standby','pareto_frac_standby_ener', 'pareto_T_hist'...
};

% Only add this variable if it exists in the workspace
% if exist('pareto_SEC_avg_stack','var')
     vars{end+1} = 'pareto_SEC_avg_stack';
% end

save(outName, vars{:});

end


function idxP = find_pareto(F)
    n = size(F,1);
    isP = true(n,1);
    for i = 1:n
        if ~isP(i), continue; end
        dom = all(bsxfun(@le,F,F(i,:)),2) & any(bsxfun(@lt,F,F(i,:)),2);
        isP(dom) = false;
    end
    idxP = find(isP);
end


