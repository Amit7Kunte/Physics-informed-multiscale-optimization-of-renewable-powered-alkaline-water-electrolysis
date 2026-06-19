%% User settings
% List all result files (you can auto-generate this or paste them)
% filesList = { ...
%     'ga_sens_cdeta0.10_....mat', ...
%     'ga_sens_cdeta0.15_....mat', ...
%     'ga_sens_cdeta0.20_....mat' ...
% };
clc;
clear;
%close all
pvgen  = readmatrix('pvgen_profile.csv');      % 8760x1
windgen = readmatrix('windgen_profile.csv');   % 8760x1


MH2=0.002;
filesList = dir('combined_cdeta0.*_m0.0010_nj1.00_PV75.0_gr3.0.mat');
nF = numel(filesList);
base_degrate = 0.125*8.760;
%filesList = dir('ga_sens_cdeta0.100_m0.0000_nj0.5_PV_rated75.0_gr3.0_lcoh_heater*_.mat');
%genratio_vals = nan(1,nF);
% Logical mask: 1 = include file, 0 = ignore
useMask = true(1, numel(filesList));   % include all by default
  % example: use first two files only
for i = 1:length(filesList)
    if ~useMask(i), continue; end
    tokens = regexp(filesList(i).name,'_cdeta([0-9\.]+)_','tokens');

    turndown_vals(i) = str2double(tokens{1})*100;
    t2 = regexp(filesList(i).name,'_gr([0-9\.]+)\.mat','tokens','once');
    genratio_vals(i) = str2double(t2{1});

end
[turndown_vals, sortIdx] = sort(turndown_vals);

filesList = filesList(sortIdx);
useMask   = useMask(sortIdx);
genratio = mean(genratio_vals);
%% Load and concatenate data
all_F  = [];
all_X  = [];
all_LCOE  = [];
all_curt  = [];
all_maxT  = [];
file_id   = [];   % to remember which file each row came from
all_loadfactor = [];
all_W_sys75 = [];
all_td = [];
all_SEC_sta=[];
all_cd_eta75 = [];
all_curtail_factor= [];
all_degrate_eff= [];
all_frac_heating_limited = [];
all_hours_idle= [];
all_hours_prod= [];
all_hours_standby= [];
all_load_factor= [];
all_maxT= [];
all_n_starts= [];
all_therm_tau= [];
all_vap_h2_pdt= [];
all_p_min_standby =[];
all_p_min=[];
all_frac_standby_ener = [];
all_heater_rating =[];
all_Nrep = [];
all_BM_pow = [ ];
all_BM_refrig = [];
all_BM_pumps = [];
all_BM_lyecool = [];
all_BM_heaters = [];
all_BM_h2purif = [];
all_BM_h2compression = [];
all_BM_gasliqsep = [];
all_SEC_sta = [ ];
all_Wsys75=[];
all_SEC_avg_stack=[];
all_T_hist=[];
%all_cd_eta75 =[];
%all_BM_heaters =[];
for k = 1:numel(filesList)
    if ~useMask(k), continue; end

  S= load(filesList(k).name);   % assumes variables exist in each .mat

    % Basic checks (optional)
    if ~isfield(S,'pareto_F'),  error('pareto_F missing in %s', filesList{k}), end
    if ~isfield(S,'pareto_X'),  error('pareto_X missing in %s', filesList{k}), end

    nK = size(S.pareto_F,1);

    all_F     = [all_F;     S.pareto_F];
    all_X     = [all_X;     S.pareto_X];
    all_LCOE  = [all_LCOE;  S.pareto_LCOE_awe(:)];
    all_curt  = [all_curt;  S.pareto_curtail_factor(:)];
    all_maxT  = [all_maxT;  S.pareto_maxT(:)];
    all_loadfactor = [all_loadfactor;  S.pareto_load_factor(:)];
    all_W_sys75 = [all_W_sys75;  S.pareto_W_sys75(:)];
    file_id   = [file_id;   k*ones(nK,1)];
    all_td   = [all_td;   turndown_vals(k)*ones(nK,1)];
  %  all_heater_rating =  [all_heater_rating ; heater_vals(k)*ones(nK,1)];
    all_cd_eta75 = [all_cd_eta75; S.pareto_cd_eta75(:)];
    all_curtail_factor= [all_curtail_factor;S.pareto_curtail_factor(:) ];
    all_degrate_eff= [all_degrate_eff ; S.pareto_degrate_eff(:)];
    all_frac_heating_limited = [all_frac_heating_limited ; S.pareto_frac_heating_limited(:)];
    all_hours_idle= [all_hours_idle ; S.pareto_hours_idle(:)];
    all_hours_prod= [all_hours_prod ; S.pareto_hours_prod(:)];
    all_hours_standby= [all_hours_standby ; S.pareto_hours_standby(:)];
    all_load_factor= [all_load_factor ;S.pareto_load_factor(:)];
    %all_maxT= [ all_maxT; S.pareto_maxT];
    all_n_starts= [all_n_starts ; S.pareto_n_starts(:)];
    all_therm_tau= [all_therm_tau; S.pareto_therm_tau(:)];
    all_vap_h2_pdt= [all_vap_h2_pdt ;  S.pareto_vap_h2_pdt(:)];
    all_p_min_standby =[all_p_min_standby ; S.pareto_p_min_standby(:)];
    all_p_min = [all_p_min ; S.pareto_p_min(:)];
    all_frac_standby_ener = [all_frac_standby_ener ; S.pareto_frac_standby_ener(:)];
    all_SEC_sta = [all_SEC_sta ; S.pareto_SEC_sta(:)];
    all_SEC_sys = (all_F(:,1))./((all_LCOE));
    all_Nrep = [all_Nrep ; S.pareto_Nrep(:)];
   % all_BM_heaters = [all_BM_heaters ; S.BM_heaters(:)/10^6]; % in $ Million
    all_BM_pow = [all_BM_pow ; S.pareto_BM_pow(:)];
    all_BM_refrig = [all_BM_refrig ; S.pareto_BM_refrig(:)];
    all_BM_pumps = [all_BM_pumps ; S.pareto_BM_pumps(:)];
    all_BM_lyecool = [all_BM_lyecool ; S.pareto_BM_lyecool(:)];
    all_BM_heaters = [all_BM_heaters ; S.pareto_BM_heaters(:)];
    all_BM_h2purif = [all_BM_h2purif ; S.pareto_BM_h2purif(:)];
    all_BM_h2compression = [all_BM_h2compression ; S.pareto_BM_h2compression(:)];
    all_BM_gasliqsep = [all_BM_gasliqsep ; S.pareto_BM_gasliqsep(:)];
    all_Wsys75 = [all_Wsys75 ; S.pareto_W_sys75(:)];
    all_SEC_avg_stack = [all_SEC_avg_stack ; S.pareto_SEC_avg_stack(:)];
    all_T_hist=[all_T_hist  ; S.pareto_T_hist];
    %all_cd_eta75 = [all_cd_eta75 ; S.pareto_cd_eta75(:)];
    
end

fprintf('Total candidates loaded: %d\n', size(all_F,1));
%% Global Pareto front over selected files
%isPareto = identify_pareto(all_F);   % all objectives are minimization here
F_P      = all_F ;%(isPareto,:);
X_P      = all_X ;%(isPareto,:);
LCOE_P   = all_LCOE ;%(isPareto);
curt_P   = all_curt ;%(isPareto);
maxT_P   = all_maxT ;%(isPareto);
fileP    = file_id ;% (isPareto);

energy_cost = all_F(:,1);   % $/kg
capex_cost  = all_F(:,2);   % $/kg
td          = all_td;       % %
%heater_rating = all_heater_rating;
LCOH_all = energy_cost + capex_cost;



all_Wgde = X_P(:,1);
all_P = X_P(:,2);
all_vin=X_P(:,3);
all_pore = X_P(:,4);
all_sepwidth = X_P(:,5);
all_cd =  X_P(:,6);
td_unique = unique(td);
%heater_unique = unique(heater_rating);
n = numel(td_unique);
%n = numel(heater_unique);
LCOH_min   = nan(n,1);
ener_ratio = nan(n,1);
idxg_vec   = nan(n,1);
LCOE_min    = nan(n,1);
curt_min    = nan(n,1);
maxT_min    = nan(n,1);
cd_eta75_min = nan(n,1);
curtail_factor_min = nan(n,1);
degrate_eff_min    = nan(n,1);
frac_heat_lim_min  = nan(n,1);
hours_idle_min     = nan(n,1);
hours_prod_min     = nan(n,1);
hours_standby_min  = nan(n,1);
load_factor_min    = nan(n,1);
n_starts_min       = nan(n,1);
therm_tau_min      = nan(n,1);
vap_h2_pdt_min     = nan(n,1); % annual kg production rate of H2 from Elec
p_min_standby_min     = nan(n,1);
p_min_min = nan(n,1);
frac_standby_ener_min     = nan(n,1);
SEC_sta_min = nan(n,1);
Nrep_min = nan(n,1);
BM_pow_min = nan(n,1);
BM_refrig_min = nan(n,1);
BM_pumps_min = nan(n,1);
BM_lyecool_min = nan(n,1);
BM_heaters_min = nan(n,1);
BM_h2purif_min = nan(n,1);
BM_h2compression_min = nan(n,1);
BM_gasliqsep_min = nan(n,1);
W_sys75_min = nan(n,1);
cd_eta75_min = nan(n,1);
SEC_avg_stack = nan(n,1);
T_hist = nan(n,1);
%dur_stack =nan(n,1);
%Lmin = 7;%min(LCOH_all);
%Lmax = max(LCOH_all);
for j = 1:n
    mask = (td == td_unique(j));

    [~, idx_local] = min(LCOH_all(mask));   % index within this td level
    idxg_list = find(mask);
    idxg = idxg_list(idx_local);                 % global index

    idxg_vec(j)   = idxg;
    LCOH_min(j)   = LCOH_all(idxg);
    ener_ratio(j) = energy_cost(idxg) / capex_cost(idxg);

    % grab all other variables at this same min-LCOH point
    param_min(j,:)          = X_P(idxg,:);
    LCOE_min(j)           = all_LCOE(idxg);
    curt_min(j)           = all_curt(idxg);
    maxT_min(j)           = all_maxT(idxg);
    cd_eta75_min(j)       = all_cd_eta75(idxg);
    curtail_factor_min(j) = all_curtail_factor(idxg);
    degrate_eff_min(j)    = all_degrate_eff(idxg);
    frac_heat_lim_min(j)  = all_frac_heating_limited(idxg)*100;
    hours_idle_min(j)     = all_hours_idle(idxg);
    hours_prod_min(j)     = all_hours_prod(idxg);
    hours_standby_min(j)  = all_hours_standby(idxg);
    load_factor_min(j)    = all_load_factor(idxg);
    n_starts_min(j)       = all_n_starts(idxg);
    therm_tau_min(j)      = all_therm_tau(idxg);
    vap_h2_pdt_min(j)     = all_vap_h2_pdt(idxg);
    p_min_standby_min(j)  = all_p_min_standby(idxg);
    p_min_min(j) = all_p_min(idxg);
    frac_standby_ener_min(j) = all_frac_standby_ener(idxg)*100;
    SEC_sta_min(j) =  all_SEC_sta(idxg);
    Nrep_min(j) =  all_Nrep(idxg);
    BM_pow_min(j) = all_BM_pow(idxg);
    BM_refrig_min(j) = all_BM_refrig(idxg);
    BM_pumps_min(j) = all_BM_pumps(idxg);
    BM_lyecool_min(j) = all_BM_lyecool(idxg);
    BM_heaters_min(j) = all_BM_heaters(idxg);
    BM_h2purif_min(j) = all_BM_h2purif(idxg);
    BM_h2compression_min(j) = all_BM_h2compression(idxg);
    BM_gasliqsep_min(j) = all_BM_gasliqsep(idxg);
    SEC_sys_min(j) = all_SEC_sys(idxg);
    BM_capex_min(j) = BM_pow_min(j) +BM_refrig_min(j) + BM_pumps_min(j) +  BM_lyecool_min(j) + BM_heaters_min(j) + BM_h2purif_min(j) + BM_h2compression_min(j) + BM_gasliqsep_min(j);
    W_sys75_min(j) = all_W_sys75(idxg);
    SEC_avg_stack_min(j)= all_SEC_avg_stack(idxg);
    BM_capex_min_per_kW(j) = BM_capex_min(j)/(W_sys75_min(j) );
    T_hist_min(j,:) = all_T_hist(idxg,:);
    pvgen_scaled_min(j,:) = pvgen*genratio/2*W_sys75_min(j);
    windgen_scaled_min(j,:) = windgen*genratio/2*W_sys75_min(j)*(1/1504.5);
    hybrid_min(j,:) =  pvgen_scaled_min(j,:) + windgen_scaled_min(j,:);
    P_min(j) = p_min_min(j) ; %(td_unique(j)/100*W_sys75_min(j)); % min. power reqd for production
    [td_min_val, idx_td_min] = min(td_unique);
    
    %     cd_eta75_min(j) = all_cd_eta75(idxg);
    % BM_heaters_min(j) = all_BM_heaters(idxg);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOT of temporal power profiles for checcking the thermal
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% resposnze and dispatch model
    if td_unique(j) ==10 %j == idx_td_min
    figure;
    plot(hybrid_min(j,:)/10^5)
    hold on 
    plot(T_hist_min(j,:))
    p_min_standby_min_vec = p_min_standby_min(j)*ones(8760,1);
    plot(p_min_standby_min_vec*1000/10^5)
    P_min_vec = (td_unique(j)/100*W_sys75_min(j))*ones(8760,1);
    W_sys75_min_vec = W_sys75_min(j)*ones(8760,1);
    plot(P_min_vec*1000/10^5)
    plot(W_sys75_min_vec*1000/10^5)
    plot(W_sys75_min_vec*1.1*1000/10^5)
    hold off

    end 
    param_all{:,j}       = X_P(mask,:);
    energy_all{j}      = energy_cost(mask);
    capex_all{j}       = capex_cost(mask);
    LCOH_all_cdeta{j}  = LCOH_all(mask);
    LCOE_all_cdeta{j}  = all_LCOE(mask);
    curt_all_cdeta{j}  = all_curt(mask);
    maxT_all_cdeta{j}  = all_maxT(mask);
    cd_eta75_all{j}    = all_cd_eta75(mask);
    
    frac_heat_lim_all{j}   = all_frac_heating_limited(mask);
    hours_idle_all{j}      = all_hours_idle(mask);
    hours_prod_all{j}      = all_hours_prod(mask);
    hours_standby_all{j}   = all_hours_standby(mask);
    load_factor_all{j}     = all_load_factor(mask);
    n_starts_all{j}        = all_n_starts(mask);
    therm_tau_all{j}       = all_therm_tau(mask);
    vap_h2_pdt_all{j}      = all_vap_h2_pdt(mask);
    p_min_standby{j}       = all_p_min_standby(mask);
    frac_standby_ener{j}   = all_frac_standby_ener(mask)*100; % in percent
    SEC_sta_all{j}   = all_SEC_sta(mask);
    Nrep{j} = all_Nrep(mask);
    SEC_sys_all{j}   = all_SEC_sys(mask);
    degrate_eff_all{j} = all_degrate_eff(mask);
    BM_pow_all{j} = all_BM_pow(mask);
    BM_refrig_all{j} = all_BM_refrig(mask);
    BM_pumps_all{j} = all_BM_pumps(mask);
    BM_lyecool_all{j} = all_BM_lyecool(mask);
    BM_heaters_all{j} = all_BM_heaters(mask);
    BM_h2purif_all{j} = all_BM_h2purif(mask);
    BM_h2compression_all{j} = all_BM_h2compression(mask);
    BM_gasliqsep_all{j} = all_BM_gasliqsep(mask);
    SEC_avg_stack_all{j} = all_SEC_avg_stack(mask);
    Tsp = 80;
    Pavail = hybrid_min(j,:);         % power profile for design j in W
    Thist  = T_hist_min(j,:);         % temperature profile
    Pmin_j = P_min(j);                % min production power
    prod_power_ok = Pavail >= Pmin_j;
    below_Tsp     = Thist  < 353.15;
    startup_loss_mask = prod_power_ok & below_Tsp;
    startup_hrs_lost_min(j) = sum(startup_loss_mask);
    E_lost_startup_kWh_min(j) = sum(0.001*Pavail(startup_loss_mask)); % in kWh per year
    %T_hist_min{j} = all_T_hist(mask,:);
   % BM_heaters{j} = all_BM_heaters(mask);
end


% 2) choose Lmin/Lmax from the vector of minima
Lmin = min(LCOH_min);
Lmax = max(LCOH_min);
%some useful heaterrating plots to show the tradeoff between lower frac
%heater limited and 
% plot(heater_unique,frac_heat_lim_min*100)
% hold on
% plot(heater_unique,frac_standby_ener_min*100)
% plot(heater_unique,LCOH_min)
% plot(heater_unique,BM_heaters_min)
% plot(heater_unique,n_starts_min*0.01)
figure;
scatter3(energy_cost, capex_cost, td, 40, LCOH_all, 'filled');  % background cloud

%T = table(energy_cost(:), capex_cost(:), td(:), LCOH_all(:), ...
%          'VariableNames', {'Energy_cost','Capex_cost','TD','LCOH_all'});

%writetable(T,'pareto3D_data.csv');   % or .xlsx

xlabel('Energy cost [$ / kg H_2]');
ylabel('CAPEX cost [$ / kg H_2]');
zlabel('Partial load limit[%]');
cb = colorbar; ylabel(cb,'LCOH [$ / kg]');
grid on; box on; hold on;
colormap(parula);

ms_vec = nan(n,1);
for j = 1:n
    idxg = idxg_vec(j);
    Lval = LCOH_min(j);

    % map LCOH_min(j) to marker size
    tnorm    = (Lval - Lmin) / max(Lmax - Lmin, eps);  % 0→best, 1→worst
    ms_vec(j)= 5 + 15 * tnorm;

    plot3(energy_cost(idxg), capex_cost(idxg), td(idxg), ...
          'o', 'MarkerSize', 8, 'LineWidth', 1.5, ...
          'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'k');

    % plot3(energy_cost(idxg), capex_cost(idxg), td(idxg), ...
    %       'o', 'MarkerSize', ms_vec(j), 'LineWidth', 1.5, ...
    %       'MarkerFaceColor', [1 0 0], 'MarkerEdgeColor', 'k');
end
hold off;

%cdeta_unique: vector of unique partial-load limits (same index j as *_all)
n = numel(td_unique);

hours_idle_mean    = zeros(n,1);
hours_prod_mean    = zeros(n,1);
hours_standby_mean = zeros(n,1);
frac_heat_mean     = zeros(n,1);
frac_standby_ener_mean =zeros (n,1);
n_starts_mean =zeros (n,1);

for j = 1:n
ener_mean(j)            = mean(energy_all{j}); 
%param_mean(j)         = mean(LCOH_all_cdeta{j});
LCOH_mean(j)          = mean(LCOH_all_cdeta{j});
LCOE_mean(j)          = mean(LCOE_all_cdeta{j});
curt_mean(j)          = mean(curt_all_cdeta{j});
maxT_mean(j)          = mean(maxT_all_cdeta{j});
SEC_sys_mean(j)       =mean(energy_all{j})./(mean(LCOE_all_cdeta{j}));
cd_eta75_mean(j)      = mean(cd_eta75_all{j});
frac_heat_lim_mean(j) = mean(frac_heat_lim_all{j});
hours_idle_mean(j)    = mean(hours_idle_all{j});
hours_prod_mean(j)    = mean(hours_prod_all{j});
hours_standby_mean(j) = mean(hours_standby_all{j});
load_factor_mean(j)   = mean(load_factor_all{j});
n_starts_mean(j)      = mean(n_starts_all{j});
therm_tau_mean(j)     = mean(therm_tau_all{j});
vap_h2_pdt_mean(j)    = mean(vap_h2_pdt_all{j});
p_min_standby_mean(j) = mean(p_min_standby{j});
frac_standby_ener_mean(j) = mean(frac_standby_ener{j}); % in percent
SEC_sta_mean(j)       = mean(SEC_sta_all{j});
param_mean(j,:)         = mean(param_all{j},1);
Nrep_mean(j,:) = mean(Nrep{j},1);
degrate_eff_mean(j,:) = mean(degrate_eff_all{j},1);
end

% figure; hold on; box on;
% plot(td_unique, hours_prod_mean,    '-o','LineWidth',1.5);
% plot(td_unique, hours_standby_mean, '-s','LineWidth',1.5);
% plot(td_unique, hours_idle_mean,    '-d','LineWidth',1.5);
% %plot(td_unique, frac_heat_mean,     '-^','LineWidth',1.5);
% xlabel('Partial load limit, \cdot\eta^{min}');
% ylabel('Hours / fraction');
% legend({'Prod hours','Standby hours','Idle hours','Frac heating-limited'}, ...
%        'Location','best');
% set(gca,'Position',[0.15 0.15 0.8 0.75]);
% set(gca,'LineWidth',1.5,'FontSize',12,'Color','w');
% 
% figure('Color','w'); hold on; box on;
% set(gca,'Color','w', ...      % white axes background
%         'LineWidth',1.5, ...
%         'FontSize',12, ...
%         'Box','on');


figure
hold on
plot(td_unique, n_starts_min*0.01, '-s','LineWidth',1.5);
plot(td_unique, frac_standby_ener_min, '-^','LineWidth',1.5);
plot(td_unique, degrate_eff_min, '-*','LineWidth',1.5)
plot(td_unique, Nrep_min, '-o','LineWidth',1.5)
plot(td_unique, maxT_min, '-.','LineWidth',1.5)
plot(td_unique, SEC_sys_min*0.1, '--','LineWidth',1.5)
plot(td_unique, SEC_avg_stack_min*0.1, '-x','LineWidth',1.5)
%plot(td_unique, SEC_sta_min*0.1, '-c','LineWidth',1.5)

plot(td_unique, vap_h2_pdt_min*10e-7, '-k','LineWidth',1.5)

xlabel('Partial load limit, c\_d\eta^{min}');
ylabel('% of annual hrs heating-limited');
set(gca,'LineWidth',1.5,'FontSize',12,'Color','w');
set(gca,'Color','w', ...      % white axes background
        'LineWidth',1.5, ...
        'FontSize',12, ...
        'Box','on');
set(gcf,'Color','w');      % figure background
set(gca,'Color','w','Box','on','LineWidth',1.5);  % axes background + frame


%Calc. of degradation effectiveness here 
sw = 1;
m_degrate = 0.001;
nj = 0.5;
bj=0.00055; 
degrate = base_degrate + m_degrate*(max(maxT_min,1).^nj).*exp(bj*param_min(:,6));  
alpha = sw*2*5*10^-4 ;
degrate_eff_calc = degrate.*(1.0 + alpha.*n_starts_min) ;


% --- prepare table for Origin export ---
td_unique = unique(td);              % already computed
n = numel(td_unique);
LCOH_min  = nan(n,1);
ener_ratio = nan(n,1);
ms_vec    = nan(n,1);                % marker sizes you used

Lmin = 8;                 % same as in your code
Lmax = max(LCOH_all);

for j = 1:n
    mask = td == td_unique(j);
    [~, idx_local] = min(LCOH_all(mask));
    idxg = find(mask);
    idxg = idxg(idx_local);

    Lval = LCOH_all(idxg);
    LCOH_min(j)   = Lval;
    ener_ratio(j) = energy_cost(idxg)/capex_cost(idxg);
    tnorm = (Lval - Lmin) / max(Lmax - Lmin, eps);
    ms_vec(j) = 5 + 15*(tnorm);  % same size formula
end

% full cloud data for scatter
T_cloud = table(energy_cost, capex_cost, td, all_LCOE, LCOH_all, ...
    'VariableNames', {'Energy_dolkgh2','Capex_dolkgh2','Turndown_pct', ...
                      'LCOE_dolkWh','LCOH_dolkgh2'});

% min-LCOH points per turndown
T_min = table(td_unique, LCOH_min, ener_ratio, ms_vec, ...
    'VariableNames', {'Turndown_pct','LCOH_min_dolkgh2', ...
                      'Energy_to_Capex_ratio','MarkerSize'});

% write to CSV
% writetable(T_cloud, 'GA_pareto_cloud.csv');
% writetable(T_min,   'GA_minLCOH_per_turndown.csv');


% === 2D Pareto curve for first turndown only ===
td_unique = unique(td);              % first turndown value
td1 = td_unique(1);

mask1 = (td == td1);
E1    = energy_cost(mask1);
C1    = capex_cost(mask1);

% sort by energy for a clean curve
[E1s, idx1] = sort(E1);
C1s        = C1(idx1);

% create figure and set white background BEFORE plotting
figure;
set(gcf,'Color','w');                % figure background white
set(gca,'FontName','Arial','FontSize',12,'LineWidth',1);  % nice axes
plot(E1s, C1s, '-o', 'LineWidth',1.5, 'MarkerSize',6);
xlabel('Energy cost [$ / kg H_2]');
ylabel('CAPEX annuity [$ / kg H_2]');
title(sprintf('Pareto front at turndown = %.1f %%', td1));
grid off; box on;


