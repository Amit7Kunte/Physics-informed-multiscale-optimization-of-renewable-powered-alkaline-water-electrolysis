clc; clear;
format long g
%close all;
%% Economic Parameters
% 1. Capital recovery factor
t_sys = 30;
n_years = t_sys;    % plant lifetime
r = 0.08;      % discount rate (% per year)
CRF = (r*(1+r)^n_years)/((1+r)^n_years-1); % as before
%CRF = 0.08;                % Capital Recovery Factor (8%)closec
OPEX_percent = 3;          % O&M as % of CAPEX
degrade_stack =10;
%t_oper = 3500;             % Operating hours per year
n_lifetime = t_sys;           % Plant lifetime (years)
gc=10^6;
conv_rate =1.17;
Nc =5;
rho_steel = 7850; rho_Ni = 8900;
% Physical constants
J_kWh = 2.7778E-7;
MH2 = 0.002;               % Molecular weight of H2 (kg/mol)
HHV_H2 = 285.8*1000*J_kWh/MH2;            % Higher Heating Value (kWh/kg)
Vm = 22.414;                 % Molar volume at STP (L/mol)
% Gather all files and sort by cdeta
fileList = dir('combined_cdeta0.*_m0.0010_nj0.00_PV75.0_gr3.0.mat');
%fileList = dir('ga_sens_cdeta0.078_m0.0010_nj0.5_PV_rated75.0_gr4.0_lcoh_BESScapex220.0_alpha0.001_FUH1.0_job11086306_SECsys_batt.mat');
pvgen  = readmatrix('pvgen_profile.csv');      % 8760x1
windgen = readmatrix('windgen_profile.csv')/1000;   % 8760x1, power in W per 1500 W of rated turbine
HHV_H2 = 285.8*1000*2.7778E-7/0.002;

%fileList = dir('ga_sens_cdeta0.073_m0.0010_nj0.5_PV_rated75.0_gr1.5_lcoh_capex.mat');
%excludeIdx = contains(fileList.name, 'cdeta0.08_') | contains(fileList.name, 'cdeta0.15_')| contains(fileList.name, 'cdeta0.25_');
Tmax_vals = zeros(length(fileList),1);
for i = 1:length(fileList)
    tokens = regexp(fileList(i).name,'_cdeta([0-9\.]+)_', 'tokens');
     t2 = regexp(fileList(i).name,'_gr([0-9\.]+)\.','tokens','once');
     t3 = regexp(fileList(i).name,'_PV([0-9\.]+)_','tokens','once');
     t4 = regexp(fileList(i).name,'_m([0-9\.]+)_','tokens','once');
       genratio_vals(i) = str2double(t2{1});
       PV_vals(i) = str2double(t3{1});
       m_vals(i) = str2double(t4{1});
    Tmax_vals(i) = str2double(tokens{1})*100;
end
[gerw, sortIdx] = sort(genratio_vals);
genratio  = mean(gerw);
PV =mean(PV_vals);
mvals = mean(m_vals);
% gr_vals = zeros(length(fileList),1);
% 
% for i = 1:length(fileList)
%     tokens = regexp(fileList(i).name,'_gr([0-9\.]+)_','tokens');
%     gr_vals(i) = str2double(tokens{1}{1});
% end
% 
% for i = 1:length(fileList)
%     tokens = regexp(fileList(i).name,'_PV_rated([0-9\.]+)_','tokens');
%     PV_rated(i) = str2double(tokens{1}{1});
% end
% 
% for i = 1:length(fileList)
%     tokens = regexp(fileList(i).name,'_m([0-9\.]+)_','tokens');
%     m_degrate(i) = str2double(tokens{1}{1});
% end
% 
% for i = 1:length(fileList)
%     tokens = regexp(fileList(i).name,'_nj([0-9\.]+)_','tokens');
%     n_j(i) = str2double(tokens{1}{1});
% end
% fileList = fileList(sortIdx);




% Preallocate
nFiles = length(fileList); 
nObj = 2; % [SEC, H2prod, CAPEX]
nParams = 6; % update if needed
LCOH_params=zeros(nFiles, nParams);
LCOH_obj = zeros(nFiles, 1);
LINMAP_obj = zeros(nFiles, nObj);
MIN_obj = zeros(nFiles, nObj);
MAX_obj = zeros(nFiles, nObj);

LINMAP_params = zeros(nFiles, nParams);
MIN_params = zeros(nFiles, nParams);
MAX_params = zeros(nFiles, nParams);

for k = 1:length(fileList)
    data = load(fileList(k).name);
%%%%%%%%%%%%%%%%%%%%
    F = data.pareto_F;   
    X =  data.pareto_X;   
    % Extract Pareto front data
    Energy_dolperkg = data.pareto_F(:, 1);           
                                            
          
    CAPEX_dolperkg = data.pareto_F(:, 2);
    % [N x 3]
    curtail = data.pareto_curtail_factor;
    LCOE= data.pareto_LCOE_awe;
    vap_h2_pdt = data.pareto_vap_h2_pdt;
    maxT = data.pareto_maxT;
    LF = data.pareto_load_factor;
    W_sys75= data.pareto_W_sys75;
    LCOE_RES = data.pareto_LCOE_awe./data.pareto_curtail_factor;
    SEC_sys =  Energy_dolperkg./LCOE ;   % System SEC (kWh/kg)
    TIC_sys =  CAPEX_dolperkg.*vap_h2_pdt/(1.03*CRF) ;
    TICperkW = TIC_sys./W_sys75;
    Nrep =data.pareto_Nrep;
    %Nrep =data.Nrep;
     n_starts = data.pareto_n_starts;
     %n_starts = data.n_starts;
      frac_stdby_ener = data.pareto_frac_standby_ener;
      frac_heating_limited = data.pareto_frac_heating_limited; 
     degrate_eff = data.pareto_degrate_eff;
     SEC_sta = data.pareto_SEC_sta;
     cd_eta75 = data.pareto_cd_eta75;
    t_oper = LF/100*8760;
    dur_stack = degrade_stack./degrate_eff*8760; % in 1000 hrs
            hours_idle   = data.pareto_hours_idle;
           hours_prod    = data.pareto_hours_prod;
          hours_stdby     = data.pareto_hours_standby;
         p_min_stdby      = data.pareto_p_min_standby;
         P_min = data.pareto_p_min;
         therm_tau      = data.pareto_therm_tau;
            T_hist  = data.pareto_T_hist;
                    params = data.pareto_X;              % [N x 6]
                 BM_pow = data.pareto_BM_pow;
                 BM_pumps = data.pareto_BM_pumps;
               BM_heaters = data.pareto_BM_heaters; 
               BM_lyecool = data.pareto_BM_lyecool;
               BM_h2purif = data.pareto_BM_h2purif;
             BM_gasliqsep = data.pareto_BM_gasliqsep;
                BM_refrig = data.pareto_BM_refrig; 
         BM_h2compression = data.pareto_BM_h2compression;
    % Store LINMAP, min, max - objecti

     MIN_obj(k,:)    = min(F,[],1);
     MAX_obj(k,:)    = max(F,[],1);
     MEAN_obj(k,:)   = mean(F,1);
    % Decision parameters
    % LINMAP_params(k,:) = params(idx_linmap,:);
    MIN_params(k,:)    = min(params,[],1);
    MAX_params(k,:)    = max(params,[],1);
    
   
    % Design parameters
    %X = data.pareto_X;
    
    % Number of solutions
    n_sol = length(Energy_dolperkg);
    

    
%%    CAPEX_kW = CAPEX_dolperkg./(W_sys75'*CRF*1.03);

    LCOH = CAPEX_dolperkg + Energy_dolperkg ;  % $/kg H2


    %% Find Minimum N LCOH Solutions
    N=1;
    [LCOH_sorted, idxs_sorted] = sort(LCOH);  
    LCOH_best = LCOH_sorted(1:N);                       % Take best N
    idxs_best = idxs_sorted(1:N);
    % For robust plot/trend, use:
    LCOH_mean(k) = mean(LCOH_best);                        % Average
    LCOH_median(k) = median(LCOH_best);                    % Median (even less sensitive to outlier)

 % For extracting parameters corresponding to these solutions:
    params_best = params(idxs_best, :);
    params_meanbest(k,:)   = mean(params_best,1);
    [LCOH_min, idx_min] = min(LCOH);
    LCOH_params(k,:) = params(idx_min,:);
    LCOH_obj(k) = min(LCOH);
%%    load_factor_obj(k) = mean(data.pareto_load_factor);

    optimal_design.CAPEX_dolperkg(k) =  CAPEX_dolperkg(idx_min);
    %optimal_design.OPEX_annual(k) = OPEX_annual (idx_min);
    %optimal_design.CAPEX_dolperkg(k) =  (CAPEX_annual(idx_min))./H2_annual(idx_min);
    %optimal_design.CAPEX_kW(k) =  CAPEX_kW(idx_min);
    %optimal_design.H2_annual(k) =  H2_annual(idx_min);
    %optimal_design.Energy_annual(k) =  Energy_annual(idx_min).*H2_annual(idx_min);
    optimal_design.Energy_dolperkg(k) = Energy_dolperkg(idx_min);
    optimal_design.W_sys75(k) = W_sys75(idx_min)/1000;
    optimal_design.LCOE(k) = LCOE(idx_min);
    optimal_design.curtail(k) =curtail(idx_min);
    optimal_design.vap_h2_pdt(k) = vap_h2_pdt(idx_min)/1000; % in tonnes per annum
    optimal_design.maxT(k) = maxT(idx_min);
    optimal_design.LF(k) = LF(idx_min);
    optimal_design.LCOE_RES(k) = LCOE_RES(idx_min);
    optimal_design.SEC_sys(k) = SEC_sys(idx_min);
    optimal_design.frac_stdby_ener = frac_stdby_ener(idx_min);
    optimal_design.frac_heating_limited = frac_heating_limited(idx_min);
    optimal_design.TIC_sys(k) = TIC_sys(idx_min);
    optimal_design.dur_stack(k) = dur_stack(idx_min);          
    optimal_design.degrate_eff(k) = (degrate_eff(idx_min));
    optimal_design.Nrep(k) = (Nrep(idx_min));
    optimal_design.TICperkW(k) =  (TICperkW( idx_min));
    optimal_design.t_hist(k,:) = T_hist(idx_min,:);
    optimal_design.hours_idle(k) = hours_idle(idx_min);
    optimal_design.hours_prod(k) = hours_prod(idx_min);
    optimal_design.hours_stdby(k) = hours_stdby(idx_min);
    optimal_design.p_min_stdby(k) = p_min_stdby(idx_min);
    optimal_design.P_min(k) = P_min(idx_min);
    optimal_design.therm_tau(k) = therm_tau(idx_min);
    optimal_design.SEC_sta(k) = SEC_sta(idx_min);
    optimal_design.cd_eta75(k) = cd_eta75(idx_min);
    optimal_design.t_oper(k) = t_oper(idx_min);

    optimal_design.BM_pow(k) = (BM_pow(idx_min));
    optimal_design.BM_pumps(k) = (BM_pumps(idx_min));
    optimal_design.BM_heaters(k) = (BM_heaters(idx_min));
    optimal_design.BM_lyecool(k) = (BM_lyecool(idx_min));
    optimal_design.BM_h2purif(k) = (BM_h2purif(idx_min));
    optimal_design.BM_gasliqsep(k) = (BM_gasliqsep(idx_min));
    optimal_design.BM_refrig(k) = (BM_refrig(idx_min));
    optimal_design.BM_h2compression(k) = (BM_h2compression(idx_min));
    optimal_design.n_starts(k) = (n_starts(idx_min));
    optimal_design.cd(k) = X(idx_min,6)  ;  % Input

    pvgen_scaled = pvgen.*(genratio/2).*optimal_design.W_sys75(k); % pvgen in W per 1kW cc

    windgen_scaled = windgen*genratio/2*optimal_design.W_sys75(k)*(1/1504.5)*1000;
    
    hybrid =  pvgen_scaled + windgen_scaled; % inkW

    %P_min = (Tmax_vals(k)/100*optimal_design.W_sys75(k)*1000); % min. power reqd for production in kW



    Tsp = 80;
    Pavail = hybrid;         % power profile for design j in W
    
    ggsum = sum(Pavail); %Total energy available from RES in kWh, and depends only on the rated power of the AWEBOp (for a fixed genratio)

    Thist  = optimal_design.t_hist(k,:);         % temperature profile

    Pmin_j =  optimal_design.P_min(k);                % min production power in kW

    prod_power_ok = Pavail >= Pmin_j/1000;

    prod_power_ok_hrs = sum(prod_power_ok); % seems to be a const number.

    below_Tsp     = Thist  < 353.15;

    below_Tsp_hrs = sum(below_Tsp );

    startup_loss_mask = prod_power_ok & below_Tsp';

    startup_loss_mask_hrs= sum(startup_loss_mask);

    E_lost_startup_kWh = sum(Pavail(startup_loss_mask)); % in kWh per year

   % E_lost_startup_min(j) = E_lost_startup_kWh(j)/vap_h2_pdt_min(j);

    optimal_design.frac_E_lost_strtup(k)  =E_lost_startup_kWh./(optimal_design.W_sys75(k)*1000*length(Pavail)); % Input

    % optimal_design.CAPEX_dolperkg(k) =  (CAPEX_dolperkg( idxs_best));
    % optimal_design.Energy_dolperkg(k) = Energy_dolperkg(idxs_best);
    % optimal_design.LCOE_RES(k) =  (LCOE_RES( idxs_best));
    % optimal_design.SEC_sys(k) =  (SEC_sys( idxs_best));
    % optimal_design.TIC_sys(k) =  (TIC_sys( idxs_best));
    % optimal_design.TICperkW(k) =  (TICperkW( idxs_best));
    % %optimal_design.repl_cost(k) =  (repl_cost( idxs_best));
    % optimal_design.n_starts(k) = (n_starts(idxs_best));
    % %optimal_design.standby_time(k) = (standby_time(idxs_best));
    % optimal_design.degrate_eff(k) = (degrate_eff(idxs_best));
    % optimal_design.dur_stack(k) = (dur_stack(idxs_best));
    % 
    % optimal_design.Nrep(k) = (Nrep(idxs_best));
    % optimal_design.BM_pow(k) = (BM_pow(idxs_best));
    % optimal_design.BM_pumps(k) = (BM_pumps(idxs_best));
    % optimal_design.BM_heaters(k) = (BM_heaters(idxs_best));
    % optimal_design.BM_lyecool(k) = (BM_lyecool(idxs_best));
    % optimal_design.BM_h2purif(k) = (BM_h2purif(idxs_best));
    % optimal_design.BM_gasliqsep(k) = (BM_gasliqsep(idxs_best));
    % optimal_design.BM_refrig(k) = (BM_refrig(idxs_best));
    % optimal_design.BM_h2compression(k) = (BM_h2compression(idxs_best));
    % 
    % optimal_design.SEC_sta(k) = (SEC_sta(idxs_best));
    % optimal_designmean.Energy_dolperkg(k) = mean(Energy_dolperkg( idxs_best));
    % optimal_design.cd_eta75(k) = (cd_eta75(idxs_best));
    % optimal_design.hours_idle(k) = hours_idle(idxs_best);
    % optimal_design.hours_prod(k) = hours_prod(idxs_best);
    % optimal_design.hours_stdby(k) = hours_stdby(idxs_best);
    % optimal_design.p_min_stdby(k) = p_min_stdby(idxs_best);
    % optimal_design.therm_tau(k) = therm_tau(idxs_best);
    % optimal_design.t_oper(k) = t_oper(idxs_best);

%%    optimal_design.W_sys75(k) =  W_sys75(idx_min);
    
    % optimal_design.CAPEX(k) = absCAPEX(idx_min);
    % optimal_designmean.CAPEX(k) = mean(absCAPEX(idxs_best,:));
    % optimal_design.W_sys75(k) = W_sys75(idx_min);
    %optimal_designmean.W_sys75(k) = mean(W_sys75(idxs_best,:));
    optimal_design.X(k,:) = X(idx_min, :);
 
     % the no. of stack replacements in plant lifetime. to ccompare SEC_avg_stack for various m, can also be interpreted as the avg SEC consumption of first stack, when Nrep = 1. and so on...
    % --- H2 compressor ---
    eta_isen_comp = 0.85;
    T_amb = 298.15; % Ambient temp.
    P_final = 30; % final h2 delivery pressure in bar
    T_max_comp = 273.15 + 140;
    k_isen = 1.41;
    Z_comp = 1.05;
    delP_fan = 500 ;% Pa ; % needs to be revisited 
    eta_fan =0.55;
    T_suc_comp = T_amb + 11; % suction temp. 
    %rc_max1 = ((T_max_comp/T_amb-1)*eta_isen_comp+1)^(k_isen/(k_isen-1));
    rc_max = ((T_max_comp/T_suc_comp-1)*eta_isen_comp+1)^(k_isen/(k_isen-1));
    N_comp(k) = ceil(log(P_final/(optimal_design.X(k,2)))/log(rc_max));
    delT_air = 10;%10K
    Cp_air = 1005;
    rho_air = 1.18;% kg/m^3;
    eta_mech_motor = 0.95;
    eta_mech_comp= 0.9;
    R_const = 8.314;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calc. of % replacement cost of TIC
    % STACK

    Ni_foam = 0.5/10^-6; % $0.5/cm^3,https://shop.nanografi.com/battery-equipment/nickel-foam-for-battery-cathode-substrate-size-1000-mm-x-300-mm-x-1-6-mm/
    zf_sep = gc*150*0.05*0.008*conv_rate*Nc; % 150 Euro/m^2 from AGFA Data 'Present and future cost of alkaline and PEM electrolyser stacks'
    steel = gc*(0.05*0.008*0.003*2+0.05*0.008*0.001*(Nc-1))*rho_steel*4.5;    % 0.9 $ /kg for carbon steel, 4.5$/kg for SS316L. SS used for longer stack life and in advanced stack designs operating at higher cd. SOURCE:https://www.imarcgroup.com/stainless-steel-316-price-trend
    Ni =  gc*2*(0.05*0.008*(X(:,1))*0.001*(Nc))*Ni_foam; % 15 $/kg Ni price as on 4/10/2025, based on volume of electrode
    Fp_stack = 1+0.2*((X(:,2))-1)/(15-1);% Saba etal 2018, reported around 20% increase in stack costs for a pressure increase from 1 bar to 15 bar, assuming a linear relationship between operating pressure and stack material costs
    stack_mat_cost = 1.3*(zf_sep+steel+Ni).*Fp_stack;  % $/kW, 30% additional for balance of stack components such as gaskets etc.
    dir_stack_cost = stack_mat_cost+ 0.5*stack_mat_cost+0.125*stack_mat_cost; % ratio is as 8:4:1 from 'Present and future cost of alkaline and PEM electrolyser stacks'.
    Tot_stack_costs = 2*dir_stack_cost; % including overheads $/kW   
    degrade_stack = 10;
    Tot_stack_cost(k) = Tot_stack_costs(idxs_best);
    PV_repl =0;
    %optimal_design.t_oper(k) = optimal_design.LF(k)*8760/100;
    repl_inter_yrs = optimal_design.dur_stack(k)./optimal_design.t_oper(k);
    for l = 1:optimal_design.Nrep(k)
        PV_repl= PV_repl + Tot_stack_cost(k)./(1+r).^(l*repl_inter_yrs);
    end
    repl_inter_yrs_vec(k) = repl_inter_yrs;
    Tot_repl_cost = PV_repl;
    optimal_design.repl_cost(k) = Tot_repl_cost./optimal_design.TIC_sys(k)*100 ; %Tot_repl_cost./optimal_design.TIC_sys(k)*100, in %;
    Tot_repl_cost_vec(k) = Tot_repl_cost;
    % Calc. of total capex costs
    optimal_design.Tot_stack_life_cost(k) = Tot_stack_cost(k) + PV_repl;
    optimal_design.Tot_stack_cost(k) = Tot_stack_cost(k) ;
    BM_capex(k)  = optimal_design.Tot_stack_cost(k) + optimal_design.BM_pow(k) + optimal_design.BM_pumps(k) + optimal_design.BM_heaters(k) + optimal_design.BM_lyecool(k) + optimal_design.BM_h2purif(k) + optimal_design.BM_gasliqsep(k) + optimal_design.BM_refrig(k) +  optimal_design.BM_h2compression(k) ;
    %optimal_design.repl_cost(k) = Tot_repl_cost ./BM_capex(k)*100 ; %Tot_repl_cost./optimal_design.TIC_sys(k)*100, in %;

    allCapexComponents = [optimal_design.BM_h2compression(k) ,optimal_design.BM_refrig(k) ,optimal_design.BM_gasliqsep(k) ,optimal_design.BM_h2purif(k) ,optimal_design.BM_lyecool(k),optimal_design.BM_heaters(k), optimal_design.BM_pumps(k),optimal_design.BM_pow(k), optimal_design.Tot_stack_cost(k)];
    pieVals = [
    allCapexComponents(:,1)
    allCapexComponents(:,2)
    allCapexComponents(:,3)
    allCapexComponents(:,4)
    allCapexComponents(:,5)
    allCapexComponents(:,6)
    allCapexComponents(:,7)
    allCapexComponents(:,8)
    allCapexComponents(:,9)
];
    pieLabels = {
    'H2 compression'
    'Refrigeration'
    'Gas-liq separation'
    'H2 purification'
    'Lye coolers'
    'BM_heater'
    'pumps'
    'Power'
    'Stack'
    };
    totalCapex = BM_capex(k);
    piePct = 100 * pieVals / totalCapex;
    pieLabelsPct = cell(size(pieLabels));
    for i = 1:numel(pieLabels)
    pieLabelsPct{i} = sprintf('%s (%.1f%%)', pieLabels{i}, piePct(i));
    end
    % 3) Quick check pie in MATLAB (optional)
%     figure;
%     pie(pieVals, pieLabelsPct);
%     title(sprintf('CAPEX breakdown at %.2f percent (Total = %.2f USD)', Tmax_vals(k),totalCapex));
% 
% 
% % 4) Export values + labels to CSV for Origin
% T = table(pieLabels, pieVals, 'VariableNames', {'Label','Value'});
end

%%%%%%%%%%%%DATA COLELCTION FOR PUBLICATION PLOT%%%%%%%%%%%%%%%%

metric_names = {'vap_h2_pdt','TIC_sys','repl_cost','Load Factor', 'SEC_sys', 'LCOE'};
input_names = {'n_starts','maxT','elec width', 'pres','inlet vel', 'pore size','sep width','curr dens'};
LCOH_names ={'LCOH','Energy per kg','Capex per kg'};
LCOH_bkp = [LCOH_obj';optimal_design.Energy_dolperkg ; optimal_design.CAPEX_dolperkg];
[LCOH_min, idx_col_min] = min(LCOH_bkp(1,:));
 % 
 T = array2table(LCOH_bkp', 'VariableNames', LCOH_names);  % rows = loads
 T.min_load = Tmax_vals(:);                         % add as last column
 T = movevars(T,'min_load','Before',1);                % make it first column

 %writetable(T,'DD_LCOH_vs_load_m0.001_PV75.0_gr2.0.xlsx','Sheet','data');
 fname = sprintf('DD_LCOH_vs_load_m%.3f_PV%.1f_gr%.1f.xlsx', mvals,PV, genratio);
 writetable(T, fname, 'Sheet', 'data');

metric_bkp = [optimal_design.vap_h2_pdt/100;optimal_design.TIC_sys/10^6;optimal_design.repl_cost;optimal_design.LF; optimal_design.SEC_sys;optimal_design.LCOE*100 ]; %;optimal_design.curtail];
% M = metric_bkp;
% %Mmin = min(metric_bkp ,[],2);
% %Mmax = max(metric_bkp ,[],2);
% M_opt = metric_bkp(:, idx_col_min);
% kg = 100;
% R     = M ./ M_opt;
% D = (R - 1) * kg;

% plot(hybrid*0.003)
% hold on
% plot(optimal_design.t_hist(11,:))

% %input_names = {'maxT', 'elec width', 'pres','inlet vel', 'pore size','sep width', 'curr dens'};
%  T = array2table(D', 'VariableNames', metric_names);  % rows = loads
%   T.min_load = Tmax_vals(:);                         % add as last column
%   T = movevars(T,'min_load','Before',1);                % make it first column
%   writetable(T,'DD_metricbkp_m0.001_PV75.0_gr2.0.xlsx','Sheet','data');

nMetrics = size(metric_bkp,1);

M = metric_bkp;
%Mmin = min(metric_bkp ,[],2);
%Mmax = max(metric_bkp ,[],2);
M_opt = metric_bkp(:, idx_col_min);
kg = 100;
R     = M ./ M_opt;
D = (R - 1) * kg;

 T = array2table(M', 'VariableNames', metric_names);  % rows = loads
 T.min_load = Tmax_vals(:);                         % add as last column
 T = movevars(T,'min_load','Before',1);                % make it first column
fname = sprintf('D_metrics_vs_load_m%.3f_PV%.1f_gr%.1f.xlsx', mvals,PV, genratio);
 writetable(T, fname, 'Sheet', 'data');
 writetable(T,'D_metrics_vs_load.xlsx','Sheet','data');
% 
 input_param_bkp = [optimal_design.n_starts'/100 optimal_design.maxT' ,optimal_design.X(:,1), optimal_design.X(:,2),optimal_design.X(:,3)*100,optimal_design.X(:,4)/100,optimal_design.X(:,5)*5,optimal_design.X(:,6)*0.001]; %;optimal_design.curtail];

 

% clear M
M = input_param_bkp;
input_names = {'n_starts','maxT', 'elec width', 'pres','inlet vel', 'pore size','sep width', 'curr dens'};
  T = array2table(input_param_bkp, 'VariableNames', input_names);  % rows = loads
  T.min_load = Tmax_vals(:);                         % add as last column
  T = movevars(T,'min_load','Before',1);                % make it first column
  fname = sprintf('DD_inputparam_vs_load_m%.3f_PV%.1f_gr%.1f.xlsx', mvals,PV, genratio);
  writetable(T, fname, 'Sheet', 'data');



% figure;
% bar(Tmax_vals, D.', 'grouped');    % bars grouped by load
% yline(0,'k-','LineWidth',1);   % reference = optimum
% 
% xlabel('Minimum load limit (%)');
% 
% ylabel('\Delta metric relative to optimal load (%)');
% 
% legend(metric_names, ...
%        'Location','northoutside', ...
%        'Orientation','horizontal');
% grid on;
% figure;
% for m = 1:nMetrics
%     subplot(nMetrics,1,m);
%     bar(Tmax_vals, D(m,:));
%     yline(0,'k-');
%     ylabel(metric_names{m});
%     if m==1, title('\Delta metric vs optimal load (%)'); end
%     if m==nMetrics, xlabel('Minimum load limit (%)'); end
% end

% Mmax(Mmax==0) = 1;           % avoid division by zero
% M_norm =metric_bkp ./ Mmax;          % each row now has max = 1
% M_opt = M(:, idx_opt);      % value at optimal partial load
% R     = M ./ M_opt;         % ratio vs optimal (1 at 20% for all metrics)
% 
% 

figure('Color','w');

%% Subplot 1: LCOH vs Tmax
subplot(1,4,1); hold on; grid off;
plot(Tmax_vals, LCOH_obj(:), '-o', 'LineWidth',2,'MarkerSize',9,'Color',[0.1 0.5 0.1],'MarkerFaceColor',[0.1 0.5 0.1]);
plot(Tmax_vals, LCOH_mean(:), '--s', 'LineWidth',2,'MarkerSize',8,'Color',[0.4 0.6 0.1],'MarkerFaceColor',[0.4 0.6 0.1]);
%plot(Tmax_vals, LCOH_median(:), ':^', 'LineWidth',2,'MarkerSize',8,'Color',[0.85 0.33 0.1],'MarkerFaceColor',[0.85 0.33 0.1]);
xlabel('minimum load/rated load (%)','FontWeight','bold','FontSize',16);
ylabel('LCOH (kg H_2)','FontWeight','bold','FontSize',16);
%legend({'LCOH-min','LCOH-mean (best 10)','LCOH-median (best 10)'},'FontSize',11, 'Location','best','Box','off');
title('LCOH vs. minimum load/rated load (%)');
ax = gca; 
ax.Box = 'on';         % draws the frame around the axes
ax.LineWidth = 1.5;    % set the frame thickness (adjust as needed)

hold off;

%% Subplot 2: Objective function values (distinct colors)
subplot(1,4,2); hold on; grid off;

plot(Tmax_vals, optimal_design.CAPEX_dolperkg,    '-s', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.447 0.741],'MarkerFaceColor',[0 0.447 0.741]);

plot(Tmax_vals, 1*optimal_design.Energy_dolperkg,    '-x', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.747 0.341],'MarkerFaceColor',[0 0.747 0.341]);

plot(Tmax_vals, 0.001*optimal_design.vap_h2_pdt,    '-o', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.707 0.141],'MarkerFaceColor',[0 0.707 0.141]);
plot(Tmax_vals, 1*optimal_design.LCOE,    '-d', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.047 0.341],'MarkerFaceColor',[0 0.047 0.341]);
plot(Tmax_vals, 0.1*optimal_design.W_sys75,    '-^', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.189 0.841],'MarkerFaceColor',[0 0.189 0.841]);
plot(Tmax_vals, 0.1*optimal_design.LF ,    '-p', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.947 0.791],'MarkerFaceColor',[0 0.947 0.791]);
plot(Tmax_vals, 1*optimal_design.curtail,    '-g', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.381 0.901],'MarkerFaceColor',[0 0.381 0.901]);
plot(Tmax_vals, 1*optimal_design.maxT,    '-r', 'LineWidth',2,'MarkerSize',8,'Color',[.80 0.547 0.782],'MarkerFaceColor',[.80 0.547 0.782]);

xlabel('minimum load/rated load (%)','FontWeight','bold','FontSize',16);
ylabel('Obj. function value','FontWeight','bold','FontSize',16);
legend({ 'CAPEX dolperkg. (LCOH)',  ...
        'Energy dolperkg (LCOH)', ...
        '0.001 \times vap_h2_pdt (LCOH)', ...
        'LCOE (LCOH)', ...
        ' 0.1 \times W_sys75 (LCOH)', ...
        ' 0.1 \times LF (LCOH)', ...
        ' curtail (LCOH)', ...
        ' maxT (LCOH)'}, ...
        'FontSize',11, 'Location','best','Box','off');
title('Obj. functions vs. minimum load/rated load (%)');
ax = gca; 
ax.Box = 'on';         % draws the frame around the axes
ax.LineWidth = 1.5;    % set the frame thickness (adjust as needed)

hold off;
subplot(1,4,3); hold on; grid off;

plot(Tmax_vals, optimal_design.LCOE_RES,    '-s', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.447 0.741],'MarkerFaceColor',[0 0.447 0.741]);

plot(Tmax_vals, 10^-6*optimal_design.TIC_sys,    '-x', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.747 0.341],'MarkerFaceColor',[0 0.747 0.341]);

plot(Tmax_vals, 1*optimal_design.SEC_sys,    '-o', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.707 0.141],'MarkerFaceColor',[0 0.707 0.141]);
plot(Tmax_vals, 1*optimal_design.repl_cost,    '-d', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.047 0.341],'MarkerFaceColor',[0 0.047 0.341]);
plot(Tmax_vals, 0.001*optimal_design.TICperkW,    '-^', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.189 0.841],'MarkerFaceColor',[0 0.189 0.841]);
plot(Tmax_vals, 0.1*optimal_design.n_starts,    '-r', 'LineWidth',2,'MarkerSize',8,'Color',[.80 0.547 0.782],'MarkerFaceColor',[.80 0.547 0.782]);
%plot(Tmax_vals, 0.01*optimal_design.standby_time,    '--', 'LineWidth',2,'MarkerSize',8,'Color',[.80 0.547 0.782],'MarkerFaceColor',[.40 0.547 0.782]);
plot(Tmax_vals, 1*optimal_design.degrate_eff,    '-*', 'LineWidth',2,'MarkerSize',8,'Color',[.80 0.547 0.782],'MarkerFaceColor',[.80 0.547 0.782]);
% plot(Tmax_vals, 0.1*optimal_design.LF ,    '-p', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.947 0.791],'MarkerFaceColor',[0 0.947 0.791]);
% plot(Tmax_vals, 1*optimal_design.curtail,    '-g', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.381 0.901],'MarkerFaceColor',[0 0.381 0.901]);
% plot(Tmax_vals, 1*optimal_design.maxT,    '-r', 'LineWidth',2,'MarkerSize',8,'Color',[.80 0.547 0.782],'MarkerFaceColor',[.80 0.547 0.782]);

xlabel('minimum load/rated load (%)','FontWeight','bold','FontSize',16);
ylabel('Obj. function value','FontWeight','bold','FontSize',16);
legend({%'H2 annual_{sys} (LCOH)',   ...
        'LCOE_RES',  ...
        'CAPEX (M$) (LCOH)', ...
        'SEC_sys (LCOH)', ...
        'repl_cost(%) (LCOH)', ...
        'TIC per MW (LCOH)', ...
        'n_starts (LCOH)', ...
        'degrate_eff (LCOH)'}, ...
        'FontSize',11, 'Location','best','Box','off');
title('Obj. functions vs. minimum load/rated load (%)');
ax = gca; 
ax.Box = 'on';         % draws the frame around the axes
ax.LineWidth = 1.5;    % set the frame thickness (adjust as needed)

hold off;



%% Subplot 3: Parameter values (distinct colors)
subplot(1,4,4); hold on; grid off;
plot(Tmax_vals, params_meanbest(:,1),   '-s', 'LineWidth',2,'MarkerSize',8,'Color',[0 0.447 0.741],'MarkerFaceColor',[0 0.447 0.741]);
plot(Tmax_vals, params_meanbest(:,2),   '-o', 'LineWidth',2,'MarkerSize',8,'Color',[0.85 0.325 0.098],'MarkerFaceColor',[0.85 0.325 0.098]);
plot(Tmax_vals, 100*params_meanbest(:,3),'-^', 'LineWidth',2,'MarkerSize',8,'Color',[0.929 0.694 0.125],'MarkerFaceColor',[0.929 0.694 0.125]);
plot(Tmax_vals, 0.005*params_meanbest(:,4),'-d','LineWidth',2,'MarkerSize',8,'Color',[0.494 0.184 0.556],'MarkerFaceColor',[0.494 0.184 0.556]);
plot(Tmax_vals, params_meanbest(:,5),   '-x', 'LineWidth',2,'MarkerSize',10,'Color',[0.466 0.674 0.188],'MarkerFaceColor',[0.466 0.674 0.188]);
plot(Tmax_vals, 0.0001*params_meanbest(:,6),'-p','LineWidth',2,'MarkerSize',10,'Color',[0.301 0.745 0.933],'MarkerFaceColor',[0.301 0.745 0.933]);
dx = 0.2;              % small x-offset so text does not overlap marker
dy = 0;                % y-offset if needed
for k = 1:numel(Tmax_vals)
    txt = sprintf('%.2f', optimal_design.SEC_sta(k));   % 2 digits after decimal

    text(Tmax_vals(k)+dx, 0.0001*params_meanbest(k,6)+dy, txt, ...
         'FontSize', 10, 'Color', 'k');
end



for k = 1:numel(Tmax_vals)
    text(Tmax_vals(k)+dx, params_meanbest(k,2)+dy, ...
         num2str(N_comp(k)), ...
         'FontSize', 10, 'Color', 'k');
end
xlabel('minimum load/rated load (%)','FontWeight','bold','FontSize',16);
ylabel('Parameter value','FontWeight','bold','FontSize',16);
legend({'Electrode width (mm) (LCOH)', ...
        'Pressure (bar) (LCOH)', ...
        '100 \times Inlet velocity (m/s)(LCOH)', ...
        '0.005 \times Pore (\mum)(LCOH)', ...
        'Separator width (mm)(LCOH)', ...
        '0.0001 \times Current density(LCOH)'}, ...
        'FontSize',13, 'Location','northwest','Box','off');
%title(sprintf('m\\_degrate = %.4f, genratio = %.2f,nj = %.2f, PV_rated = %.2f', mean(m_degrate),mean(gr_vals) , mean(n_j), mean(PV_rated)));
TileFigures;          % or tilefigs
%title(sprintf('m\\_degrate = %.4f, genratio = %.2f', m_degrate, genratio));
ax = gca; 
ax.Box = 'on';         % draws the frame around the axes
    ax.LineWidth =1.5;    % set the frame thickness (adjust as needed)

hold off;


