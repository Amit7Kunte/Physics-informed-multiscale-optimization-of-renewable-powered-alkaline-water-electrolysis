clc;
clear;
format long g
heggg=2;
close all;
tic
xa=0;xb=0;xc=0;xd=0;xe=0;xf=0;xg=0;xh=0;xiii=0;xjjj=0;xl=0;xm=0;xn=0;
xk=0;
base_degrate = 0.125*8.760; % base degadation rate in years
base = 397; %CEPCI cost index for base year in 2001.
now = 800;
CI = now/base; % cost index factor
conv_rate = 1.17; % Eur to USD conversion rate
sc=1; %scale factor
Nc =5*sc; % No. of cells in stack
gc = 10^6;% This factor is used to multiply each functional unit, such as heat loads, shaft power etc, to bring the minimum size above the cost estimate bound. For the paper
rho_steel= 7850 ;% density of kg/m^3;
rho_Ni = 8900;
sd = 1;% BOP cost scale down factor
%scale exponents for various equipments: Reference turnton
nc= 1; % compressor
nhe=1;% HE
nv = 1;% vessels
nf = 1;% blower
np = 1;% pumps
% all vessels use SS clad construction
eta_pow = 0.98;
AllWsys70=[];
cd_eta75Vals=[];
Cp_cw = 4184 ; % cooling water heat capacity
T_cwr = 298.15 ; % cooling water return
T_cws = 293.15 ; % cooling water supply 
T_glsepHXout = 298.15;
T_evap = 17+273.15;
T_cond = 273.15 + 36; 
T_refrcomp = 273.15 + 45; 
T_amb = 273.15+25;
delT_aircool = 10;
T_aircooler = T_amb + delT_aircool;
U_desup = 30; % 30 value given in heuristics of book  of turnton Gas to Gas
U_cond_refr = 500; 
R_const = 8.314;
D_tube = 0.0254*1.5;
V_target = 3 ; %# target velocity of liquid flow  inside HX at rated capacity
pt=1.25*D_tube ; %# pitch length (triangular pitch)
de = 4.0 * (0.5 * pt * 0.86 * pt - 0.5 * pi * D_tube^2 / 4.0) / (0.5 * pi * D_tube) ; %# Equivalent diameter for shell pressure drop calc. from KERN (triangular pitch)
lam = -0.5;
alpha_delPHX = 0.2;
alpha_delPlyect =0.2;
heater_frac_ratedpower = 0.05;
J_kWh = 2.7778E-7;
MH2 = 0.002;
HHV_H2 = 285.8 * 1000 * J_kWh / MH2;
k_isen = 1.41 ;%# as given in the The Techno- Economics of
eta_heater = 0.95;
Z_comp =1.05;
FF =0.9; %shell and tube HX correction factor
T_suc_comp = T_amb + 11;
delT_air = 10 ;
Cp_air = 1005 ; %# J/(kg*K)
rho_air = 1.18 ; %# kg/m^3
eta_mech_motor = 0.95; %
eta_mech_comp= 0.9; %
eta_isen_comp = 0.85; % 
P_final = 30;  % final h2 delivery pressure in bar abs
T_max_comp = 273.15 + 140 ;% max allowable h2 pressure
eta_fan =0.55 ;
delP_fan = 500;
%rc_max1 = ((T_max_comp/T_amb-1)*eta_isen_comp+1)^(k_isen/(k_isen-1)); % max. compression ratio with inlet at 25C
rc_max = ((T_max_comp/T_suc_comp-1)*eta_isen_comp+1)^(k_isen/(k_isen-1));% max. compression ratio with inlet at T_suc_comp 
T_deoxo_out=273.15+150 ;
T_glsephXout =298.15;





% filePattern = fullfile(pwd, 'iModel_Wgde_0.7_P_5_vin_0.0075_dpore_500_dpored_0_Wsep_0.1_Ls_2000_data.mat'); % adjust extension if needed
% fileList = dir(filePattern);

matFiles = dir(fullfile('iModel_Wgde_0.7_P_1_vin_0.025_dpore_500_dpored_0_Wsep_0.485_Ls_2000_datan.mat'));
%matFiles = dir(fullfile('iModel_Wgde_*_P_*_vin_*_dpore_*_dpored_0_Wsep_0.*_Ls_2000_datan.mat'));
fileNames = {matFiles.name};
excludeIdx = contains(fileNames, 'P_20_') | contains(fileNames, 'P_12_');
fileList = matFiles(~excludeIdx);
allInputscapex = [];
allOutputscapex = [];
allOutputsabscapex=[];
allCapexComponents = []; % Will become (Ncases × Ncomponents)
paramVals = []; % Value of 'vin' (or whichever parameter you swept)
   Wsys70Vals=[];
   absCapexVals=[]; allOutputHeatmap=[];
for k = 1:length(fileList)
    % Get full file name
    fileName = fullfile(pwd, fileList(k).name);
%%% collect filename params 
    tokens = split(fileName, '_');
    paramNames = {'Wgde', 'P', 'vin', 'dpore', 'Wsep'};
    params = zeros(1, length(paramNames));
    for i = 1:length(paramNames)
        idx = find(strcmp(tokens, paramNames{i}));
        if ~isempty(idx) && idx < length(tokens)
            val = str2double(tokens{idx+1});
            assert(~isnan(val), sprintf('Value for %s not found or invalid', paramNames{i}));
            params(i) = val;
        else
            error('Parameter %s not found in filename %s', paramNames{i}, fileName);
        end
    end
    data = load(fileName);
   
% Load all variables from the file into the workspace
    if max(data.currentDensity) < 10000
        warning('Skipping file %s: data only covers up to %.2f A/m^2', fileList(k).name, max(data.currentDensity));
        %skippedFiles{end+1} = fileList(k).name; % Add to skipped file list
        continue; % Skip remaining processing for this file
    end
fprintf('Loaded file: %s\n', fileList(k).name);
% Now you can access the variables directly by their names
% For example, if 'saveData' contained a variable 'result', you can use:
%disp(data);

paramNames = {'W_GDE_an', 'P', 'v_in', 'dp_cat', 'W_sep', 'Lin'};

patternExprs = {'Wgde_(\d+\.?\d*)', 'P_(\d+\.?\d*)', 'vin_(\d+\.?\d*)', ...
                'dpore_(\d+\.?\d*)', 'dpored_(\-?\d+\.?\d*)', 'Wsep_(\d+\.?\d*)', 'Ls_(\d+\.?\d*)'};


    modelName = fileName;
    fprintf('\nProcessing model: %s\n', modelName);

    % Extract values from filename
    extractedVals = nan(1, numel(patternExprs));
    for j = 1:numel(patternExprs)
        match = regexp(modelName, patternExprs{j}, 'tokens');
        if ~isempty(match)
            extractedVals(j) = str2double(match{1});
        else
            fprintf('Warning: Could not extract %s from filename.\n', patternExprs{j});
        end
    end
    Wgde = extractedVals(1)/1000; P= extractedVals(2) ; v_in= extractedVals(3) ; dpore_cat = extractedVals(4); dpore_an = extractedVals(4) +  extractedVals(5); Wsep = extractedVals(6);

    extractedVals(2)=extractedVals(2)*10^5; % convert P in name to Pa units 
    extractedVals(4)=extractedVals(4)*10^-3;
    extractedVals(5)=extractedVals(5)*10^-3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data loading from saved matlab files

xdataExpr = 'aveop_an_bipolar(+tcd.nIs)';

% Extract current density sweep data points
% currentDensity = mphglobal(model, xdataExpr);
numPoints = length(data.currentDensity);

xFine = linspace(min(numPoints), 15000, 500);
% Interpolate and extrapolate using piecewise cubic Hermite interpolation (pchip)
varNames = fieldnames(data); % List of all variables in your saved struct
xFine = linspace(min(data.currentDensity), 15000, 500);
Pg=P-1; %gauge pressure for Fp calc.
% Interpolate every variable except 'currentDensity' itself
for v = 1:numel(varNames)
    vName = varNames{v};
    dataValue = data.(vName);

    if strcmp(vName, 'currentDensity')
        continue
    end

    if numel(dataValue) == numel(data.currentDensity)
        % Interpolate vectors that match currentDensity
        interpName = [vName '_interp'];
        assignin('base', interpName, interp1(data.currentDensity, dataValue, xFine, 'pchip', 'extrap'));
    elseif isscalar(dataValue)
        % Directly assign scalars to base workspace
        assignin('base', vName, dataValue);
    end
    % You can optionally handle other cases (arrays/matrices) here
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Avg SEC consumption calc. over stack lifetime, and calc. of total
% SEC_avg_sys and then calc. electricity cost of the plant
J_kWh = 2.7778E-7;
MH2 = 0.002;
degrade_stack = 10; % max % degradation over stack lifetime
t_oper = 3000; % hrs/ yr
t_sys = 40; %in years
m_degrate = 0.001; % per K, degradation rate increase per K delT
T_thresh = 0;
%a=0.7;
%degrate = 0.125 + m_degrate*(exp(max(maxT_interp - T_thresh, 0)) - 1) ; % in % per 1000 hrs
maxT_interp =6*ones(length(maxT_interp),1);
degrate = base_degrate  + m_degrate.*max(maxT_interp' - T_thresh, 1).^(0.5).*exp(0.00045*xFine);

dur_stack = degrade_stack./degrate; % in  yrs
Nrep = floor((t_oper*t_sys)./(dur_stack*8760)); % the no. of stack replacements in plant lifetime. to ccompare SEC_avg_stack for various m, can also be interpreted as the avg SEC consumption of first stack, when Nrep = 1. and so on...
plot(xFine,Nrep)
hold on
%m_degrate = 0.001;
t_oper = 3000;
maxT_interp =6*ones(length(maxT_interp),1);
degrate = base_degrate  + m_degrate.*max(maxT_interp' - T_thresh, 1).^(0.5).*exp(0.0005*xFine); % default used is 0.00035
dur_stack = degrade_stack./degrate; % in  yrs
Nrep = floor((t_oper*t_sys)./(dur_stack*8760)); % the no. of stack replacements in plant lifetime. to ccompare SEC_avg_stack for various m, can also be interpreted as the avg SEC consumption of first stack, when Nrep = 1. and so on...
plot(xFine,Nrep)
%E_deg = 
%degrate = 0.125 + m_degrate * (exp(a * max(maxT_interp - T_thresh, 0)) - 1);

dur_rem = (t_oper*t_sys) - Nrep.*dur_stack;
% Approach 2, SEC_stack_interp is the start of life of stack, and
% thereafter SEC_stack increases with degrate
SEC_initial = SEC_stack_interp; % kWh/kg H₂ — from your simulation, constant
t_total = t_oper * t_sys; % total plant operating hours (e.g., 3000 h/yr × 30 yr)
SEC_end = SEC_initial * (1 + degrade_stack/100); % 10% higher at EOL
SEC_avg_per_stack = 0.5 * (SEC_initial + SEC_end); % average over one life
% Average SEC over partial final stack
if dur_rem > 0
    frac_life = dur_rem ./ dur_stack; % fraction of final stack life used
    SEC_end_partial = SEC_initial + frac_life .* (SEC_end - SEC_initial);
    SEC_avg_partial = 0.5 * (SEC_initial + SEC_end_partial);
else
    SEC_avg_partial = 0;
end
SEC_avg_stack = (Nrep .* dur_stack .* SEC_avg_per_stack + dur_rem .* SEC_avg_partial) ./t_total; % avg stack SEC over plant lifetime



% calc. SEC averaged over system
SEC_avg_sys = SEC_avg_stack +(heater_ads_interp+heater_deoxo_interp+W_comp_interp+W_fan_interp+W_pump_cws_interp+W_pump_dmw_interp+W_pump_oxlye_interp+W_pump_hylye_interp+heater_lye_interp+heater_anlye_interp+W_refrcomp_interp+W_refrfan_interp)*J_kWh./(vap_h2_pdt_interp*MH2);
SEC_avg_sys=SEC_avg_sys/eta_pow; % This value is currently not being used in the glob. optim GA code. This shud be ideally the value to minimize.


Tin = 353.15;
%Q_lyecooler = (data.T_gl_out>Tin).*(data.Q_gl_out.*(4.101*10^3-3.526*10^3.*data.w_KOH_gl_out+9.644*10^-1.*(data.T_gl_out-273)+1.776.*(data.T_gl_out-273).*data.w_KOH_gl_out).*data.rho_KOH_gl_out.*(data.T_gl_out-Tin)) + (data.T_angl_out>Tin).*(data.Q_angl_out.*(4.101*10^3-3.526*10^3.*data.w_KOH_angl_out+9.644*10^-1.*(data.T_angl_out-273)+1.776.*(data.T_angl_out-273).*data.w_KOH_angl_out).*data.rho_KOH_angl_out.*(data.T_angl_out-Tin)); 



% Find the current density at which stack efficiency is 70%
% Find current density at 70% stack efficiency — last occurrence
target_eta = 75; % Target efficiency in %

% Find all indices where eta_stack_interp crosses 70% (both up and down)
eta_diff = diff(sign(eta_stack_interp - target_eta));

% Find falling edge crossings (from above 70% to below) — last high-current point
falling_edges = find(eta_diff < 0); % Where efficiency drops below 70%

if isempty(falling_edges)
    % If no falling edge, use last rising edge
    rising_edges = find(eta_diff > 0);
    if isempty(rising_edges)
        error('No point reaches 70%% stack efficiency.');
    else
        idx = rising_edges(end); % Last rising edge
    end
else
    idx = falling_edges(end); % Last falling edge (highest current density at 70%)
end

% Interpolate between the two points around the crossing
x1 = xFine(idx);   y1 = eta_stack_interp(idx);
x2 = xFine(idx+1); y2 = eta_stack_interp(idx+1);

% Linear interpolation for exact 70%
cd_eta75 = x1 + (target_eta - y1) * (x2 - x1) / (y2 - y1);
%cd_eta75 = 10294;
fprintf('Current density at 75%% stack efficiency (high-current side): %.2f A/m^2\n', cd_eta75);
if cd_eta75<1000
    print('cd_eta75<1000')
end

%fprintf('Current density at 70%% stack efficiency: %.2f A/m^2\n', cd_eta75);

% If you want the corresponding cell voltage and rated power:
Ecell_eta70 = interp1(xFine, E_cell_interp, cd_eta75, 'linear', 'extrap');
A_cell = 0.008*0.05; % [m^2], set this to your stack's cell area
I_rated = cd_eta75 * A_cell;
P_rated = gc*I_rated * Ecell_eta70/1000; % in kW
%W_sys75= 0.001*interp1(xFine, W_sys_interp*gc, cd_eta75, 'linear', 'extrap');%system power consumption interpolated in kW
vap_h2_pdt_eta75 = interp1(xFine, vap_h2_pdt_interp*gc, cd_eta75, 'linear', 'extrap') 
%W_refrcomp_at_eta70   = 0.001*interp1(xFine, W_refrcomp_interp*gc, cd_eta75, 'linear', 'extrap') ; %in kW

SEC_stack_at_eta75 = interp1(xFine, SEC_stack_interp, cd_eta75, 'linear', 'extrap');
plot(data.currentDensity, data.SEC_stack,'o');
%hold on
eta_stack_at_eta75  = (HHV_H2 / SEC_stack_at_eta75) * 100;
glsep_O2_interp = (data.degas_eff_catO2*(O2_catout_interp-O2ref_gl_out_interp));

glsep_O2_at_eta75 = interp1(xFine, glsep_O2_interp*gc, cd_eta75, 'linear', 'extrap') ;


T_angl_out_at_eta75 = interp1(xFine, T_angl_out_interp, cd_eta75, 'linear', 'extrap') ;
T_gl_out_at_eta75   =  interp1(xFine, T_gl_out_interp, cd_eta75, 'linear', 'extrap') 
Q_cond_ads_at_eta75 = interp1(xFine, Q_cond_ads_interp*gc, cd_eta75, 'linear', 'extrap') 
Q_cond_deoxo_at_eta75 = interp1(xFine, Q_cond_deoxo_interp*gc, cd_eta75, 'linear', 'extrap') 
Q_cond_h2cool_at_eta75 = interp1(xFine, Q_cond_h2cool_interp*gc, cd_eta75, 'linear', 'extrap') 
T_reg_eta75= interp1(xFine, T_reg_interp, cd_eta75, 'linear', 'extrap') ;
w_KOH_angl_out_eta75 = interp1(xFine, w_KOH_angl_out_interp, cd_eta75, 'linear', 'extrap') ;
w_KOH_gl_out_eta75 = interp1(xFine, w_KOH_gl_out_interp, cd_eta75, 'linear', 'extrap') ;
Q_gl_out_at_eta75 = interp1(xFine, Q_gl_out_interp, cd_eta75, 'linear', 'extrap')*gc 
Q_angl_out_at_eta75 = interp1(xFine, Q_angl_out_interp, cd_eta75, 'linear', 'extrap')*gc ;
ancell_inP_at_eta75 = interp1(xFine, ancell_inP_interp, cd_eta75, 'linear', 'extrap');
ancell_delP_at_eta75 = ancell_inP_at_eta75 - P*10^5;
cell_inP_at_eta75 = interp1(xFine, cell_inP_interp, cd_eta75, 'linear', 'extrap');
cell_delP_at_eta75 = cell_inP_at_eta75 - P*10^5;
gasfrac_at_eta75 = interp1(xFine, gasfrac_interp, cd_eta75, 'linear', 'extrap');

% rho_KOH_gl_out_at_eta75 = interp1(xFine, rho_gl_out_interp, cd_eta75, 'linear', 'extrap') ;
% rho_KOH_angl_out_at_eta75 = interp1(xFine, rho_angl_out_interp, cd_eta75, 'linear', 'extrap') ; 

Cp_KOH_lye_at_eta75 = (4.101e3 - 3.526e3 .* w_KOH_gl_out_eta75 + ...
              9.644e-1 .* (T_gl_out_at_eta75 - 273.15) + ...
              1.776 .* (T_gl_out_at_eta75 - 273.15) .* w_KOH_gl_out_eta75);

Cp_KOH_anlye_at_eta75 = (4.101e3 - 3.526e3 .* w_KOH_angl_out_eta75 + ...
              9.644e-1 .* (T_angl_out_at_eta75 - 273.15) + ...
              1.776 .* (T_angl_out_at_eta75 - 273.15) .* w_KOH_angl_out_eta75);


rho_KOH_gl_out_at_eta75 = (1001.53 - 0.08343 .* (T_gl_out_at_eta75 - 273.15) - ...
                  0.004 .* (T_gl_out_at_eta75 - 273.15).^2 + ...
                  5.51232e-6 .* (T_gl_out_at_eta75 - 273.15).^3 - ...
                  8.21e-10 .* (T_gl_out_at_eta75 - 273.15).^4) .* exp(0.86 .* w_KOH_gl_out_eta75);

rho_KOH_angl_out_at_eta75 = (1001.53 - 0.08343 .* (T_angl_out_at_eta75 - 273.15) - ...
                    0.004 .* (T_angl_out_at_eta75 - 273.15).^2 + ...
                    5.51232e-6 .* (T_angl_out_at_eta75 - 273.15).^3 - ...
                    8.21e-10 .* (T_angl_out_at_eta75 - 273.15).^4) .* exp(0.86 .* w_KOH_angl_out_eta75);



% lye heater duty at rated load, kW
Q_lyeheater_at_eta75 = 0.001 * ( ...
    max( (273.15 + 80 - T_gl_out_at_eta75) .* Cp_KOH_lye_at_eta75 .* Q_gl_out_at_eta75 .* rho_KOH_gl_out_at_eta75, 0 ) + ...
    max( (273.15 + 80 - T_angl_out_at_eta75) .* Cp_KOH_anlye_at_eta75 .* Q_angl_out_at_eta75 .* rho_KOH_angl_out_at_eta75, 0 ) );


Q_lyecooler_at_eta75 = max((T_gl_out_at_eta75-273.15-80)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0) + max((T_angl_out_at_eta75-273.15-80)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0)

% cooling water circulation rate at eta75, kg/s
m_cw_circ_at_eta75 = ( Q_lyecooler_at_eta75 + Q_cond_ads_at_eta75 + ...
                       Q_cond_deoxo_at_eta75 + Q_cond_h2cool_at_eta75 ) ...
                     ./ ( Cp_cw .* (T_cwr - T_cws) );






% elec_cost = .05; % 4 $/kWh;
% opex_elec = elec_cost*SEC_avg_sys; % $/kg H2 delivered

disc_rate = .08 ;% discount rate;
CRF = (disc_rate*(1+disc_rate)^t_sys)/((1+disc_rate)^t_sys-1); % capital recovery factor;

%%% Bottom up capex cost calculation of Stack components
od_pumps =1.15;% overdeisgn factor
F = 0.9;% correction  factor for HX, typically 0.9 for shell and tube
T_cws = 293.15; %in K
T_cwr = T_cws + 5 ;
% STACK
Ni_foam = 0.5/10^-6; % $0.05/cm^3,https://shop.nanografi.com/battery-equipment/nickel-foam-for-battery-cathode-substrate-size-1000-mm-x-300-mm-x-1-6-mm/
zf_sep = gc*150*0.05*0.008*conv_rate*Nc; % 150 Euro/m^2 from AGFA Data 'Present and future cost of alkaline and PEM electrolyser stacks'
steel = gc*(0.05*0.008*0.003*2+0.05*0.008*0.001*(Nc-1))*rho_steel*4.5;    % 0.9 $ /kg for carbon steel, 4.5$/kg for SS316L. SS used for longer stack life and in advanced stack designs operating at higher cd. SOURCE:https://www.imarcgroup.com/stainless-steel-316-price-trend
Ni =  gc*2*(0.05*0.008*Wgde*(Nc))*Ni_foam; % 15 $/kg Ni price as on 4/10/2025, based on volume of electrode
Fp_stack = 1+0.2*(P-1)/(15-1);% Saba etal 2018, reported around 20% increase in stack costs for a pressure increase from 1 bar to 15 bar, assuming a linear relationship between operating pressure and stack material costs
stack_mat_cost = 1.3*(zf_sep+steel+Ni)*Fp_stack;  % $/kW, 30% additional for balance of stack components such as gaskets etc.
dir_stack_cost = stack_mat_cost+ 0.5*stack_mat_cost+0.125*stack_mat_cost; % ratio is as 8:4:1 from 'Present and future cost of alkaline and PEM electrolyser stacks'.
Tot_stack_cost = 2*dir_stack_cost; % including overheads $/kW


% Gas–Liquid separator vessels
% Pressure vessels, 2 min residence time

od_glsep = 1.15;

Pg = P - 1;                              % gauge pressure

vol_liq_glsep = 120 * (Q_gl_out_at_eta75+ Q_angl_out_at_eta75);  % m^3
vol_glsep     = od_glsep * vol_liq_glsep;                          % overdesigned volume

glsep_D = ((4 * vol_glsep) / (3 * pi))^(1/3);   % diameter [m]
glsep_L = 3 * glsep_D;                          % length [m]

% Purchase cost
CP_glsep = 2 * 10^(3.5565 + ...
    0.3776 * log10(max(vol_glsep, 0.1)) + ...
    0.09   * (log10(max(vol_glsep, 0.1)))^2);

CP_glsep_min = 2 * 10^(3.5565 + ...
    0.3776 * log10(0.1) + ...
    0.09   * (log10(0.1))^2);

% Pressure‑vessel factor
Fp_glsep = max(((Pg + 1) * glsep_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0);

FBM_glsep  = 1.49 + 1.52 * 1.7 * Fp_glsep;

BM_glsep   = FBM_glsep * CP_glsep * CI;
BM_0_glsep = CP_glsep * (1.49 + 1.52) * CI;

% 6/10 rule if minimum capacity governs
if (CP_glsep == CP_glsep_min) || (vol_glsep > 1200)
    BM_glsep_min = CP_glsep * FBM_glsep;
    BM_glsep     = 2 * sd * BM_glsep_min * (vol_glsep / 0.1)^nv;
    BM_0_glsep   = CP_glsep_min * (1.49 + 1.52) * CI;
end

adspec_H2O = 5e-06;
od_ads     = 1.2;
m_ads      = 0.03 * gc;                 % adsorbent mass [kg]
t_b        = 8*3600;                    % s
t_des      = 5/8 * t_b;
t_cool     = 3/8 * t_b;
por_ads    = 0.3;                       % porosity of adsorber
phi        = 1.0;                       % sphericity
ads_dia    = 0.0015875;                 % 1/16 inch [m]
B          = 0.152;
C          = 0.000316;

rho_ads    = 650;                       % density of adsorbent [kg/m^3]
vol_ads    = m_ads ./ rho_ads;         % adsorbent volume [m^3]
vol_ads_g  = od_ads .* mean(vol_ads);   % (check original intent)

ads_D      = 3.0;                       % adsorber diameter [m]
L_ads      = vol_ads ./ (pi .* ads_D.^2 / 4);  % bed height [m]

mu_H2_25   = 1.95e-5;                   % Pa·s
rho_H2_25  = 0.08;                    % kg/m^3

eta_ads    = 0.6;                       % utilization factor
q_max      = 19.43;                      % mol/kg max. capacity
purge_ads  = 0.5 * 10^(-4) * gc;        % mol/s purge gas flow (check units)

Cp_ads     = 900.0;                     % J/(kg·K)
H_ads      = 50000;                     % J/mol, heat of adsorption
b_ref       = 93.5;                      % ???
alpha_purge = 0.5;
Cp_H2_80 = 29;
Cp_O2_80 = 29.72;
Cp_KOH = 3156.1;
Cp_watvap_80 = 34.6;
lat_heat_water = 44000;
rho_cw = 1000; %kg/m^3 
mu_cw = 0.001;
rho_steel= 7850; %# density of kg/m^3;
Cp_steel = 475; %# J/kg/K, Cp CS is also same as Cp SS
rho_Ni = 8900;
Cp_Ni = 440;


p_vapor_h2cool_out = 0.61121*exp((18.678-(T_glsepHXout-273.15)/234.5)*((T_glsepHXout-273.15)/(257.14+T_glsepHXout-273.15)))*1000;

b_T    = b_ref*exp(H_ads ./ R_const .* (1./T_glsephXout - 1/293.15));  % b at 25°C
p_H2O_adspec = mean(adspec_H2O .* P);   % example: adjust index as needed
 dry_gas_at_eta75 = vap_h2_pdt_eta75*(1-adspec_H2O);
x_H2O_h2cool_out = p_vapor_h2cool_out ./ (P*1e5);
%vap_H2_pdt_eta       = vap_h2_pdt_interp(x(1));  % from your interpolation function
vap_ads_at_eta75 = vap_h2_pdt_eta75 * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) ;   % to adsorber

ads_H2O_in_at_eta75 = (dry_gas_at_eta75 .* x_H2O_h2cool_out) ./ (1 - x_H2O_h2cool_out); % check formula
ads_H2O_out_at_eta75 = (vap_ads_at_eta75*adspec_H2O)/(1-adspec_H2O);                 % as in Python

ads_cap_rate_at_eta75 = ads_H2O_in_at_eta75-ads_H2O_out_at_eta75 ;%#; % rate of moisture capture required
qcc_at_eta75 = ads_cap_rate_at_eta75*t_b/(m_ads*eta_ads) ;% #; % anticipated ads cc density in mol/kg
ads_cc_at_eta75 = m_ads*qcc_at_eta75*eta_ads ;%#; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required

ads_cap_rate_at_eta75 = ads_cc_at_eta75/t_b;

q_res_at_eta75 = q_max-qcc_at_eta75 ;%# excess residual capacity above the reqd capture rate
rq_at_eta75 = q_res_at_eta75 / q_max ;%# fraction of reduction in capacity after reqd capture capacity of the adsorbent
des_rate_at_eta75 = ads_cc_at_eta75 / t_des ;%#; % the reqd des_rate increases with current density as more mositure is in gas stream.

rp_at_eta75 = purge_ads/(des_rate_at_eta75) ;%#; % ratio of purge flow rate to des rate (based on desorb time)

xH2O_reg_out_at_eta75 = (x_H2O_h2cool_out+1.0/rp_at_eta75)/(1.0+1.0/rp_at_eta75);% #; % mositure moefrac at regen exit

p_H2O_reg_out_at_eta75 = P*10^5*(x_H2O_h2cool_out+1.0/rp_at_eta75)/(1.0+1.0/rp_at_eta75);

p_vap_reg_eff_at_eta75 = P*10^5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out_at_eta75-x_H2O_h2cool_out));

b_reg_at_eta75 = ((rq_at_eta75^0.2472)/(1.0-rq_at_eta75^0.2472))^(1.0/0.2472)*(1.0/p_vap_reg_eff_at_eta75);

T_reg_at_eta75 = max(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*log(b_reg_at_eta75/b_ref))^-1); %# T_reg is much more sensitive to the rq (residual ads cc after required adsorption as per specifications) than the purge gas circulation rate.

T_base_vec_eta_75 = T_glsepHXout ;%* np.ones(len(T_reg_at_eta75)) ;  %# replicate as in MATLAB

thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063); %# 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

m_steel_ads = ((pi * (ads_D / 2 + thick_ads) ^ 2 - pi * (ads_D / 2) ^ 2)* L_ads* rho_steel* 1.5);

heater_ads_at_eta75 = 0.001*(purge_ads * Cp_H2_80 * (T_reg_at_eta75 - T_base_vec_eta_75) + des_rate_at_eta75 * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg_at_eta75- T_base_vec_eta_75) / t_des / eta_heater) * t_des / t_b);
therm_cat_ads = m_ads*Cp_ads ;
therm_st_ads = m_steel_ads*Cp_steel;

condensate_H2O_regen_cond = (purge_ads * (1.0 - x_H2O_h2cool_out) * xH2O_reg_out_at_eta75 / (1.0 - xH2O_reg_out_at_eta75)- x_H2O_h2cool_out * purge_ads); % condensate rate from the regeneration condenser
cond_ads = (condensate_H2O_regen_cond*lat_heat_water+ purge_ads * Cp_H2_80 * (T_reg_at_eta75 - T_glsepHXout)) * t_des / t_b ; % heat duty of condensate adsrober
m_cw_cond_ads = cond_ads / (Cp_cw * (T_cwr - T_cws));

% purchase cost of 2 adsorber cols
CP_ads = 2 * 10 ^ (3.4974 + 0.4485 * log10(max(vol_ads, 0.3)) + 0.1074 * (log10(max(2 * vol_ads, 0.3)))^2);

CP_ads_min = 2 * 10 ^ (3.4974+ 0.4485 * log10(0.3)+ 0.1074 * (log10(0.3))^2);

     % # Pressure vessel factor
      Fp_ads = max(((Pg + 1) * ads_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0);

      FBM_ads = 2.25 + 1.82 * 1.7 * Fp_ads;

      BM_ads = FBM_ads * CP_ads  * CI;
      BM_0_ads = CP_ads*(2.25 + 1.82) * CI;

     % # 6/10 rule for minimum capacity
      if (CP_ads_min == CP_ads) || (vol_ads > 520)
          BM_ads_min = CP_ads_min * FBM_ads;
          BM_ads = sd * BM_ads_min * (vol_ads / 0.3)^nv*CI;
          BM_0_ads = CP_ads_min *(2.25 + 1.82)*CI;
          self.xc = self.xc+1;
      end    

    od_deoxo = 1.2;
    GHSV = 2000.0;  % # per hr, reference ......
    tcycle_deoxo = 8*3600; %# number of seconds per cycle (8 hrs)
    %# Volume and geometry
    vol_deoxo = od_deoxo * max(0.001* vap_h2_pdt_eta75 * 22.414 * 3600.0 / GHSV, 0.0) ;
    D_deoxo = 1.0;  % # m
    L_deoxo = vol_deoxo / (pi * D_deoxo^2 / 4.0) ;
    thick_deoxo = max(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063); % # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness
    mass_deoxo = vol_deoxo*650; % # bulk density of deoxo catalyst is 650 kg/m^3
    Cp_deoxo = 900;
    mass_st_deoxo = (pi*(D_deoxo+thick_deoxo)^2.0/4.0*L_deoxo-pi*(D_deoxo)^2.0/4.0*L_deoxo)*rho_steel ;% #%2mm thickness
    therm_st_deoxo = mass_st_deoxo * Cp_steel / tcycle_deoxo ; % #* np.ones(len(deoxo_heat)) % thermal absorption rate of steel in the deoxidizer
    therm_cat_deoxo = mass_deoxo * Cp_deoxo / tcycle_deoxo;  %#* np.ones(len(deoxo_heat)) % thermal absorption rate of catalyst in the deoxidizer
    deoxo_H2O    = 2.0 * glsep_O2_at_eta75;
    deoxo_H2reac = 2.0 * glsep_O2_at_eta75;
    deoxo_heat_eta75    = 2.0 * glsep_O2_at_eta75 * 244.9 * 10.0^3 ;  %# 244.9 kJ/mol → J/mol

    Cp_vap_h2cool = (Cp_watvap_80 * x_H2O_h2cool_out+ Cp_H2_80 * (1-x_H2O_h2cool_out));
    
    T_deoxo_at_eta75   = ( deoxo_heat_eta75 + therm_cat_deoxo * (T_glsepHXout + 125.0)+ therm_st_deoxo * (T_glsepHXout + 125.0)+ vap_h2_pdt_eta75 * Cp_vap_h2cool * T_glsepHXout) / (vap_h2_pdt_eta75 * Cp_vap_h2cool + therm_st_deoxo + therm_cat_deoxo);

    deoxo_T = 150 +273.15; %# deoxo maintains temp. at 150C
    heater_deoxo_at_eta75 = 0.001*((deoxo_T > T_deoxo_at_eta75 )* vap_h2_pdt_eta75* Cp_vap_h2cool* (deoxo_T - T_deoxo_at_eta75)/ eta_heater);
    heater_regen = 0.001*((therm_st_deoxo + therm_cat_deoxo)* (723.15 - deoxo_T)/ eta_heater); %# regen temp of deoxo is 450C

    % purchase cost of 2 deoxo cols
    CP_de = 2 * 10 ^ (3.4974 + 0.4485 * log10(max(vol_deoxo, 0.3)) + 0.1074 * (log10(max(vol_deoxo, 0.3)))^2);

    CP_de_min = 2 * 10 ^ (3.4974+ 0.4485 * log10(0.3)+ 0.1074 * (log10(0.3))^2);
    Fp_deoxo = max(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0);

    FBM_de = 2.25 + 1.82 * 1.7 * Fp_deoxo;

    BM_de = sd * FBM_de * CP_de* CI;
    BM_0_de = (2.25 + 1.82)*CP_de*CI;
    
    %# 6/10 rule when minimum capacity governs
    if (CP_de_min == CP_de) || (vol_deoxo/2 > 520)
          BM_de_min = CP_de_min * FBM_de ;
          BM_de = sd * BM_de_min * (vol_deoxo / 0.3)^nv ;
          BM_0_de = CP_de_min * CI*(2.25 + 1.82);
          self.xd = self.xd + 1;
    end


    P_final =30 ;
    Ncomp = ceil(log(P_final / (P )) / log(rc_max));
    Ncomp = max(Ncomp, 1); %       # at least one stage
    
    od_h2comp = 1.15;
    
    if Ncomp <= 0
    %  # physically impossible, punish the design
    rc = 1.0;
    elseif Ncomp == 1
    %# single‑stage compression: all ratio in one step
    rc = rc_max; %#P_final / (x[1] * rc_max1);
    else
    %# multi‑stage: equal ratio per stage
    rc = (P_final / (P))^(1.0 / (Ncomp)); % uniform rc across all compressors
    end
    
    T_dis_comp = T_suc_comp * (1 + (rc^((k_isen - 1.0) / k_isen) - 1) / eta_isen_comp);  
    if T_dis_comp > (273.15+140)
        Tdiscomp = T_dis_comp-(273.15+140);
    end    
    % if P_final / P > 1.0
    %   term_over = (k_isen / (k_isen - 1.0) *R_const * T_amb * Z_comp *(rc_max1 ^ ((k_isen - 1.0) / k_isen) - 1.0) * vap_h2_pdt_eta75 / (eta_isen_comp  * eta_mech_comp));
    % else
        term_over = 0.0;
    % end    
    
    W_shaft_h2comp = 0.001*(max(Ncomp, 0) * k_isen / (k_isen - 1.0) *R_const * T_suc_comp * Z_comp *(rc ^ ((k_isen - 1.0) / k_isen) - 1.0) *vap_h2_pdt_eta75  / (eta_isen_comp *  eta_mech_comp)+ term_over); % # in kW, at rated current density for CAPEX calc., assuming uniform work across all comps based on uniform rc
    intercool_comp_at_eta75 = (T_dis_comp - T_suc_comp) * vap_h2_pdt_eta75 * Cp_H2_80 ;
    
    air_fan = intercool_comp_at_eta75/ (delT_air * Cp_air * rho_air); % # [web:1]
    
    W_fan = 0.001*air_fan * delP_fan / eta_fan; % # [web:1]
    
    W_h2comp = W_shaft_h2comp/eta_mech_motor; %# in kW
    CP_shaft_h2comp=Ncomp*(10^(2.2897+1.36*log10(max((od_h2comp*W_shaft_h2comp)/Ncomp,50))-0.1027*(log10(max((od_h2comp*W_shaft_h2comp)/Ncomp,50)))^2)); % min. input pow is 450kW reciprocating, max is 3000kW per stage
    CP_shaft_h2comp_min =  Ncomp*(10^(2.2897+1.36*log10(max(50))-0.1027*(log10(max(50) ))^2)); % min. input pow is 450kW as per turton, but using limit of 300, as the function is smooth
    FBM_shaft_h2comp = 7;
    BM_shaft_h2comp = sd * FBM_shaft_h2comp * CP_shaft_h2comp * CI;
    BM_0_shaft_h2comp = BM_shaft_h2comp / FBM_shaft_h2comp;
    if (CP_shaft_h2comp_min == CP_shaft_h2comp) || (od_h2comp*W_shaft_h2comp/Ncomp > 3000)
    BM_shaft_h2comp_min = CP_shaft_h2comp_min * FBM_shaft_h2comp * CI;
    BM_shaft_h2comp = (sd* BM_shaft_h2comp_min* ((od_h2comp*W_shaft_h2comp / Ncomp) / 50.0) ^ 0.86);
    BM_0_shaft_h2comp = BM_shaft_h2comp / FBM_shaft_h2comp * CI;
    self.xf = self.xf + 1;
    end
    
    od_refrcomp =1.20;
    eta_isen_refrcomp = 0.7;  
    refr_lat_heat_evap = 199.3 * 1000 ;%  # %j/kg, this is not used in the calc. of the m_refr, as the expansion valve outlet is a two phase mixture at the t and p of the evaporator ( both lower than cond)

    refr_S_evap = 1781.8; %# vapor entropy of Freon at 17C
    Pevap = 1323.7 ; %#kPa, pressure of evaporator at 17C
    refr_H_comp_s = 444.9*1000 ; %#443.6*1000 was for 50C #% 60C at the sampe entropy of the evap, but pressure of condenser,ideal compression
    refr_H_cond_vap = 424.6*1000 ; %# 427.0 * 1000 was for 40C # condensate vapor enthalpy
    refr_H_cond_liq = 286.9*1000 ; %# 267.1*1000 was for 40C
    refr_H_evap_vap = 426.4*1000 ;
    refr_H_evap_liq = 227.1*1000 ;
    Pcond = 3051 ;%2411 was for 40C
    rho_dm =1000 ;
    refr_lat_evap = refr_H_evap_vap-refr_H_cond_liq ;%#; % This is the actual latent heat from 2 phase mixture to vapor phase in the evap.
    refr_H_comp = refr_H_evap_vap+(refr_H_comp_s-refr_H_evap_vap)/eta_isen_refrcomp; %#; ie 452.828 kJ/kg, coressponding to an actual comp discharge temp. of 65C % enthalpy at comp discharge for non-isen condition
    
    cond_duty = refr_H_comp - refr_H_cond_liq ;
    
    delH_comp = refr_H_comp-refr_H_evap_vap;
    
    %#%refr_H_cond = 427.4*1000;
    
    refr_lat_heat_cond= 137.7*1000;            %  # %   considering condensation at 50C , 159.9*1000 ( was for 40C);
    delH_desup = refr_H_comp-refr_H_cond_vap ;  %  # enthalpy loss in the desuperheater
    m_refr_at_eta75 = m_cw_circ_at_eta75*Cp_cw*(T_cwr-T_cws)/refr_lat_evap;
    W_refrcomp_at_eta75 = 0.001*(delH_comp*m_refr_at_eta75/(eta_mech_comp*eta_mech_motor));
    W_refrfan_at_eta75 = 0.001*((m_refr_at_eta75*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan;
    W_refrcompshaft_at_eta75 = W_refrcomp_at_eta75*eta_mech_motor; % # in kW

    CP_shaft_refrcomp = 10 ^ (2.2897+ 1.36 * log10(max(od_refrcomp*W_refrcompshaft_at_eta75, 50.0))- 0.1027 * (log10(max(od_refrcomp*W_refrcompshaft_at_eta75, 50.0))) ^ 2);
    CP_shaft_refrcomp_min = 10 ^ (2.2897+ 1.36 * log10(50.0)- 0.1027 * (log10(50.0)) ^ 2);
    FBM_shaft_refrcomp = 3.3;
    %# Base shaft costs
    BM_shaft_refrcomp = sd * FBM_shaft_refrcomp * CP_shaft_refrcomp* CI;
    BM_0_shaft_refrcomp = BM_shaft_refrcomp / FBM_shaft_refrcomp;
    %# Refr. shaft min‑capacity check
    if (CP_shaft_refrcomp_min == CP_shaft_refrcomp) || (od_refrcomp*W_refrcompshaft_at_eta75 > 3000)
          BM_shaft_refrcomp_min = CP_shaft_refrcomp_min * FBM_shaft_refrcomp* CI;
          BM_shaft_refrcomp = (sd * BM_shaft_refrcomp_min * (od_refrcomp*W_refrcompshaft_at_eta75 / 50.0) ^ 0.86);
          BM_0_shaft_refrcomp = BM_shaft_refrcomp / FBM_shaft_refrcomp;
          self.xe = self.xe +1    ;
    end

    CP_refrcomp = 10 ^ (2.9308+ 1.0688 * log10(max(od_refrcomp*W_refrcomp_at_eta75, 5.0))- 0.1315 * (log10(max(od_refrcomp*W_refrcomp_at_eta75, 5.0))) ^ 2);
    CP_refrcomp_min = 10 ^ (2.9308+ 1.0688 * log10(75.0)- 0.1315 * (log10(75.0)) ^ 2);
    CP_h2comp = Ncomp * 10 ^ (2.9308+ 1.0688 * log10(max(od_h2comp*W_h2comp / Ncomp, 5.0))- 0.1315 * (log10(max(od_h2comp*W_h2comp / Ncomp, 5.0))) ^ 2); % # min. drive power is 75kW,2600kW max
    CP_h2comp_min = Ncomp * 10 ^ (2.9308+ 1.0688 * log10(5.0)- 0.1315 * (log10(5.0)) ^ 2);
    FBM_comp = 1.5;
    BM_refrcomp = sd * FBM_comp * CP_refrcomp*CI;
    BM_h2comp   = sd * FBM_comp * CP_h2comp*CI;
    BM_0_h2comp   = BM_h2comp   / FBM_comp;
    BM_0_refrcomp = BM_refrcomp / FBM_comp;
      %# Refrigeration compressor min-capacity scaling
    if (CP_refrcomp == CP_refrcomp_min) || (od_refrcomp*W_refrcomp_at_eta75> 2800)
          BM_refrcomp_min = CP_refrcomp_min * FBM_comp;
          BM_refrcomp = sd * BM_refrcomp_min * (od_refrcomp*W_refrcomp_at_eta75 / 5.0) ^ 0.6 * CI;
          BM_0_refrcomp = BM_refrcomp / FBM_comp;
          self.xg = self.xg + 1;
    end      
     % # H2 compressor min-capacity scaling
    if (CP_h2comp == CP_h2comp_min) || (od_h2comp*W_h2comp / Ncomp > 2600)
          BM_h2comp_min = CP_h2comp_min * FBM_comp;
          BM_h2comp = sd * BM_h2comp_min * (od_h2comp*W_h2comp / 5.0) ^ 0.6*CI;
          BM_0_h2comp = BM_h2comp / FBM_comp;
          self.xh = self.xh + 1;
    end

    %# intercoolers , air cooled condensers and fans 
    %  # Intercooler duty
      od_aircool = 1.20;
     
      
      U_intercool = 30; % #%%%%%%%%%%%%%%% GAS TO GAS is 30
      delTlm_intercool = ((T_dis_comp - T_aircooler) - (T_suc_comp - T_amb)) / log((T_dis_comp - T_aircooler) / (T_suc_comp - T_amb));

      area_intercool = intercool_comp_at_eta75 / (U_intercool * delTlm_intercool*FF); %# area of each set of intercoolers per compressor;


      %# Number of intercoolers
      N_intercool = max(1, (floor(od_aircool* area_intercool / 20000.0 + 0.8))); %# No. of intercool, per compressor

      %# Purchase cost
      CP_intercool = (Ncomp)*(N_intercool * 10 ^ (4.0336+ 0.2341 * log10(max(od_aircool*area_intercool  / N_intercool, 1.0))+ 0.0497 * (log10(max(od_aircool*area_intercool  / N_intercool, 1.0))) ^ 2));

      CP_intercool_min = (Ncomp)*(N_intercool * 10 ^ (4.0336+ 0.2341 * log10(1.0)+ 0.0497 * (log10(1.0)) ^ 2));

      FBM_intercool = 0.96 + 1.21 * 2.9 ;  %# SS construction

      BM_intercool = sd * FBM_intercool * CP_intercool*CI;
      BM_0_intercool = CP_intercool*(0.96 + 1.21)*CI;
      
      if (CP_intercool == CP_intercool_min) || (od_aircool*area_intercool / N_intercool > 11000)
          BM_intercool_min = CP_intercool_min * FBM_intercool * CI;
          BM_intercool = sd * BM_intercool_min * (od_aircool*area_intercool / 1.0) ^ nhe;
          BM_0_intercool = CP_intercool_min*(0.96 + 1.21)*CI;
          self.xll =self.xll + 1;
      end    

      %# aircooled refr condenser
      T_cond = 273.15 + 50; %#condenser saturation, ie 25C approach over ambient
      T_refrcomp = 273.15 + 65 ;%# refr comp outlet temp , 65 C is the actual comp o/l temp from the Freon thermo tables, at the condernser pressure, and the eta_isen of 0.7
      Q_desup = delH_desup * m_refr_at_eta75;
      Q_cond_refr = refr_lat_heat_cond * m_refr_at_eta75;
      U_desup = 30;
      U_cond_refr = 40; %# both are close, as airside controls HT

      delTlm_cond_refr = ((T_cond - (T_amb+9)) - (T_cond - T_amb)) / log((T_cond - (T_amb+9)) / (T_cond - T_amb)); %# Temp. rise of 1K is assumed across the desuperheater

      delTlm_desup = ((T_refrcomp - (T_aircooler)) - (T_cond - (T_amb+9))) / log((T_refrcomp - (T_aircooler)) / (T_cond - (T_amb+9)));

      area_desup = Q_desup / (U_desup * delTlm_desup*FF);
      area_cond_refr = Q_cond_refr / (U_cond_refr * delTlm_cond_refr*FF);
      area_refr = (area_desup + area_cond_refr); %# total area of the air cooled refrigerant condenser

      N_refr = max(1, (floor(od_aircool*area_refr / 20000.0 + 0.5)));% # max area for air cooler is 10000 m^2
      CP_refr_min = N_refr * 10 ^ (4.0336+ 0.2341 * log10(10.0)+ 0.0497 * (log10(10.0)) ^ 2);

      CP_refr = N_refr * 10 ^ (4.0336+ 0.2341 * log10(max(od_aircool*area_refr / N_refr, 10.0))+ 0.0497 * (log10(max(od_aircool*area_refr / N_refr, 10.0))) ^ 2);

      Fp_refr = 10 ^ (-0.125+ 0.15361 * log10(29.0 + eps)- 0.02861 * (log10(29.0 + eps)) ^ 2) ;%# 21 bar is the guage pressure of the refrigerant inside the vapor compression cycle

      FBM_refr = 0.96 + 1.21 * 1.0 * Fp_refr ; % # CS construction, 30 bar pressure

      BM_refr = sd * FBM_refr * CP_refr* CI;
      BM_0_refr = (0.96+1.21)*CP_refr*CI  ;

      if (CP_refr_min == CP_refr) || (od_aircool*area_refr / N_refr > 20000)
          BM_refr_min = CP_refr_min * FBM_refr * CI;
          BM_refr = sd * BM_refr_min * (od_aircool*area_refr / 10.0) ^ nhe;
          BM_0_refr = (0.96+1.21)*CP_refr_min*CI;
          self.xk = self.xk + 1;
      end
      %# Air flowrate and fan power
     
      od_fans = 1.20;
      air_fan = intercool_comp_at_eta75 / (delT_air * Cp_air * rho_air) ;
      W_compfan = 0.001*air_fan * delP_fan / eta_fan;
      N_fan = max(1, (floor(od_fans*air_fan / 100.0 + 0.8)));
      CP_fan = N_fan * 10 ^ (3.5391- 0.3533 * log10(max(od_fans*air_fan / N_fan, 0.01))+ 0.4477 * (log10(max(od_fans*air_fan / N_fan, 0.01))) ^ 2);

      CP_fan_min = N_fan * 10 ^ (3.5391- 0.3533 * log10(0.01)+ 0.4477 * (log10(0.01)) ^ 2);

      FBM_fan = 2.7;
      BM_fan = sd * FBM_fan * CP_fan * CI;
      BM_0_fan = BM_fan / FBM_fan;

      if (CP_fan == CP_fan_min) || (od_fans*air_fan / N_fan > 110)
          BM_fan_min = CP_fan_min * FBM_fan* CI;
          BM_fan = sd * BM_fan_min * (od_fans*air_fan / 0.01) ^ nf;
          BM_0_fan = BM_fan / FBM_fan ;
          self.xn = self.xn + 1 ;eta_isen
      end  
            
      air_refrfan = m_refr_at_eta75*(delH_desup+refr_lat_heat_cond)/(delT_air*Cp_air*rho_air) ;%# %in m^3/s
      W_refrfan = air_refrfan*delP_fan/eta_fan;
      %# Number of refrigeration fans
      N_refrfan = max(1, (floor(od_fans*air_refrfan / 100.0 + 0.9)));

      %# Fan purchase cost
      CP_refrfan = N_refrfan * 10 ^ (3.5391- 0.3533 * log10(max(od_fans*air_refrfan / N_refrfan, 0.01))+ 0.4477 * (log10(max(od_fans*air_refrfan / N_refrfan, 0.01))) ^ 2) ;

      CP_refrfan_min = N_refrfan * 10 ^ (3.5391- 0.3533 * log10(0.01)+ 0.4477 * (log10(0.01)) ^ 0.01) ;

      BM_refrfan = sd * FBM_fan * CP_refrfan* CI ;
      BM_0_refrfan = BM_refrfan / FBM_fan ;

      if (CP_refrfan == CP_refrfan_min) || (od_fans*air_refrfan / N_refrfan > 110) 
          BM_refrfan_min = CP_refrfan_min * FBM_fan* CI ;
          BM_refrfan = sd * BM_refrfan_min * (od_fans*air_refrfan / 0.01) ^ nf;
          BM_0_refrfan = BM_refrfan / FBM_fan;
          self.xm = self.xm  + 1;
      end    
    
%# refr evaporator HX (shell and tube)
      od_HX = 1.15 ;
      Q_evap_refr = refr_lat_evap*m_refr_at_eta75;
      U_evap = 850; %# OHTC in W/m^2K;  
      delTlm_evap = ((T_cwr-T_evap) - (T_cws-T_evap))/(log((T_evap- T_cwr)/(T_evap-T_cws)));
      area_evap = Q_evap_refr / (U_evap * delTlm_evap * FF);
      N_evap = max(1, (floor(od_HX*area_evap / 1000.0 + 0.9)));
      

      CP_evap_min = N_evap * 10 ^ (4.1884- 0.2503 * log10(2)+ 0.1974 * (log10(2)) ^ 2);

      CP_evap = N_evap * 10 ^ (4.1884- 0.2503 * log10(max(od_HX*area_evap / N_evap, 2))+ 0.1974 * (log10(max(od_HX*area_evap / N_evap, 2))) ^ 2);

      Fp_evap = 10 ^ (0.03881- 0.11272 * log10(12.0 + eps)+ 0.08183 * (log10(12.0 + eps)) ^ 2);

      FBM_evap = 1.63 + 1.66 * 1.0 * Fp_evap;

      BM_evap = sd * FBM_evap * CP_evap * CI;
      BM_0_evap = (1.63 + 1.66)*CP_evap*CI;

      if (CP_evap == CP_evap_min) || (od_HX*area_evap / N_evap > 1100)
          BM_evap_min = CP_evap_min * FBM_evap* CI;
          BM_evap = sd * BM_evap_min * (od_HX*area_evap / 2.0) ^ nhe;
          BM_0_evap = (1.63 + 1.66)*CP_evap_min*CI;
          self.xjjj = self.xjjj + 1;
      end  
      
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%lye cooler HX (shell and tube)


      U_lyecooler = 280 ; %#OHTC in W/m^2K; %%%%%%%%%%%Liquid to Liquid
      
      %# Q_lyecooler_at_eta75

      Q_catlyecool_eta75 = max((T_gl_out_at_eta75-273.15-80)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0) ;
      Q_anlyecool_eta75 = max((T_angl_out_at_eta75-273.15-80)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0) ;



      delTlm_catlyecooler_at_eta75 = ((T_gl_out_at_eta75 - T_cwr) - (353.15 - T_cws)) / log((T_gl_out_at_eta75 - T_cwr) / (353.15 - T_cws));
      delTlm_anlyecooler_at_eta75 = ((T_angl_out_at_eta75- T_cwr) - (353.15 - T_cws)) / log((T_angl_out_at_eta75 - T_cwr) / (353.15 - T_cws));


      area_catlyecooler_at_eta75 = Q_catlyecool_eta75 / (U_lyecooler * delTlm_catlyecooler_at_eta75 * FF);

      area_anlyecooler_at_eta75 = Q_anlyecool_eta75 / (U_lyecooler * delTlm_anlyecooler_at_eta75 * FF);

      area_lyecooler_at_eta75 = max(area_catlyecooler_at_eta75,area_anlyecooler_at_eta75 ) ;%# the higher of the 2 areas is taken for sizing of the lye cooler HX
      %# calc. of lye pump power and capexrho_KOH_g1_out_at_eta75

      L_tube = 6; %# per pass
      flow_area = pi*D_tube^2/4.0;  %# per tube flow area

      N_tube_lyecooler = ceil(od_HX*area_lyecooler_at_eta75/(pi*D_tube*L_tube));  %#/N_lye # Total No. of tubes per lyecool HX , total number of 2 passes
     % #N_tube_anlyecooler = np.ceil(area_anlyecooler_at_eta75/(pi*D_tube*L_tube)) 

      N_pass = 2 ;%# number of passes
      N_tube_pass_lyecooler = N_tube_lyecooler/N_pass; %# calc. number of parallel tubes per pass


      area_lyecooler_flow_pass = N_tube_pass_lyecooler* flow_area;

                        
      vel_tubes_lyecooler = max(Q_gl_out_at_eta75,Q_angl_out_at_eta75) /area_lyecooler_flow_pass; %# velocity of lye in HX tubes at rated cc
     

      vel_ratio = vel_tubes_lyecooler/V_target;
      N_tube_pass_lyecooler = N_tube_pass_lyecooler*max(vel_ratio,1);
      area_lyecooler_flow_pass = N_tube_pass_lyecooler* flow_area;
      area_lyecooler_at_eta75 = (pi*D_tube*L_tube)*N_tube_pass_lyecooler*N_pass;
      N_tube_lyecooler = N_tube_pass_lyecooler*N_pass;
      val = floor(area_lyecooler_at_eta75 / 1000 + 0.5);
      N_lye = max(1, (val));

      vel_tubes_lyecooler_eta75 = (max(Q_gl_out_at_eta75,Q_angl_out_at_eta75)) /area_lyecooler_flow_pass; %# recalculate the velocity in cathode lyecooler tubes after calc. of N_lye  , at rated operation.

      CP_lyecooler = 2*N_lye * 10 ^ (4.1884- 0.2503 * log10(max(2*area_lyecooler_at_eta75 /(2* N_lye), 2.0))+ 0.1974 * (log10(max(2*area_lyecooler_at_eta75 / (2*N_lye), 2.0))) ^ 2) ;

      CP_lyecooler_min = 2*N_lye * 10 ^ (4.1884- 0.2503 * log10(2.0)+ 0.1974 * (log10(2.0)) ^ 2) ;

      
      Fp_HX = (1.0 * (Pg == 0)+ (Pg > 0)* ((Pg < 5)+ (Pg >= 5)* 10 ^ (0.03881- 0.11272 * log10(Pg + eps)+ 0.08183 * (log10(Pg + eps)) ^ 2)));

      FBM_lyecool = 1.63 + 1.66 * 1.8 * Fp_HX;
      BM_lyecool = sd * FBM_lyecool * CP_lyecooler* CI;
      BM_0_lyecool = CP_lyecooler*(1.63+1.66)*CI;

      if (CP_lyecooler == CP_lyecooler_min) || (area_lyecooler_at_eta75 / (N_lye) > 1500)
          BM_lyecool_min = CP_lyecooler_min * FBM_lyecool* CI;
          BM_lyecool = sd * BM_lyecool_min * (area_lyecooler_at_eta75 / 2.0) ^ nhe;
          BM_0_lyecool = CP_lyecooler_min*(1.63+1.66)*CI;
          self.xiii =  self.xiii + 1;
      end


      %%%%%capex calc. of coolers in the glsep and purification section
       U_purif= 60; %# %%%%%%%%%%%%%%% LIQUID TO GAS is 60, common for these HX

      %# LMTDs
      delTlm_h2cool_at_eta75 = ((T_gl_out_at_eta75 - T_cwr) - (T_amb - T_cws)) / log((T_gl_out_at_eta75 - T_cwr) / (T_amb - T_cws));

      delTlm_deoxo_at_eta75 = ((T_deoxo_out - T_cwr) - (T_amb - T_cws)) / log((T_deoxo_out - T_cwr) / (T_amb - T_cws));

      delTlm_des_cool_at_eta75 = ((T_reg_at_eta75 - T_cwr) - (T_amb - T_cws)) / log((T_reg_at_eta75 - T_cwr) / (T_amb - T_cws));

      %# HT Areas, shell dias, cross-sectional areas etc
      area_h2cool_at_eta75      = od_HX*Q_cond_h2cool_at_eta75 / (U_purif * delTlm_h2cool_at_eta75 * FF);
      area_deoxo_cool_at_eta75  = od_HX*Q_cond_deoxo_at_eta75 / (U_purif * delTlm_deoxo_at_eta75 * FF);
      area_des_cool_at_eta75    =  od_HX*Q_cond_ads_at_eta75 / (U_purif * delTlm_des_cool_at_eta75 * FF);

      N_tube_h2cooler = ceil(area_h2cool_at_eta75/(pi*D_tube*L_tube));
      N_tube_deoxocool = ceil(area_deoxo_cool_at_eta75/(pi*D_tube*L_tube))  ;
      N_tube_descool = ceil(area_des_cool_at_eta75/(pi*D_tube*L_tube));

      Ds_h2cooler = ((4*N_tube_h2cooler*(3/4)^0.5*pt^2)/pi)^0.5 ; %# shell dia of h2 cooler estimate for delP calculations, based on a vornoi cell area belonging and surrounding each tube
      Ds_deoxocool = ((4*N_tube_deoxocool*(3/4)^0.5*pt^2)/pi)^0.5;
      Ds_descool = ((4*N_tube_descool*(3/4)^0.5*pt^2)/pi)^0.5;
      Ds_lyecool = ((4*N_tube_lyecooler*(3/4)^0.5*pt^2)/pi)^0.5;
      
      As_h2cooler =   Ds_h2cooler*(0.4*Ds_h2cooler)*(pt-D_tube)/pt;
      As_deoxocool =   Ds_deoxocool*(0.4*Ds_deoxocool)*(pt-D_tube)/pt;
      As_descool =   Ds_descool*(0.4*Ds_descool)*(pt-D_tube)/pt;
      As_lyecool =   Ds_lyecool*(0.4*Ds_lyecool)*(pt-D_tube)/pt;

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##### CW circ power calc.
      m_circ_lyecool_at_eta75 = (1/(2*N_lye))*Q_lyecooler_at_eta75/(Cp_cw*(T_cwr-T_cws)); %# per lye cooler unit
      m_circ_h2cooler_at_eta75 = Q_cond_h2cool_at_eta75/(Cp_cw*(T_cwr-T_cws))  ;
      m_circ_deoxocool_at_eta75 = Q_cond_deoxo_at_eta75/(Cp_cw*(T_cwr-T_cws));
      m_circ_descool_at_eta75 = Q_cond_ads_at_eta75/(Cp_cw*(T_cwr-T_cws));

      Re_lyecool_at_eta75 = (m_circ_lyecool_at_eta75/As_lyecool )*de/mu_cw ;
      Re_h2cooler_at_eta75 = (m_circ_h2cooler_at_eta75/As_h2cooler  )*de/mu_cw;
      Re_deoxocool_at_eta75 = (m_circ_deoxocool_at_eta75/As_deoxocool )*de/mu_cw;
      Re_descool_at_eta75 = (m_circ_descool_at_eta75/As_descool )*de/mu_cw;
      
      shell_delP_lyecool_at_eta75 = 2*(L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(exp(0.576 - 0.19 * log(Re_lyecool_at_eta75))*(m_circ_lyecool_at_eta75/As_lyecool )^2/(2*9.81*rho_cw*de))*N_lye ;
      shell_delP_h2cooler_at_eta75 = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(exp(0.576 - 0.19 * log(Re_h2cooler_at_eta75))*(m_circ_h2cooler_at_eta75/As_h2cooler  )^2/(2*9.81*rho_cw*de));
      shell_delP_deoxocool_at_eta75 = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(exp(0.576 - 0.19 * log(Re_deoxocool_at_eta75))*(m_circ_deoxocool_at_eta75 /As_deoxocool )^2/(2*9.81*rho_cw*de)) ;
      shell_delP_descool_at_eta75 = (L_tube/(0.4*Ds_descool))*Ds_descool*(exp(0.576 - 0.19 * log(Re_descool_at_eta75))*(m_circ_descool_at_eta75 /As_descool )^2/(2*9.81*rho_cw*de)) ;

      delP_cw_circ_at_eta75 = (shell_delP_lyecool_at_eta75+ shell_delP_h2cooler_at_eta75 + shell_delP_deoxocool_at_eta75 + shell_delP_descool_at_eta75)*1.2 ;
     
      %#Number of H2 coolers
      N_h2cool = max(1, (floor(area_h2cool_at_eta75 / 2000.0 + 0.5))) ;


      %# Cost correlations
      CP_h2cool = N_h2cool * 10 ^ (4.1884- 0.2503 * log10(max(area_h2cool_at_eta75 / N_h2cool, 2.0))+ 0.1974 * (log10(max(area_h2cool_at_eta75  / N_h2cool, 2.0))) ^ 2);

      CP_h2cool_min = N_h2cool * 10 ^ (4.1884- 0.2503 * log10(2.0)+ 0.1974 * (log10(2.0)) ^ 2);

      CP_deoxo_cool = 10 ^ (4.1884- 0.2503 * log10(max(area_deoxo_cool_at_eta75, 2.0))+ 0.1974 * (log10(max(area_deoxo_cool_at_eta75, 2.0))) ^ 2);

      CP_deoxo_min_cool = 10 ^ (4.1884- 0.2503 * log10(2.0)+ 0.1974 * (log10(2.0)) ^ 2);

      CP_des_cool = 10 ^ (4.1884- 0.2503 * log10(max(area_des_cool_at_eta75, 2.0))+ 0.1974 * (log10(max(area_des_cool_at_eta75, 2.0))) ^ 2);

      CP_des_cool_min = 10 ^ (4.1884- 0.2503 * log10(2.0)+ 0.1974 * (log10(2.0)) ^ 2)  ;

      FBM_purif = 1.63 + 1.66 * 1.8 * Fp_HX  ;

      BM_h2cool   = sd * FBM_purif * CP_h2cool * CI;
      BM_0_h2cool = sd *(1.63 + 1.66) * CP_h2cool * CI;
      BM_deoxocool    = sd * FBM_purif * CP_deoxo_cool * CI;
      BM_0_deoxocool = sd * (1.63 + 1.66) * CP_deoxo_cool * CI;
      BM_des_cool = sd * FBM_purif * CP_des_cool * CI;
      BM_0_des_cool = sd * (1.63 + 1.66)* CP_des_cool * CI  ;
      BM_HXpurif(k) = BM_deoxocool +BM_des_cool


      od_pump = 1.15; %# 15 % overdeisgn factor for pumps
      eta_pump =0.6;
      %%%###### lye cooler tube side delP and lye circ pump power calc.
      %Re_tube = 1260*  vel_tubes_lyecooler_eta75*D_tube/0.00093; %# Re number of tubes inside the HX
      %ff = (0.316*Re_tube^-0.25); %# friction factor for smooth tubes turbulent flow , Blasius equation
      Re_tube_at_eta75 = 1260*  vel_tubes_lyecooler_eta75*D_tube/0.00093; %# Re number of tubes inside the HX
      ff_at_eta75 = (0.316*Re_tube_at_eta75^-0.25);

      delP_catlyecooler_at_eta75 = (1+alpha_delPHX)*ff_at_eta75 *L_tube*1260*vel_tubes_lyecooler_eta75^2/(2*D_tube)*N_lye ;
      delP_catlyect_at_eta75 = (1+alpha_delPlyect)*delP_catlyecooler_at_eta75 ;      

      delP_anlyecooler_at_eta75 = (1+alpha_delPHX)*ff_at_eta75 *L_tube*1260*vel_tubes_lyecooler_eta75^2/(2*D_tube)*N_lye ; %# total pressure drop inside lye cooler.
      delP_anlyect_at_eta75  = (1+alpha_delPlyect)*delP_anlyecooler_at_eta75 ; %# total pressure drop in individual circuits  

      %W_catlyepump =  0.001*(delP_catlyect + cell_delP)*Q_gl_out /eta_pump; 

      W_anlyepump_eta75 = 0.001*(delP_anlyect_at_eta75 + ancell_delP_at_eta75)*Q_angl_out_at_eta75 /eta_pump ; %# in kW

      W_catlyepump_eta75 = 0.001*(delP_catlyect_at_eta75 + cell_delP_at_eta75)*Q_gl_out_at_eta75 /eta_pump ;% # in kW

      W_lyepump75 = max(W_anlyepump_eta75 , W_catlyepump_eta75) ; % # maximum value is selected for lye pump sizing, in kW

      W_cwpump75 = 0.001*delP_cw_circ_at_eta75*(m_cw_circ_at_eta75/rho_cw)/eta_pump ; %   # in kW  

      W_pump75 = (W_anlyepump_eta75 + W_catlyepump_eta75 + W_cwpump75);
      

      if W_lyepump75 > 0
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# Pump cost
        CP_pump_lye = 2*10 ^ (3.3892+ 0.0536 * log10(max(od_pumps*W_lyepump75, 0.05))+ 0.1538 * (log10(max(od_pumps*W_lyepump75, 0.05)))^2) ; % # factor of 2 because of 2 identical lye pumps

        CP_pump_lye_min = 2*10 ^ (3.3892+ 0.0536 * log10(0.05)+ 0.1538 * (log10(0.05))^2); %  # 1 kW

        FBM_pump_lye = 1.89 + 1.35 * (mean([2.3, 1.4])*((P < 11) * 1 + (P > 11) * 1.157));

        BM_lyepumps = sd * CP_pump_lye * FBM_pump_lye * CI;
        BM_0_lyepumps = CP_pump_lye*(1.89+1.35)* CI;
        

      if (CP_pump_lye == CP_pump_lye_min) || (od_pumps*W_lyepump75 > 500)
          BM_lyepumps_min = CP_pump_lye_min * FBM_pump_lye* CI;
          BM_lyepumps = sd * BM_lyepumps_min * (od_pumps*W_lyepump75 / 0.05) ^ 0.6;
          BM_0_lyepumps = CP_pump_lye_min*(1.89+1.35)* CI;
          %#xa = xa + 1;
      end    
      else
         BM_lyepumps =0; BM_0_lyepumps =0  ;
      end

      if W_cwpump75 > 0
      %# Pump cost
        CP_cwpump = 10 ^ (3.3892+ 0.0536 * log10(max(od_pumps*W_cwpump75, 0.05))+ 0.1538 * (log10(max(od_pumps*W_cwpump75, 0.05)))^2) ; % # factor of 2 because of 2 identical lye pumps

        CP_cwpump_min = 10 ^ (3.3892+ 0.0536 * log10(0.05)+ 0.1538 * (log10(0.05))^2) ;%  # 1 kW

        FBM_cwpump = 1.89 + 1.35 * (mean([2.3, 1.4]) * ((P < 11) * 1 + (P > 11) * 1.157));

        BM_cwpump = sd * CP_cwpump * FBM_cwpump * CI;
        BM_0_cwpumps = CP_cwpump*(1.89+1.35)* CI;
      

     % # apply 6/10 rule if min. cc > calc. cc
      if (CP_cwpump == CP_cwpump_min) || (od_pumps*W_cwpump75> 500)
          BM_cwpump_min = CP_cwpump_min * FBM_cwpump* CI;
          BM_cwpump = sd * BM_cwpump_min * (od_pumps*W_cwpump75 / 0.05) ^ 0.6;
          BM_0_cwpumps = CP_cwpump_min*(1.89+1.35)* CI;
       %   #xa += 1
      end 
      else
         BM_cwpump = 0 ; BM_0_cwpumps =0  ;
      end  
      BM_0_pumps = BM_0_cwpumps + BM_0_lyepumps;  

%# ===== INITIALIZE W_sys75 =====
      W_sys75 = 0.0;

            
%            # ===== COMPUTE W_sys75 HERE (inside valid condition) =====

      W_sys75 = W_pump75 + SEC_stack_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 + (heater_ads_at_eta75 + heater_deoxo_at_eta75 + heater_regen + Q_lyeheater_at_eta75)/eta_heater +W_h2comp +W_refrcomp_at_eta75 +W_refrfan_at_eta75 +W_compfan;
      allPower_eta75(k,:) = [ W_pump75  SEC_stack_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600  heater_ads_at_eta75/eta_heater heater_deoxo_at_eta75/eta_heater heater_regen/eta_heater Q_lyeheater_at_eta75/eta_heater W_h2comp W_refrcomp_at_eta75 W_refrfan_at_eta75 W_compfan];
      W_sys75 = W_sys75/eta_pow;   
      SEC_stack75(k) = SEC_stack_at_eta75;
      SEC_sys75(k) = 1000*((W_sys75))*J_kWh./(vap_h2_pdt_eta75*MH2); % 1000x for conv, from kW to W, SEC_sys at rated power without considering stack replacements, mainly used for comparing different capex designs
      Q_cond(k,:) = [Q_cond_deoxo_at_eta75 Q_cond_h2cool_at_eta75 Q_cond_ads_at_eta75 Q_lyecooler_at_eta75]  

      Pow_lyeheater =  heater_frac_ratedpower*W_sys75*1000; % # rated lye heater power is heater_frac_ratedpower% of the W_sys75 in Watt
      P_rated_el   = eta_heater * Pow_lyeheater; % # this is the power actually transferred to the heat of the lye
      BM_heaters = 7.5 * 3*W_sys75; %      # 7.5 $/kW, %7.5$/kW , 3 heaters in B.O.P,Grid connected Hydrogen production via large-scale water electrolysis{Nguyen, 2019 #186}, W_sys75 is in kW, od = 1.15
      BM_heaters_lye = Pow_lyeheater*0.001*150*(800/619) ; % # heaters have to be sized higher when the AWE is totally reliant on renewable power compared to when a battery is available, so as  to maximize the full load hours ( only when renew power is used), and increase load factor. $150/kW
       

      BM_pow = 199*W_sys75 ; %           # 199 $/kW, power electronics includes buck converter, transfomer, DC to DC converter, 1.15 over design factor


      BM_capex = Tot_stack_cost+BM_pow + BM_heaters_lye + BM_heaters + BM_h2cool+ BM_ads + BM_de+BM_glsep + BM_cwpump+BM_lyepumps + BM_evap +BM_refr + BM_shaft_h2comp +BM_shaft_refrcomp+ BM_h2comp + BM_refrcomp + BM_intercool + BM_fan +BM_refrfan +BM_lyecool +BM_deoxocool + BM_des_cool;
      BM_vesselpurif(k) = BM_de + BM_ads;
      BM_0_capex = Tot_stack_cost/Fp_stack +BM_pow + BM_heaters_lye + BM_heaters + BM_0_h2cool + BM_0_ads + BM_0_de + BM_0_glsep + BM_0_cwpumps + BM_0_lyepumps + BM_0_evap + BM_0_refr + BM_0_shaft_h2comp+BM_0_shaft_refrcomp + BM_0_h2comp +BM_0_refrcomp + BM_0_intercool + BM_0_fan + BM_0_refrfan + BM_0_lyecool +BM_0_deoxocool + BM_0_des_cool;
      capex_aux = BM_0_capex*0.5; %# Auxilliaries
      capex_cont =  0.18*BM_capex; %# contingnecy as a percentage of total fixed costs
      TIC = 1.18*BM_capex + capex_aux;

 %%% Loop for comparison bar chart
      BM_h2compression = BM_intercool + BM_shaft_h2comp + BM_h2comp + BM_fan;
      BM_refrig = BM_refr+BM_evap+BM_shaft_refrcomp+BM_refrcomp+BM_refrfan;
      BM_gasliqsep = BM_glsep+BM_h2cool;
      BM_h2purif = BM_des_cool + BM_deoxocool + BM_ads + BM_de + BM_heaters; %#contains deoxo heater and ads heater

      BM_lyecool=BM_lyecool;
      BM_heaters=BM_heaters_lye;
      BM_pumps = BM_cwpump+BM_lyepumps;
      BM_pow = BM_pow;
    
    %compVector = [BMG,BM_h2compression ,BM_refrig ,BM_gasliqsep ,BM_h2purif ,BM_lyecool,BM_pumps, Tot_stack_cost];%+mean(replacement_pv)];% this will go in the ANN training and heatmap; GA code will calc. the stack replacement, as per the degradation rate.
    %compVectorANN = [BMG*W_sys75,Tot_stack_cost*W_sys75];% abs capex required for GA optimization, so that the pareto front contains the true minimum LCOH.
    compVectorHeatmap  = [TIC,TIC/W_sys75,BM_h2compression/W_sys75 ,BM_refrig/W_sys75 ,BM_gasliqsep/W_sys75 ,BM_h2purif/W_sys75 ,BM_lyecool/W_sys75,BM_pumps/W_sys75, Tot_stack_cost/W_sys75];
    allCapexvector = [BM_h2compression/W_sys75 ,BM_refrig/W_sys75 ,BM_gasliqsep/W_sys75 ,BM_h2purif/W_sys75 ,BM_lyecool/W_sys75,BM_heaters_lye/W_sys75, BM_pow/W_sys75,BM_pumps/W_sys75, Tot_stack_cost/W_sys75];%+mean(replacement_pv)];% this will go in the local bar chart and the pie plot for breakup visulaization of typical cases
   
    allCapexComponents = [allCapexComponents; allCapexvector];
    paramVals = [paramVals; params(heggg)]; % If 'vin' is 3rd param, else adjust index
    absCapexVals = [absCapexVals; TIC];
    Wsys70Vals = [Wsys70Vals; W_sys75];
    CapexperkW = absCapexVals./Wsys70Vals;
    %IC_comp = [capex_cont, capex_aux, BM_capex];
    cd_eta75Vals = [cd_eta75Vals;cd_eta75];
   allOutputHeatmap = [allOutputHeatmap; compVectorHeatmap];
   allInputscapex = [allInputscapex; params];
  % % allOutputscapex = [allOutputscapex; compVectorANN];
  % allOutputsabscapex = [   allOutputsabscapex; mean(BMG)*W_sys75];
%total_capex_ratedlist(k,:) =mean(total_capex_rated)
end
% figure;
% hb = bar(allCapexComponents, 'grouped');
% hold on;
% 
% xlabel('Parameter Value', 'FontSize', 12);
% ylabel('Component Capex ($/kW)', 'FontSize', 12);
% title('Capex Component Breakdown with % Bar Labels', 'FontSize', 14);
% set(gca, 'XTick', 1:length(paramVals), 'XTickLabel', string(paramVals));
% 
% % legend({
% %     'Purification HX',
% %     'Compressor Intercooler',
% %     'Refrigerant Condenser',
% %     'Evaporator',
% %     'Lye Cooler',
% %     'Shaft (Refr Comp)',
% %     'Shaft (H2 Comp)',
% %     'H2 Compressor',
% %     'Refrigerant Compressor',
% %     'Pumps',
% %     'Heater',
% %     'Stack (Tot)',
% %     'Gas-Liq. Separator',
% %     'Adsorber',
% %     'Deoxo',
% %     'Fan'
% % }, 'Location', 'bestoutside');
% 
% legend({
%     'BM_h2compression',
%     'BM_refrigeration',
%     'BM_gasliqsep',
%     'BM_h2purif',
%     'BM_lyecool',
%     'BM_heater',
%     'BM_pow' ,
%     'BM_pumps',
%     'Tot_stack_cost'
% 
% 
% }, 'Location', 'bestoutside');
% 
% grid on;
% 
% %---- Absolute capex label above each group -----%
% for k = 1:length(paramVals)
%     yMax = max(allCapexComponents(k,:));
%     text(k, yMax * 1.14, sprintf('%.2f k$', absCapexVals(k)/1000), ...
%         'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize',11, 'Color','k');
% end
% 
% %---- Percent labels above each bar -----%
% [numCases, numComps] = size(allCapexComponents);
% totalCapexPerCase = sum(allCapexComponents, 2);
% 
% for nn = 1:numComps
%     for mm = 1:numCases
%         % Bar position for this case/component:
%         % Find X centers for grouped bars
%         if verLessThan('matlab', '9.7') % Before R2019b
%             xCenter = hb(nn).XData(mm) + hb(nn).XOffset;
%         else
%             % R2019b or newer, use XEndPoints property
%             xCenter = hb(nn).XEndPoints(mm);
%         end
%         yTop = allCapexComponents(mm, nn);
%         percent = 100 * yTop / totalCapexPerCase(mm);
%         if yTop > 0.02 * max(totalCapexPerCase) % Only label meaningful bars
%           text(xCenter, yTop * 1.04, sprintf('%.1f%%', percent), ...
%              'HorizontalAlignment','center', 'FontSize',9, 'Color', 'k');
%         end
%     end
% end
% 
% hold off;
% 
% %---- Annotate W_sys75 values below legend -----%
% annotationText = sprintf('Rated System Power W_{sys70} [kW] per case: %s', ...
%     strjoin(cellstr(num2str(Wsys70Vals(:), '%.1f')), ', '));
% annotation('textbox', [0.15, 0.72, 0.7, 0.05], 'String', annotationText, ...
%     'FitBoxToText', 'on', 'EdgeColor', 'none', 'FontSize', 11, ...
%     'HorizontalAlignment', 'center');

close all;

% Find min and max for scaling
minIncapex = min(allInputscapex, [], 1);
maxIncapex = max(allInputscapex, [], 1);

minOutcapex = min(allOutputscapex, [], 1);
maxOutcapex = max(allOutputscapex, [], 1);
minOutcapexHeat = min(allOutputHeatmap, [], 1);
maxOutcapexHeat = max(allOutputHeatmap, [], 1);

minOutabscapex = min(allOutputsabscapex, [], 1);
maxOutabscapex = max(allOutputsabscapex, [], 1);

% Scale inputs between -1 and 1
allInputscapexScaled = 2 * (allInputscapex - minIncapex) ./ (maxIncapex - minIncapex) - 1; % (or use 0 to 1 scaling if preferred)

% Scale outputs between -1 and 1
allOutputscapexScaled = 2 * (allOutputscapex - minOutcapex) ./ (maxOutcapex - minOutcapex) - 1;
allOutputcapexHeatmapScaled = 2 * (allOutputHeatmap - minOutcapexHeat) ./ (maxOutcapexHeat - minOutcapexHeat) - 1;

% Save to .mat file for ANN training
%save('preparedData_capex.mat', 'allInputscapex',
%'allCapexComponents')
% save('preparedData_capex.mat', 'allInputscapexScaled', 'allOutputscapexScaled', ...
 %    'minIncapex', 'maxIncapex', 'minOutcapex', 'maxOutcapex');



%save('preparedData_capex_rated75_heatmap.mat', 'allInputscapexScaled', 'allOutputcapexHeatmapScaled', ...
%     'minIncapex', 'maxIncapex', 'minOutcapexHeat', 'maxOutcapexHeat');
save('preparedData_pie.mat', 'allCapexComponents');% 'allOutputscapexScaled', ...
  %   'minIncapex', 'maxIncapex', 'minOutcapex', 'maxOutcapex');

  
%save('preparedData_TIC_pie.mat', 'IC_comp');
fprintf('Processed %d files.\n', length(matFiles));
fprintf('Total usable samples: %d\n', size(allInputscapex,1));
toc;


