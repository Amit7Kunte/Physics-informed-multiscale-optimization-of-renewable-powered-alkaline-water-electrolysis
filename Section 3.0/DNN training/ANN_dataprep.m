

clear; clc; %close all;
tic;
format long g

eta_isen_comp_old = 0.6;
eta_isen_comp_new = 0.85;
eta_isen_comp_ratio = 0.6/0.85; % update eta_isen_comp from 0.6 to 0.85
matFiles = dir(fullfile('iModel_Wgde*_datan.mat'));
fileNames = {matFiles.name};
excludeIdx = contains(fileNames, 'P_20_') ;% | contains(fileNames, 'P_12_');%|~contains(fileNames, 'dpored_0_');

matFiles = matFiles(~excludeIdx);


% Output variable names you want from ANN model validation
% outputVars = {'SEC_sys', 'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O', ...
%     'eta_curr_sys', 'eta_curr_stack', 'eta_shunt', 'eta_therm', 'eta_volt', ...
%     'SEC_ads', 'SEC_chiller_h2cooler', 'SEC_chiller_lyecooler', 'SEC_comp', ...
%     'SEC_deoxo', 'W_lye_pumps','W_dmw_pump', 'SEC_lyeheaters', 'T_reg', 'H2_mixedToHTO', 'Q_cond_ads', 'Q_cond_deoxo','W_refrcomp','T_gl_out','T_angl_out'};

%outputVars = {'SEC_sys', 'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O', ...
%    'w_KOH_angl_out','w_KOH_gl_out','Q_gl_out','Q_angl_out', 'T_reg', 'H2_mixedToHTO', 'Q_cond_h2cooler' ,'Q_cond_ads', 'Q_cond_deoxo','W_refrcomp','T_gl_out','T_angl_out'};

%outputVars = {'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O', ...
%    'w_KOH_angl_out','w_KOH_gl_out','Q_gl_out','Q_angl_out','glsep_O2' , 'H2_mixedToHTO', 'Q_cond_h2cooler' ,'Q_cond_ads', 'Q_cond_deoxo','T_gl_out','T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'};

outputVars = {'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O', ...
    'w_KOH_angl_out','w_KOH_gl_out','Q_gl_out','Q_angl_out','glsep_O2' , 'Q_cond_h2cooler' ,'Q_cond_ads', 'Q_cond_deoxo','T_gl_out','T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'};


delP_fan =10 ;% 10 Pa
eta_mech_motor = 0.95;

eta_mech_comp= 0.9;
rho_H2O_20 = 1000; %kg/m^2
eta_isen_refrcomp = 0.7;
refr_lat_heat_evap = 199.3*1000;%j/kg, this is not used in the calc. of the m_refr, as the expansion valve outlet is a two phase mixture at the t and p of the evaporator ( both lower than cond)
Pevap = 1323.7; % kPa sat pressure at temp. of evaporation
refr_S_evap = 1.7818*1000;
refr_H_comp_s = 443.6*1000; % 50C at the sampe entropy of the evap, but pressure of condenser,ideal compression
refr_H_cond_vap = 427*1000; % condenser vap enthalpy
refr_H_cond_liq = 267.1*1000 ;
refr_H_evap_vap = 426.4*1000;
refr_H_evap_liq = 227.1*1000;
Pcond = 2411; %condenser pressure in kPa
rho_dm =1000 ; % 1000 kg/m^3
refr_lat_evap = refr_H_evap_vap-refr_H_cond_liq; % This is the actual latent heat from 2 phase mixture to vapor phase in the evap.
refr_H_comp = refr_H_evap_vap+(refr_H_comp_s-refr_H_evap_vap)/eta_isen_refrcomp; % enthalpy at comp discharge for non-isen condition
cond_duty = refr_H_comp - refr_H_cond_liq ;
delH_comp = refr_H_comp-refr_H_evap_vap;
%refr_H_cond = 427.4*1000;
refr_lat_heat_cond= 159.9*1000;               %   considering condensation at 40C , 167.7*1000 ( was for 36C);
delH_desup = refr_H_comp-refr_H_cond_vap;
% Initialize arrays to collect input/output data from all files
allInputs = [];
allOutputs = [];
skippedFiles = {};

for k = 1:length(matFiles)
    fileName = matFiles(k).name;
    tokens = split(fileName, '_');
    paramNames = {'Wgde', 'P', 'vin', 'dpore', 'dpored', 'Wsep'};
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
   
    % Load .mat file data
    data = load(fileName);

    if ~isfield(data, 'currentDensity')
        error('currentDensity not found in file %s', fileName);
    end
    currDensOrig = data.currentDensity(:);
    if max(currDensOrig) < 10000
        fprintf('Skipping %s: max(currentDensity) < 10000 A/m^2\n', fileName);
        skippedFiles{end+1} = fileName;
        continue
    end

    numPoints = length(currDensOrig); % each .mat file may be a different length

    % Gather all outputs (one row for each point) using the actual data, not interpolated
    thisOutputs = nan(numPoints, length(outputVars));
    for v = 1:length(outputVars)
        varName = outputVars{v};
        vap_h2_pdt = data.vap_h2_pdt;
        P=params(2)*10^5 ;
        Pg = P-10^5; % guage pressure
        H= 0.05;
        D = 0.008;
        MH2 = 0.002; %kg/mol
        J_kWh = 2.7778E-7;
        R_const = 8.3145;% J/molK
        delT_air = 10;%10K
        Cp_air = 1005;
        rho_air = 1.18;% kg/m^3;
        Cp_watvap_80 = 34.6; % J/molK
        Cp_H2_80 = 29;
        Cp_O2_80 = 29.72;
        
        %%% H2 compressor power and SEC calc.
        eta_mech_motor = 0.95;
        eta_mech_comp= 0.9;
        % P = mphglobal(model,'aveop_catout(dl.pA)'); %cell outlet Pres.
        % P = mean(P);
        eta_isen_comp = 0.85;
        Tamb = 298.15; % Ambient temp.
        P_final = 30*10^5; % final h2 delivery pressure
        T_max_comp = 273.15 + 135;
        k_isen = 1.41;
        Z_comp = 1.05;
        delP_fan = 10 ;% Pa ; % needs to be revisited 
        eta_fan =0.55;
        T_suc_comp = Tamb + 11; % suction temp. 
        rc_max1 = ((T_max_comp/Tamb-1)*eta_isen_comp+1)^(k_isen/(k_isen-1));
        rc_max = ((T_max_comp/T_suc_comp-1)*eta_isen_comp+1)^(k_isen/(k_isen-1));
        N_comp = ceil(log(P_final/(P*rc_max1))/log(rc_max))+1;
        N_comp = max(N_comp,1);
        if N_comp == 1
            rc = rc_max1;
        else
            rc = (P_final/(P*rc_max1))^(1/(N_comp-1));
        end
        
        T_dis_comp = T_suc_comp*(1+(rc^((k_isen-1)/k_isen)-1)/eta_isen_comp);
        W_comp = (max(N_comp-1,0)*(k_isen)/((k_isen)-1))*R_const*T_suc_comp*Z_comp*(rc^((k_isen-1)/k_isen)-1)*vap_h2_pdt/(eta_isen_comp*eta_mech_motor*eta_mech_comp)+(P_final/P>1)*(1*(k_isen)/((k_isen)-1)*R_const*Tamb*Z_comp*(rc_max1^((k_isen-1)/k_isen)-1)*vap_h2_pdt/(eta_isen_comp*eta_mech_motor*eta_mech_comp));
        intercool_comp = max((N_comp),0)*(T_dis_comp-T_suc_comp)*vap_h2_pdt*Cp_H2_80+(1)*(T_max_comp-Tamb)*vap_h2_pdt*Cp_H2_80;
        air_fan = intercool_comp./(delT_air*Cp_air*rho_air);
        W_fan = air_fan*delP_fan/eta_fan;
        W_shaft_comp = (max(N_comp-1,0)*(k_isen)/((k_isen)-1))*R_const*T_suc_comp*Z_comp*(rc^((k_isen-1)/k_isen)-1)*vap_h2_pdt/(eta_isen_comp*eta_mech_comp)+(P_final/P>1)*(1*(k_isen)/((k_isen)-1)*R_const*Tamb*Z_comp*(rc_max1^((k_isen-1)/k_isen)-1)*vap_h2_pdt/(eta_isen_comp*eta_mech_comp));%%%%% mistake in eta_mech_motor applied here, corrected now
        

        %%% SEC lye heaters and lye cooler SEC and power calculations
        eta_heater = 0.95;
        Tin =353.15;
        heater_lye = (data.T_gl_out<Tin).*(data.Q_gl_out.*(4.101*10^3-3.526*10^3.*data.w_KOH_gl_out+9.644*10^-1.*(data.T_gl_out-273)+1.776.*(data.T_gl_out-273).*data.w_KOH_gl_out).*data.rho_KOH_gl_out.*(Tin-data.T_gl_out))./eta_heater;
        heater_anlye = (data.T_angl_out<Tin).*(data.Q_angl_out.*(4.101*10^3-3.526*10^3.*data.w_KOH_angl_out+9.644*10^-1.*(data.T_angl_out-273)+1.776.*(data.T_angl_out-273).*data.w_KOH_angl_out).*data.rho_KOH_angl_out.*(Tin-data.T_angl_out))./eta_heater;
        data.SEC_lyeheaters = (heater_lye+heater_anlye)*J_kWh./(vap_h2_pdt*MH2);
        Cp_KOH_lye = (4.101*10^3-3.526*10^3.*data.w_KOH_gl_out+9.644*10^-1.*(data.T_gl_out-273)+1.776.*(data.T_gl_out-273).*data.w_KOH_gl_out);
        Cp_KOH_anlye = 4.101*10^3-3.526*10^3.*data.w_KOH_angl_out+9.644*10^-1.*(data.T_angl_out-273)+1.776.*(data.T_angl_out-273).*data.w_KOH_angl_out;
        data.Q_lyecooler = (data.T_gl_out>Tin).*(data.Q_gl_out.*Cp_KOH_lye.*data.rho_KOH_gl_out.*(data.T_gl_out-Tin)) + (data.T_angl_out>Tin).*(data.Q_angl_out.*(4.101*10^3-3.526*10^3.*data.w_KOH_angl_out+9.644*10^-1.*(data.T_angl_out-273)+1.776.*(data.T_angl_out-273).*data.w_KOH_angl_out).*data.rho_KOH_angl_out.*(data.T_angl_out-Tin));
        Cp_cw = 4184; % J/kgK
        T_cws = 293.15;
        T_cwr = T_cws + 5;
        m_cw_lyecooler = data.Q_lyecooler./(Cp_cw.*(T_cwr-T_cws));
        %data.SEC_lyecoolers = (Q_lyecooler)*J_kWh./(vap_h2_pdt*MH2);
        m_ref = (data.Q_lyecooler./refr_lat_evap);
        refrig_lyecooler = (delH_comp.*m_ref./(eta_mech_comp*eta_mech_motor)+((m_ref.*((cond_duty)))./(delT_air*Cp_air*rho_air)).*delP_fan./eta_fan);
        data.SEC_chiller_lyecooler = (refrig_lyecooler)*J_kWh./(vap_h2_pdt*MH2);
        
        
        %%% SEC h2coolers and SEC ads coolers and deoxo coolers


        glsep_H2 = (data.degas_eff_catH2*(data.H2_catout-data.H2ref_gl_out)); % for 1000x value in the .mph file
        glsep_O2 = (data.degas_eff_catO2*(data.O2_catout-data.O2ref_gl_out));
        glsep_H2O = (glsep_H2+glsep_O2)./(1-data.xH2O_gl).*(data.xH2O_gl);
        anglsep_H2 = (data.degas_eff_anH2*(data.H2_anout-data.H2ref_angl_out));
        anglsep_O2 = (data.degas_eff_anO2*(data.O2_anout-data.O2ref_angl_out));
        anglsep_H2O = (anglsep_H2+anglsep_O2)./(1-data.xH2O_angl).*(data.xH2O_angl);
        vap_glsep = glsep_H2 + glsep_O2 + glsep_H2O + data.gasrate_cat; % vapor rate from glsep at 1000x size of stack
        vap_anglsep = anglsep_H2 + anglsep_O2 + anglsep_H2O + data.gasrate_an;
        xH2O_glsep = (glsep_H2O+data.moistrate_cat)./(vap_glsep);
        xH2O_anglsep = (anglsep_H2O+data.moistrate_an)./(vap_anglsep);
        xH2_glsep = (glsep_H2+(data.gasrate_cat-data.moistrate_cat))./(vap_glsep);
        xO2_glsep = 1-xH2_glsep-xH2O_glsep;
        xO2_glsep1= glsep_O2./(vap_glsep);
        xO2_anglsep = (anglsep_O2+(data.gasrate_an-data.moistrate_an))./(vap_anglsep);
        xH2_anglsep = 1-xO2_anglsep-xH2O_anglsep;
        Cp_watvap_80 = 34.6; % J/molK
        Cp_H2_80 = 29;
        Cp_O2_80 = 29.72;
        T_glsepHXout = 298.15 ; % Temp. out of gas liq sep cooler
        lat_heat_water = 44000; % j/mol
        Cp_vapglsep= Cp_watvap_80*xH2O_glsep+Cp_O2_80*xO2_glsep+ Cp_H2_80*xH2_glsep;
        condensate_h2cooler = (xH2O_glsep.*vap_glsep)-(vap_glsep.*(1-xH2O_glsep).*data.x_H2O_h2cool_out)./(1-data.x_H2O_h2cool_out);
        cond_h2cooler=(vap_glsep.*Cp_vapglsep.*(data.T_gl_out- T_glsepHXout)+condensate_h2cooler.*lat_heat_water);
        data.Q_cond_h2cooler = cond_h2cooler;
        m_cw_h2cooler = cond_h2cooler./(Cp_cw*(T_cwr-T_cws));
        vap_h2cool = vap_glsep-condensate_h2cooler;
        % de oxo
        deoxo_H2O = 2*glsep_O2;
        deoxo_H2reac = 2*glsep_O2;
        deoxo_heat = 2*glsep_O2*244.9*10^3; %244.9 Kj/mol; converted to W
        Cp_vap_h2cool = Cp_watvap_80.*data.x_H2O_h2cool_out+Cp_O2_80.*(glsep_O2./vap_h2cool)+ Cp_H2_80.*(xH2_glsep.*vap_glsep./vap_h2cool);
        tcycle_deoxo = 8*3600; %cycle time in sec
        GHSV = 2083;% per hr, ' A comparsion of g-Al2O3-supported deoxo catalysts for the selective removal of oxygen ......
        od_deoxo = 1.2;
        vol_deoxo = od_deoxo*max(0.001*vap_h2cool*22.414*3600/GHSV); %vol of deoxo catalyst in m^3
        mass_deoxo = vol_deoxo*650; % bulk density of alumina catalyst 650 kg/m^3
        Cp_deoxo = 900; %J/kgK
        D_deoxo = 0.015; %1m
        L_deoxo = vol_deoxo/(pi*D_deoxo^2/4);
        rho_steel = 7850; Cp_steel = 475; % kg/m^3, and J/kgK
        mass_st_deoxo = (pi*(D_deoxo+0.002)^2/4*L_deoxo-pi*(D_deoxo)^2/4*L_deoxo)*rho_steel; %2mm thickness
        %Thermal capture rate of steel and deoxo cat in the deoxo reactor
        %in W/K
        therm_st_deoxo = mass_st_deoxo*Cp_steel/tcycle_deoxo * ones(length(deoxo_heat),1) ;
        therm_cat_deoxo = mass_deoxo*Cp_deoxo/tcycle_deoxo* ones(length(deoxo_heat),1) ;
        T_deoxo = (deoxo_heat+ therm_cat_deoxo.*(T_glsepHXout+125) + therm_st_deoxo.*(T_glsepHXout+125)   +  vap_h2cool.*Cp_vap_h2cool.*(T_glsepHXout)  )./(vap_h2cool.*Cp_vap_h2cool +therm_st_deoxo + therm_cat_deoxo ); % this seems to high
        vap_deoxo = vap_h2cool+deoxo_H2O-deoxo_H2reac-glsep_O2; % mol/s, vap flow out of deoxo
        xH2O_deoxo = (deoxo_H2O+vap_h2cool.*data.x_H2O_h2cool_out)./vap_deoxo; % moisture molefrac at exit of de-oxo
        xH2_deoxo = (-deoxo_H2reac+xH2_glsep.*vap_glsep)./vap_deoxo;
        Cp_vap_deoxo = xH2_deoxo.*Cp_H2_80+xH2O_deoxo.*Cp_watvap_80;
        condensate_H2O_deoxo_cooler = (xH2O_deoxo.*vap_deoxo)-(vap_deoxo.*(1-xH2O_deoxo).*data.x_H2O_h2cool_out)./(1-data.x_H2O_h2cool_out); % water condensed in the de-oxo cooler
        deoxo_T = 150 +273.15; % deoxo temperature control
        m_cw_deoxo_cooler = (vap_deoxo.*Cp_vap_deoxo.*(deoxo_T-T_glsepHXout)+condensate_H2O_deoxo_cooler.*lat_heat_water)./(Cp_cw.*(T_cwr-T_cws)); % numerator in W
        data.Q_cond_deoxo = (vap_deoxo.*Cp_vap_deoxo.*(deoxo_T-T_glsepHXout)+condensate_H2O_deoxo_cooler.*lat_heat_water);
        eta_heater = 0.95;
        heater_deoxo = (deoxo_T>T_deoxo).*vap_h2cool.*Cp_vap_h2cool.*(deoxo_T-T_deoxo)./eta_heater;
        heater_regen = (therm_st_deoxo + therm_cat_deoxo)*(723.15-deoxo_T)./eta_heater; % regen temp at 450C, 
        data.heater_deoxo = heater_regen +heater_deoxo;
        adspec_H2O = 5E-06; % target in ppm
        R_const = 8.3145;% J/molK
        m_ads = 0.03; %30 gm
        %m_ads =25 ;% in kg
        t_b = 8*3600; %8 hrs cycle time
        t_des = 5/8*t_b;
        t_cool = 3/8*t_b;
        rho_ads =  650;% kg/m^3
        vol_ads = 1.2*m_ads/rho_ads; % vol. of the ads in m^3, % overdesign =20%
        dia_ads = 0.03 ; % 20cm 
        L_ads = vol_ads/(pi*dia_ads^2/4);
        por_ads = 0.3;
        phi = 1; %sphericity of adsorbent
        ads_dia = 0.0015875; % 1/16 inch
        B = 0.152;
        C = 0.000136;
        mu_H2_25 = 1.9E-5; % kg/ms
        rho_H2_25 = 0.08; % kg/m3
        eta_ads = .6; % adsrobent utilization factor for non-ideality
        q_max = 19.43; % mol/kg max. adsorption capacity of adsorbent
        purge_ads = 0.5*10^-4; % mol/s , purge gas flow through regen circuit
        Cp_ads = 900.00 ;%J/(kg·K);

        m_steel_ads = ((pi*(dia_ads/2+.002)^2-pi*(dia_ads/2)^2)*L_ads)*rho_steel*1.5; % 2mm thick ads shell is assumed, 50% for miscellaneous fittings

        H_ads = 50000; %[J/mol]; heta of adsorption
        b_ref = 93.5 ; %1/Pa; 
        b_T = b_ref.*exp(H_ads/R_const*(1/T_glsepHXout-1/293.15)); % b value at 25C
        p_H2O_adspec = mean(adspec_H2O*P); % par_pres of moisture in ads outlet h2 stream at 25C ads temperature
        alpha_purge = 0.5;
        
        ads_H2O_in = (vap_deoxo.*(1-xH2O_deoxo).*data.x_H2O_h2cool_out)./(1-data.x_H2O_h2cool_out); % water to adsorber;
        ads_H2O_out = (vap_deoxo.*(1-xH2O_deoxo).*adspec_H2O)./(1-adspec_H2O);
        ads_cap_rate = ads_H2O_in-ads_H2O_out; % rate of moisture capture ;
        vap_ads = (vap_deoxo.*(1-xH2O_deoxo))./(1-data.x_H2O_h2cool_out); %flow of h2 sream thru adsorber
        vel_ads = (vap_ads.*R_const.*T_glsepHXout./P)/(pi.*dia_ads^2/4);
        delP_ads = ((B.*mu_H2_25.*1000.*(vel_ads.*3.28*60))+C*(vel_ads*3.28*60).^2*(rho_H2_25*0.062428)*6894.75729);% in Pa/m
        %Using ergun eqn
        term1 = (150*mu_H2_25.*(1 - por_ads)^2)./(por_ads^3*ads_dia^2)*L_ads.*vel_ads./phi;
        term2 = (1.75*rho_H2_25.*(1 - por_ads))./(por_ads^3*ads_dia)*L_ads.*vel_ads.^2./phi^2;
        dP_ads = term1 + term2;   % Pressure drop in Pa
        
        qcc = ads_cap_rate*t_b./(m_ads*eta_ads); % required ads cc
        ads_cc = m_ads.*qcc.*eta_ads; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required
        ads_rate = ads_cc./t_b;
        
        q_res = q_max-qcc;
        rq = q_res./q_max;
        des_rate = ads_cc./t_des; % the reqd des_rate increases with current density as more mositure is in gas stream.
        rp = purge_ads./(des_rate); % ratio of purge flow rate to des rate (based on desorb time)
        xH2O_reg_out = (data.x_H2O_h2cool_out+1./rp)./(1+1./rp); % mositure moefrac at regen exit
        p_H2O_reg_out = P.*(data.x_H2O_h2cool_out+1./rp)./(1+1./rp);
        p_vap_reg_eff = P.*(data.x_H2O_h2cool_out+alpha_purge.*(xH2O_reg_out-data.x_H2O_h2cool_out));
        b_reg = ((rq.^0.2472)./(1-rq.^0.2472)).^(1/0.2472).*(1./p_vap_reg_eff);
        T_reg = max(T_glsepHXout+75,(1/(293.15)+R_const./H_ads.*log(b_reg/b_ref)).^-1);
        heater_ads = (purge_ads)*Cp_H2_80*(T_reg-T_glsepHXout(ones(length(T_reg),1)))+des_rate*H_ads+(m_steel_ads*Cp_steel+m_ads*Cp_ads).*(((T_reg-T_glsepHXout(ones(length(T_reg),1)))./t_des)/eta_heater)*t_des/t_b;% heater duty in watts, avegared over a complete cycle.
        data.heater_ads = heater_ads;
        %(purge_ads)*Cp_H2_80*(T_reg-T_glsepHXout(ones(length(T_reg),1)))+des_rate*H_ads+(m_steel_ads*Cp_steel+m_ads*Cp_ads)
        tau_cool = por_ads*vol_ads./((purge_ads*R_const*T_glsepHXout/P)); %resid time in cooling bed for purge
        condensate_H2O_regen_cond = (purge_ads.*(1-data.x_H2O_h2cool_out).*xH2O_reg_out)./(1-xH2O_reg_out)-(data.x_H2O_h2cool_out.*purge_ads);
        cond_ads = (condensate_H2O_regen_cond.*lat_heat_water+(purge_ads).*Cp_H2_80.*(T_reg-T_glsepHXout)).*t_des./t_b;
        m_cw_cond_ads = (cond_ads)./(Cp_cw.*(T_cwr-T_cws));
        data.Q_cond_ads = cond_ads;
        vap_h2_pdt = vap_ads.*(1-data.x_H2O_h2cool_out)./(1-adspec_H2O);
        m_ref = (m_cw_deoxo_cooler*Cp_cw*(T_cwr-T_cws)./refr_lat_evap);
        refrig_deoxo= (delH_comp.*m_ref./(eta_mech_comp*eta_mech_motor)+((m_ref.*((cond_duty)))./(delT_air*Cp_air*rho_air)).*delP_fan./eta_fan);
        data.SEC_deoxo = (refrig_deoxo +heater_deoxo +heater_regen)*J_kWh./(vap_h2_pdt*MH2);
        m_ref = (m_cw_cond_ads*Cp_cw*(T_cwr-T_cws)./refr_lat_evap); 
        refrig_ads= (delH_comp.*m_ref./(eta_mech_comp*eta_mech_motor)+((m_ref.*((cond_duty)))./(delT_air*Cp_air*rho_air)).*delP_fan./eta_fan);
        data.SEC_ads = (refrig_deoxo +heater_ads)*J_kWh./(vap_h2_pdt*MH2);
        m_ref = (m_cw_h2cooler*Cp_cw*(T_cwr-T_cws)./refr_lat_evap); 
        refrig_h2cooler= (delH_comp.*m_ref./(eta_mech_comp*eta_mech_motor)+((m_ref.*((cond_duty)))./(delT_air*Cp_air*rho_air)).*delP_fan./eta_fan);
        data.SEC_h2cooler = (refrig_h2cooler)*J_kWh./(vap_h2_pdt*MH2);
        
        
        %%% SEC pump and power calculations (all pumps)
        cell_delP = data.cell_inP-P; % pressure drop across cell
        ancell_delP = data.ancell_inP-P;
        data.cell_delP = cell_delP;
        data.ancell_delP = ancell_delP;

        eta_pump =0.6;
        delP_to_stack_inlet = 100; % 100 Pa of all HX and fittings to the small 5 cell stack
        W_pump_hylye = data.Q_gl_out.*(cell_delP+delP_to_stack_inlet)./eta_pump; 
        W_pump_oxlye = data.Q_angl_out.*(ancell_delP+delP_to_stack_inlet)./eta_pump; 
        W_pump_dmw =(data.dmw_netfeed./rho_dm.*(Pg+ 30))./eta_pump;% 30 Pa line delP
        cws_circ = m_cw_deoxo_cooler+m_cw_h2cooler+m_cw_lyecooler +m_cw_cond_ads;
        W_pump_cws = cws_circ./rho_dm.*(50)./eta_pump;
        data.SEC_pumps = ( W_pump_dmw + W_pump_oxlye +W_pump_hylye +  W_pump_cws)*J_kWh./(vap_h2_pdt*MH2);
     

        %%% SEC refrigeration compressor and refrigeration fan 
       % data.W_refrcomp+data.W_refrfan
        m_refr = cws_circ.*Cp_cw.*(T_cwr-T_cws)./refr_lat_evap; % total refireigenrant circulation rate inside the chiller
        data.m_refr = m_refr;
        W_refrcomp = (delH_comp.*m_refr./(eta_mech_comp*eta_mech_motor));
        W_refrfan = ((m_refr.*((cond_duty)))./(delT_air*Cp_air*rho_air)).*delP_fan./eta_fan;
        data.W_refrcomp = W_refrcomp;
        data.W_refrfan = W_refrfan;
        data.SEC_refrcomp = ( data.W_refrcomp+data.W_refrfan)*J_kWh./(vap_h2_pdt*MH2); %contains energy consumption of both the fan and the compressor


        %data.SEC_lyecoolers = (Q_lyecooler)*J_kWh./(vap_h2_pdt*MH2);








        W_sys=((data.W_stack+heater_ads+heater_deoxo+W_comp+W_fan+W_pump_cws+W_pump_dmw+W_pump_oxlye+W_pump_hylye+heater_lye+heater_anlye+W_refrcomp+W_refrfan));
        eta_sys = vap_h2_pdt.*285.8*10^3./(W_sys)*100;
        SEC_sys = ((W_sys))*J_kWh./(vap_h2_pdt*MH2);
        SEC_comp = (W_comp+W_fan)*J_kWh./(vap_h2_pdt*MH2);  
        data.SEC_comp = SEC_comp;
        data.SEC_sys = SEC_sys;
        data.eta_sys = eta_sys;
        data.W_sys = W_sys;
        data.glsep_O2 = glsep_O2;

        if isfield(data, varName)
            outDataOrig = data.(varName)(:);
            if length(outDataOrig) == numPoints
                thisOutputs(:, v) = outDataOrig;

            else
                warning('Mismatch or missing data for %s in file %s', varName, fileName);
            end
        else
            warning('Output variable %s missing in file %s', varName, fileName);
        end
    end
    
    % Construct input matrix by repeating scalar parameters for each current density
    inputMatrix = repmat(params, numPoints, 1);
    inputMatrix = [inputMatrix, currDensOrig];
    alph = 0.05;
    alph1 = 0.05;
    epsilon = 1e-10;
   % y_log = log(x_raw+epsilon);
    % 
    % thisOutputs(:,7) = thisOutputs(:,7).^alph ;
    % thisOutputs(:,8) = thisOutputs(:,8).^alph ;
    % thisOutputs(:,9) = thisOutputs(:,9).^alph ;
    % thisOutputs(:,16) = thisOutputs(:,16).^alph1 ;
    % thisOutputs(:,17) = thisOutputs(:,17).^alph1 ;
    
    % thisOutputs(:,7) = log(thisOutputs(:,7)+epsilon) ;
    % thisOutputs(:,8) = log(thisOutputs(:,8)+epsilon) ;
    % thisOutputs(:,9) = log(thisOutputs(:,9)+epsilon) ;
    % thisOutputs(:,16) = log(thisOutputs(:,16)+epsilon) ;
    % thisOutputs(:,17) = log(thisOutputs(:,17)+epsilon) ;
     lambda = -0.5;

    thisOutputs(:,7) = ((thisOutputs(:,7).^(lambda)-1)/lambda) ;
    thisOutputs(:,8) = ((thisOutputs(:,8).^(lambda)-1)/lambda) ;
    thisOutputs(:,9) = ((thisOutputs(:,9).^(lambda)-1)/lambda) ;
    thisOutputs(:,16) = ((thisOutputs(:,16).^(lambda)-1)/lambda) ;
    thisOutputs(:,17) = ((thisOutputs(:,17).^(lambda)-1)/lambda) ;

    allInputs = [allInputs; inputMatrix];
    allOutputs = [allOutputs; thisOutputs];
end

% Remove any rows with NaNs (due to missing data)
nanRows = any(isnan(allOutputs), 2) | any(isnan(allInputs), 2);
allInputs(nanRows, :) = [];
allOutputs(nanRows, :) = [];

% After collecting allInputs and allOutputs, but before scaling:
currDensCol = 7; % current density is in column 6
validIdx = allInputs(:, currDensCol) <= 15000; % logical index for valid rows

allInputs = allInputs(validIdx, :);
allOutputs = allOutputs(validIdx, :);

% Now proceed with scaling and the rest of your code


% Min-max scaling to [-1, 1]
minIn = min(allInputs, [], 1);
maxIn = max(allInputs, [], 1);
allInputsScaled = 2 * (allInputs - minIn) ./ (maxIn - minIn) - 1;
minOut = min(allOutputs, [], 1);
maxOut = max(allOutputs, [], 1);
allOutputsScaled = 2 * (allOutputs - minOut) ./ (maxOut - minOut) - 1;
%replacing Nans with zeros
 nanCols = all(isnan(allInputsScaled), 1) ;% | all(isnan(allOutputs), 1);
 allInputsScaled(:, nanCols) = [];
allInputs(:, nanCols) = [];
x_scl = allOutputsScaled(:,8);
figure;
histogram(x_scl,30);hold on;
%histogram(x_scl2,30);


% Save scaled data and scaling parameters for future inverse transform
save('preparedData_BoxCox_scaled.mat', 'allInputsScaled', 'allOutputsScaled', ...
    'minIn', 'maxIn', 'minOut', 'maxOut');
%save('preparedDataNEOMPLOT_scaled.mat', 'allInputs', 'allOutputs', ...
 %   'minIn', 'maxIn', 'minOut', 'maxOut');
fprintf('Processed %d files.\n', length(matFiles));
fprintf('Total usable samples: %d\n', size(allInputs,1));
toc;

