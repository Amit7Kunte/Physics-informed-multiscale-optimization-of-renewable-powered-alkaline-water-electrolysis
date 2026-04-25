import numpy as np
import scipy.io
import tensorflow as tf
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import math
#os.chdir(r"D:\Comsol_Tut\EquationBasedModelling\HTO_paper_models\latest_model_tcd\Neom_report_models\Publication Models\modified_model\Optimization_study\saved data\For Joule_pubPlots")
# Load scaling parameters
lam = -0.5
frac = 1    
gc =10**6
J_kWh = 2.7778E-7 # Joule to kWH conversion
MH2 = 0.002 # Mol. wt of H2
HHV_H2 = 285.8*1000*J_kWh/MH2 # HHV of H2 in kWH/kg
Cp_watvap_80 = 34.6 # J/molK
Cp_H2_80 = 29
Cp_O2_80 = 29.72
Cp_KOH = 3156.1 # electrolyte heat cc at 80C
rho_KOH = 1260 # electrolyte density at 80C
Cp_cw = 4184 # J/kgK
lat_heat_water = 44000
rho_cw = 1000 #kg/m^3 
mu_cw = 0.001 #viscosity of cooling water, Pas
R_const = 8.3145
D_tube = 0.0254*1.5
V_target = 3 # target velocity of liquid flow  inside HX at rated capacity
pt=1.25*D_tube # pitch length (triangular pitch)
de = 4.0 * (0.5 * pt * 0.86 * pt - 0.5 * np.pi * D_tube**2 / 4.0) / (0.5 * np.pi * D_tube) # Equivalent diameter for shell pressure drop calc. from KERN
alpha_delPHX = 0.2
alpha_delPlyect =0.2
FF =0.9 # correction applied to HX shell and tube
T_cws = 293.15
T_cwr = T_cws+5
T_evap = 17+273.15
T_cond = 273.15 + 36 # %condenser saturation
T_refrcomp = 273.15 + 45 # refr comp outlet temp
T_amb = 273.15+25
delT_aircool = 10
T_aircooler = T_amb + delT_aircool
U_desup = 30 # % 30 value given in heuristics of book  of turnton Gas to Gas
U_cond_refr = 500 # %%%% AIR COOLER OHTC is 500
R_const = 8.314
T_deoxo_out=273.15+150 # deoxidizer outlet temperature

eta_heater = 0.95
Z_comp =1.05
T_suc_comp = T_amb + 11
delT_air = 10 #
Cp_air = 1005 # J/(kg*K)
rho_air = 1.18 # kg/m^3
eta_mech_motor = 0.95 #
eta_mech_comp= 0.9 #
eta_isen_comp = 0.85 # 
P_final = 30 # % final h2 delivery pressure in bar abs
T_max_comp = 273.15 + 140 # max allowable h2 pressure
eta_fan =0.55
delP_fan = 500
k_isen = 1.41  #as given in the The Techno- Economics of
#Hydrogen
#Compression
#TECHNICAL BRIEF
rc_max = ((T_max_comp/T_suc_comp-1)*eta_isen_comp+1)**(k_isen/(k_isen-1))
Nc = 5# % No. of cells in stack
rho_steel= 7850 # density of kg/m^3;
Cp_steel = 475 # J/kg/K, Cp CS is also same as Cp SS
rho_Ni = 8900
Cp_Ni = 440 # J/kg/K
scaling = scipy.io.loadmat('preparedData_BoxCox_scaled.mat')
minIn = scaling['minIn'].flatten()
maxIn = scaling['maxIn'].flatten()
minIn = np.delete(minIn, 4)  # remove dpored input
maxIn = np.delete(maxIn, 4)
minOut = scaling['minOut'].flatten()
maxOut = scaling['maxOut'].flatten()

cd_min_sobol = 5000.0
cd_max_sobol = 15000.0

p_max_sobol = 10
p_min_sobol = 1

vin_min_sobol = 0.0075
vin_max_sobol = 0.05

Wsep_min_sobol = 0.1
Wsep_max_sobol = 0.485
cd_min = minIn[-1]
cd_max = maxIn[-1]

problem = {
    'num_vars': len(minIn),
    'names': ['Wgde', 'P', 'vin', 'dpore', 'Wsep', 'currentDensity'],
    'bounds': [
        (minIn[0], maxIn[0]),  # Wgde
        (p_min_sobol, p_max_sobol),  # P
        (minIn[2], maxIn[2]),  # vin
        (minIn[3], maxIn[3]),  # dpore
        (minIn[4], maxIn[4]),  # Wsep
        (cd_min_sobol, cd_max_sobol)  # j restricted
    ]
}

model = tf.keras.models.load_model('trainedANN_seed7084_boxcox_3.keras')

def scale_inputs(X):
    X_scaled = np.empty_like(X)
    for i in range(5):
        X_scaled[:, i] = 2 * (X[:, i] - minIn[i]) / (maxIn[i] - minIn[i]) - 1
    X_scaled[:, 5] = 2 * (X[:, 5] - cd_min) / (cd_max - cd_min) - 1
    return X_scaled

def unscale_outputs(Y_scaled):
    return 0.5 * (Y_scaled + 1) * (maxOut - minOut) + minOut

def inv_boxcox(y, lam, c=0.0):

    if lam == 0:
        return np.exp(y) - c
    else:
        return np.power(lam * y + 1.0, 1.0 / lam) - c
# Define your output names exactly as per your system


# Output labels (LaTeX-format for matplotlib)
output_labels = [
    r'$\mathit{SEC}_{sys}$ kWh/kg',          # 0
    r'$\mathit{SEC}_{stack}$ kWh/kg',        # 1
    r'$\mathit{SEC}_{lyecooler}$ kWh/kg',    # 2
    r'$\mathit{SEC}_{comp}$ kWh/kg',         # 3
    r'$\mathit{SEC}_{pumps}$ kWh/kg',        # 4
    r'$\mathit{SEC}_{H_2\,purif}$ kWh/kg',   # 5
    r'$\mathit{SEC}_{gas\text{-}liq\,sep}$ kWh/kg',  # 6
    'eta_volt',
    'eta_curr_stack',      
    r'$\Delta \mathit{T}_{stack}$ K',        # 7  (maxT)
    r'$\mathit{H}_2{}_{sys}$ Nm$^3$/hr',     # 8
    r'$\mathit{HTO}$ \%',   
  
                                                                                                                      # 9
]


output_labels1 = [
    r'$\mathit{SEC}_{sys}$ kWh/kg',          # 0
    r'$\mathit{SEC}_{stack}$ kWh/kg',        # 1
    r'$\mathit{SEC}_{lyecooler}$ kWh/kg',    # 2
    r'$\mathit{SEC}_{comp}$ kWh/kg',         # 3
    r'$\mathit{SEC}_{pumps}$ kWh/kg',        # 4
    r'$\mathit{SEC}_{H_2\,purif}$ kWh/kg',   # 5
    r'$\mathit{SEC}_{gas\text{-}liq\,sep}$ kWh/kg',  # 6
    'eta_volt',
    'eta_curr_stack',      
    r'$\Delta \mathit{T}_{stack}$ K',        # 7  (maxT)
    r'$\mathit{H}_2{}_{sys}$ Nm$^3$/hr',     # 8
    r'$\mathit{HTO}$ \%',   
 
                                                                                                                      # 9
]

# Original output names in y_pred 
orig_output_names = [
    'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O',
    'w_KOH_angl_out', 'w_KOH_gl_out', 'Q_gl_out', 'Q_angl_out','glsep_O2',
    'Q_cond_h2cooler', 'Q_cond_ads',
    'Q_cond_deoxo', 'T_gl_out', 'T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]


# Input labels (LaTeX-format)
input_labels = [
    r'$W$$_{GDE}$',
    r'$P$',
    r'$V$$_{in}$',
    r'$d$$_p$',
    r'$W$$_{SEP}$',
    r'$j$'
]

input_labels1 = [
    r'$W$$_{GDE}$ mm',
    r'$P$ bar(abs)',
    r'$V$$_{in}$ m/s',
    r'$d$$_p$ $\mu$m',
    r'$W$$_{SEP}$ mm',
    r'$j$ A/$\mathrm{m}^2$'
]
keep_names = [
    'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O',
    'w_KOH_angl_out', 'w_KOH_gl_out', 'Q_gl_out', 'Q_angl_out','glsep_O2',
    'Q_cond_h2cooler', 'Q_cond_ads',
    'Q_cond_deoxo', 'T_gl_out', 'T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]
#keep_names = output_labels1 

keep_indices = [orig_output_names.index(name) for name in keep_names]



# Generate samples and predict
n_samples = 1024*8
param_vals = saltelli.sample(problem, n_samples, calc_second_order=True)
param_vals_scaled = scale_inputs(param_vals)
y_pred_scaled = model.predict(param_vals_scaled, verbose=0)
y_pred = unscale_outputs(y_pred_scaled)

y_pred[:,6] = (inv_boxcox(y_pred[:,6], lam, 0.0))*gc
y_pred[:,6]= sm.nonparametric.lowess(
    endog= y_pred[:,6], exog=param_vals[:,5],
    frac= frac,         # larger -> smoother, smaller -> more local detail
    it=1,              # robustifying iterations; increase if needed
    return_sorted=False
)
y_pred[:,7] = (inv_boxcox(y_pred[:,7], lam, 0.0))*gc

y_pred[:,7]= sm.nonparametric.lowess(
    endog= y_pred[:,7], exog=param_vals[:,5],
    frac= frac,         # larger -> smoother, smaller -> more local detail
    it=1,              # robustifying iterations; increase if needed
    return_sorted=False
)
y_pred[:,8] = (inv_boxcox(y_pred[:,8], lam, 0.0))*gc
y_pred[:,14] = (inv_boxcox(y_pred[:,14], lam, 0.0))
y_pred[:,14]= sm.nonparametric.lowess(
    endog= y_pred[:,14], exog=param_vals[:,5],
    frac= frac,         # larger -> smoother, smaller -> more local detail
    it=1,              # robustifying iterations; increase if needed
    return_sorted=False
)


y_pred[:,15] = (inv_boxcox(y_pred[:,15], lam, 0.0))
y_pred[:,15]= sm.nonparametric.lowess(
    endog= y_pred[:,15], exog=param_vals[:,5],
    frac= frac,         # larger -> smoother, smaller -> more local detail
    it=1,              # robustifying iterations; increase if needed
    return_sorted=False
)

#H2_mixedHTO = y_pred[:,9]
# build a new matrix with modified outputs
#SEC_h2purif   = y_pred[:, i_ads] + y_pred[:, i_deoxo]
#SEC_gasliqsep = y_pred[:, i_chH2]   # just a rename, value unchanged




# 1) keep the other outputs you want (excluding SEC_ads, SEC_deoxo, old SEC_chiller_h2cooler)
'''keep_base = [
    orig_output_names.index('SEC_sys'),
    orig_output_names.index('maxT'),
    orig_output_names.index('SEC_stack'),
    orig_output_names.index('vap_h2_pdt'),
    orig_output_names.index('H_T_O'),
    orig_output_names.index('eta_curr_sys'),
    orig_output_names.index('eta_volt'),
    orig_output_names.index('SEC_chiller_lyecooler'),
    orig_output_names.index('SEC_comp'),
    orig_output_names.index('SEC_pumps'),
    orig_output_names.index('SEC_lyeheaters'),
    orig_output_names.index('H2_mixedToHTO')
]'''

#Y_keep = y_pred[:, keep_base]

Q_incell = param_vals[:,2]*0.008*param_vals[:,0]*0.001*Nc*10**6 # this is the theoretical value of the vol. flowrate going coolectively into the anodes and cathodes separately


# 2) append new combined outputs
#y_pred = np.column_stack([Y_keep, SEC_h2purif, SEC_gasliqsep])


#print(yy.shape)
Q_cond_ads = y_pred[:,10]*gc
Q_cond_deoxo =y_pred[:,11]*gc
Q_cond_h2cooler = y_pred[:,9]*gc


Cp_KOH_lye = (4.101*10**3-3.526*10**3*y_pred[:,5]  + 9.644*10**-1*(y_pred[:,12]-273.15)+1.776*(y_pred[:,12]-273.15)*y_pred[:,5])
Cp_KOH_anlye = (4.101*10**3-3.526*10**3*y_pred[:,4] + 9.644*10**-1*(y_pred[:,13]-273.15)+1.776*(y_pred[:,13]-273.15)*y_pred[:,4])

rho_KOH_gl_out = (1001.53 - 0.08343*(y_pred[:,12] - 273.15) - 0.004*(y_pred[:,12]-273.15)**2 + 5.51232*10**-6*(y_pred[:,12]-273.15)**3 - 8.21*10**-10*(y_pred[:,12]-273.15)**4)*np.exp(0.86*y_pred[:,5])
rho_KOH_angl_out = (1001.53 - 0.08343*(y_pred[:,13] - 273.15) - 0.004*(y_pred[:,13]-273.15)**2 + 5.51232*10**-6*(y_pred[:,13]-273.15)**3 - 8.21*10**-10*(y_pred[:,13]-273.15)**4)*np.exp(0.86*y_pred[:,4])

Q_lyecooler = np.maximum((y_pred[:,12]-273.15-80)*Cp_KOH_lye*y_pred[:,6]*rho_KOH_gl_out,0)  + np.maximum((y_pred[:,13]-273.15-80)*Cp_KOH_anlye*y_pred[:,7]*rho_KOH_angl_out,0) # lye cooler duty 

Q_lyecooler = np.maximum((y_pred[:,12]-273.15-80)*Cp_KOH_lye*Q_incell*rho_KOH_gl_out,0)  + np.maximum((y_pred[:,13]-273.15-80)*Cp_KOH_anlye*Q_incell*rho_KOH_angl_out,0) # lye cooler duty 

T_glsepHXout = 298.15

p_vapor_h2cool_out = 0.61121*np.exp((18.678-(T_glsepHXout-273.15)/234.5)*((T_glsepHXout-273.15)/(257.14+T_glsepHXout-273.15)))*1000 # vapor pressure of water at 25C and ambient conditions

Q_lyeheater = ((np.maximum((273.15+80-y_pred[:,12])*Cp_KOH_lye*y_pred[:,6]*rho_KOH_gl_out,0)  + np.maximum((273.15+80-y_pred[:,13])*Cp_KOH_anlye*y_pred[:,7]*rho_KOH_angl_out,0))) # lye heater duty , in W
Q_lyeheater = ((np.maximum((273.15+80-y_pred[:,12])*Cp_KOH_lye*Q_incell*rho_KOH_gl_out,0)  + np.maximum((273.15+80-y_pred[:,13])*Cp_KOH_anlye*Q_incell*rho_KOH_angl_out,0))) # lye heater duty , in W

m_cw_circ =   (Q_lyecooler + Q_cond_ads + Q_cond_deoxo + Q_cond_h2cooler)/(Cp_cw*(T_cwr-T_cws))

m_cw_circ_lyecool = Q_lyecooler/(Cp_cw*(T_cwr-T_cws))
m_cw_circ_adscool = Q_cond_ads/(Cp_cw*(T_cwr-T_cws))
m_cw_circ_deoxocool = Q_cond_deoxo/(Cp_cw*(T_cwr-T_cws))
m_cw_circ_h2cool = Q_cond_h2cooler/(Cp_cw*(T_cwr-T_cws))




adspec_H2O = 5E-06
od_ads = 1.2
m_ads = 0.030*gc # adsorbent mass in kg
t_b = 8*3600
t_des = 5/8*t_b
t_cool = 3/8*t_b
por_ads = 0.3 # porosity of the adsorber
phi = 1 #; %sphericity of adsorbent
ads_dia = 0.0015875 #; % 1/16 inch
B = 0.152 #;
C = 0.000136 #;
rho_ads = 650 # density of adsorbent
vol_ads = m_ads/rho_ads
vol_ads = od_ads * np.mean(vol_ads)
ads_D = 3.0   # 3.0 m ads diameter     
L_ads = vol_ads/(np.pi*ads_D**2/4)                          
mu_H2_25 = 1.9E-5 #; % kg/ms
rho_H2_25 = 0.08 #; % kg/m3
eta_ads = 0.6 #; % adsrobent utilization factor for non-ideality
q_max = 19.43 #; % mol/kg max. adsorption capacity of adsorbent
purge_ads = 0.5*10**-4*gc #; % mol/s , purge gas flow through regen circuit
Cp_ads = 900.00 #;%J/(kg·K);
H_ads = 50000 #; %[J/mol]; heta of adsorption
b_ref = 93.5 #; %1/Pa; 

vap_h2_pdt = y_pred[:,2]*gc # product gas flow rate at the candidate current density
b_T = b_ref*np.exp(H_ads/R_const*(1/T_glsepHXout-1/293.15)) #; % b value at 25C
p_H2O_adspec = np.mean(adspec_H2O*param_vals[:,1]) #; % par_pres of moisture in ads outlet h2 stream at 25C ads temperature
alpha_purge = 0.5  #;
x_H2O_h2cool_out = p_vapor_h2cool_out/(param_vals[:,1]*10**5)

dry_gas = vap_h2_pdt*(1-adspec_H2O)
vap_ads = vap_h2_pdt * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber

ads_H2O_in = (dry_gas*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
ads_H2O_out = (dry_gas*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;

ads_cap_rate = ads_H2O_in-ads_H2O_out #; % rate of moisture capture 
vel_ads = (vap_ads*R_const*T_glsepHXout/(param_vals[:,1]*10**5))/(np.pi*ads_D**2/4.0)
# delP_ads in Pa/m
delP_ads = (
    B * mu_H2_25 * 1000 * (vel_ads * 3.28 * 60)
    + C * (vel_ads * 3.28 * 60) ** 2 * (rho_H2_25 * 0.062428) * 6894.75729
)

# Ergun equation terms
term1 = (
    150 * mu_H2_25 * (1 - por_ads) ** 2
    / (por_ads ** 3 * ads_dia ** 2)
    * L_ads * vel_ads / phi
)

term2 = (
    1.75 * rho_H2_25 * (1 - por_ads)
    / (por_ads ** 3 * ads_dia)
    * L_ads * vel_ads ** 2 / phi ** 2
)

dP_ads = term1 + term2  # Pressure drop across the adsorber in Pa
qcc = ads_cap_rate*t_b/(m_ads*eta_ads) #; % required ads cc
ads_cc = m_ads*qcc*eta_ads #; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required
ads_rate = ads_cc/t_b
q_res = q_max-qcc
rq = q_res/q_max
des_rate = ads_cc/t_des #; % the reqd des_rate increases with current density as more mositure is in gas stream.
rp = purge_ads/(des_rate) #; % ratio of purge flow rate to des rate (based on desorb time)
xH2O_reg_out = (x_H2O_h2cool_out+1.0/rp)/(1.0+1.0/rp) #; % mositure moefrac at regen exit
xH2O_equil = 101.325/(param_vals[:,1]*10**2) # equil moisture content at 100C and pressure of operation
p_H2O_reg_out = param_vals[:,1]*10**5*(x_H2O_h2cool_out+1.0/rp)/(1.0+1.0/rp)
p_vap_reg_eff = param_vals[:,1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out-x_H2O_h2cool_out))

b_reg = ((rq**0.2472)/(1.0-rq**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff)

T_reg = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg/b_ref))**-1)

T_base_vec = T_glsepHXout  # replicate as in MATLAB

Pg = param_vals[:,1] -1
thick_ads = np.maximum(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)
therm_cat_ads = m_ads*Cp_ads
therm_st_ads = m_steel_ads*Cp_steel

heater_ads = (purge_ads * Cp_H2_80 * (T_reg - T_base_vec) + des_rate * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg - T_base_vec) / t_des / eta_heater) * t_des / t_b) # in kW

condensate_H2O_regen_cond = (purge_ads * (1.0 - x_H2O_h2cool_out) * xH2O_reg_out / (1.0 - xH2O_reg_out)- x_H2O_h2cool_out * purge_ads)
cond_ads = (condensate_H2O_regen_cond*lat_heat_water+ purge_ads * Cp_H2_80 * (T_reg - T_glsepHXout)) * t_des / t_b  # adsorber condenser heat duty

m_cw_cond_ads = cond_ads / (Cp_cw * (T_cwr - T_cws))
#y_pred = y_pred[:, keep_indices]


od_deoxo = 1.2
GHSV = 2000.0   # per hr, reference ......
tcycle_deoxo = 8*3600 # number of seconds per cycle (8 hrs)
# Volume and geometry
vol_deoxo = od_deoxo * np.maximum(0.001* vap_h2_pdt * 22.414 * 3600.0 / GHSV, 0.0)
D_deoxo = 1.0   # m
L_deoxo = vol_deoxo / (np.pi * D_deoxo**2 / 4.0)
thick_deoxo = np.maximum(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness
mass_deoxo = vol_deoxo*650 # bulk density of deoxo catalyst is 650 kg/m^3
Cp_deoxo = 900
mass_st_deoxo = (np.pi*(D_deoxo+thick_deoxo)**2.0/4.0*L_deoxo-np.pi*(D_deoxo)**2.0/4.0*L_deoxo)*rho_steel  #%2mm thickness
therm_st_deoxo = mass_st_deoxo * Cp_steel / tcycle_deoxo #* np.ones(len(deoxo_heat)) thermal absorption rate of steel in the deoxidizer
therm_cat_deoxo = mass_deoxo * Cp_deoxo / tcycle_deoxo #* np.ones(len(deoxo_heat)) thermal absorption rate of catalyst in the deoxidizer


glsep_O2 = y_pred[:,8]
deoxo_H2O    = 2.0 * glsep_O2
deoxo_H2reac = 2.0 * glsep_O2
deoxo_heat   = 2.0 * glsep_O2 * 244.9 * 10.0**3   # 244.9 kJ/mol → J/mol

Cp_vap_h2cool = (
    Cp_watvap_80 * x_H2O_h2cool_out          
    + Cp_H2_80 * (1-x_H2O_h2cool_out)
)

T_deoxo = ( deoxo_heat
+ therm_cat_deoxo * (T_glsepHXout + 125.0)
+ therm_st_deoxo * (T_glsepHXout + 125.0)
+ vap_h2_pdt * Cp_vap_h2cool * T_glsepHXout) / (vap_h2_pdt * Cp_vap_h2cool + therm_st_deoxo + therm_cat_deoxo)
deoxo_T = 150 +273.15 # deoxo maintains temp. at 150C

heater_deoxo = 0.001*((deoxo_T > T_deoxo).astype(float)* vap_h2_pdt* Cp_vap_h2cool* (deoxo_T - T_deoxo)/ eta_heater) # heater_deoxo in kW
heater_regen = ((therm_st_deoxo + therm_cat_deoxo)* (723.15 - deoxo_T)/ eta_heater) # regen temp of deoxo is 450C


Ncomp = np.ceil(np.log(P_final / param_vals[:, 1]) / np.log(rc_max)) 
Ncomp = np.maximum(Ncomp, 1)       # at least one stage

od_h2comp = 1.15

        # multi‑stage: equal ratio per stage
rc = (P_final / (param_vals[:,1]))**(1.0 / (Ncomp))

T_dis_comp = T_suc_comp * (1 + (rc**((k_isen - 1.0) / k_isen) - 1) / eta_isen_comp);  

term_over = 0.0
W_comp = (np.maximum(Ncomp , 0) * k_isen / (k_isen - 1.0))* R_const * T_suc_comp * Z_comp * (rc ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt / (eta_isen_comp * eta_mech_motor * eta_mech_comp) # in kW+ (P_final / x[1] > 1.0)* (1.0 * k_isen / (k_isen - 1.0) * R_const * T_amb * Z_comp* (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt / (eta_isen_comp * eta_mech_motor * eta_mech_comp)))  # [web:1], in kW
intercool_comp = np.maximum(Ncomp, 0) * (T_dis_comp - T_suc_comp) * vap_h2_pdt * Cp_H2_80 #+ 1.0 * (T_max_comp - T_amb) * vap_h2_pdt * Cp_H2_80)  # [web:1]

air_fan= intercool_comp / (delT_air * Cp_air * rho_air)  # [web:1]

W_fan = air_fan* delP_fan / eta_fan  # [web:1]
# Check that currentDensity (column 5) is within the desired range

rho_H2O_20 = 1000            # kg/m^3
eta_isen_refrcomp = 0.7

refr_lat_heat_evap = 199.3 * 1000   # %j/kg, this is not used in the calc. of the m_refr, as the expansion valve outlet is a two phase mixture at the t and p of the evaporator ( both lower than cond)
refr_S_evap = 1781.8 # vapor entropy of Freon at 17C
Pevap = 1323.7 #kPa, pressure of evaporator at 17C
refr_H_comp_s = 444.9*1000 #443.6*1000 was for 50C #% 60C at the sampe entropy of the evap, but pressure of condenser,ideal compression
refr_H_cond_vap = 424.6*1000 # 427.0 * 1000 was for 40C # condensate vapor enthalpy
refr_H_cond_liq = 286.9*1000 # 267.1*1000 was for 40C
refr_H_evap_vap = 426.4*1000
refr_H_evap_liq = 227.1*1000
Pcond = 3051 #2411 was for 40C
rho_dm =1000 
refr_lat_evap = refr_H_evap_vap-refr_H_cond_liq #; % This is the actual latent heat from 2 phase mixture to vapor phase in the evap.
refr_H_comp = refr_H_evap_vap+(refr_H_comp_s-refr_H_evap_vap)/eta_isen_refrcomp #; ie 452.828 kJ/kg, coressponding to an actual comp discharge temp. of 65C % enthalpy at comp discharge for non-isen condition

cond_duty = refr_H_comp - refr_H_cond_liq 

delH_comp = refr_H_comp-refr_H_evap_vap

#%refr_H_cond = 427.4*1000;

refr_lat_heat_cond= 137.7*1000;              # %   considering condensation at 50C , 159.9*1000 ( was for 40C);
delH_desup = refr_H_comp-refr_H_cond_vap     # enthalpy loss in the desuperheater
m_refr = m_cw_circ*Cp_cw*(T_cwr-T_cws)/refr_lat_evap    #; % total refireigenrant circulation rate inside the chiller
m_refr_lyecool = m_cw_circ_lyecool*Cp_cw*(T_cwr-T_cws)/refr_lat_evap
m_refr_h2cool = m_cw_circ_h2cool*Cp_cw*(T_cwr-T_cws)/refr_lat_evap
m_refr_adscool = m_cw_circ_adscool*Cp_cw*(T_cwr-T_cws)/refr_lat_evap
m_refr_deoxo  = m_cw_circ_deoxocool*Cp_cw*(T_cwr-T_cws)/refr_lat_evap

W_refrcomp = (delH_comp*m_refr/(eta_mech_comp*eta_mech_motor))
W_refrfan = ((m_refr*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan

W_refr_lyecool = (delH_comp*m_refr_lyecool/(eta_mech_comp*eta_mech_motor)) + ((m_refr_lyecool*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan # refrigeration power consumed to lye cool heat duty
W_refr_h2cool = (delH_comp*m_refr_h2cool /(eta_mech_comp*eta_mech_motor)) + ((m_refr_h2cool *((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan
W_refr_adscool = (delH_comp*m_refr_adscool/(eta_mech_comp*eta_mech_motor)) + ((m_refr_adscool*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan
W_refr_deoxo = (delH_comp*m_refr_deoxo/(eta_mech_comp*eta_mech_motor)) + ((m_refr_deoxo*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan


eta_pump =0.55
U_lyecooler = 280 
L_tube=6 # length of tubes in m
vel_tubes_lyecooler  = 3.0
N_lye = 1
Re_tube = 1260*  vel_tubes_lyecooler*D_tube/0.00093 # Re number of tubes inside the HX
ff = (0.316*Re_tube**-0.25)
delP_catlyecooler = (1+alpha_delPHX)*ff*L_tube*1260*vel_tubes_lyecooler**2/(2*D_tube)*N_lye
delP_anlyecooler = (1+alpha_delPHX)*ff*L_tube*1260*vel_tubes_lyecooler**2/(2*D_tube)*N_lye
delP_catlyect  = (1+alpha_delPlyect)*delP_catlyecooler
delP_anlyect  = (1+alpha_delPlyect)*delP_anlyecooler
W_catlyepump =  (delP_catlyect + y_pred[:,14])*param_vals[:,2]*0.008*param_vals[:,0]*0.001*Nc*10**6/eta_pump             #*y_pred[:,6] /eta_pump 
W_anlyepump =   (delP_anlyect + y_pred[:,15])*param_vals[:,2]*0.008*param_vals[:,0]*0.001*Nc*10**6/eta_pump            #*y_pred[:,7] /eta_pump
W_pump =  (W_anlyepump + W_catlyepump) # power of the lye circulation pumps

############ SEC calc of AWEBOP unit operations
SEC_pumps =  0.001*W_pump/( vap_h2_pdt * MH2 * 3600 )
SEC_comp = 0.001*(W_comp+W_fan) /(vap_h2_pdt * MH2 * 3600)
SEC_lyecooler = 0.001*W_refr_lyecool/(vap_h2_pdt * MH2 * 3600)
SEC_stack = y_pred[:,1]
SEC_H2purif = 0.001*(W_refr_adscool + W_refr_deoxo+ heater_ads + heater_deoxo +heater_regen) /(vap_h2_pdt * MH2 * 3600)
SEC_gl_sep = 0.001*(W_refr_h2cool)/(vap_h2_pdt * MH2 * 3600)
SEC_refrig = 0.001*(W_refrcomp+ W_refrfan)/(vap_h2_pdt * MH2 * 3600)
SEC_sys = SEC_pumps + SEC_comp + SEC_lyecooler + SEC_stack + SEC_H2purif + SEC_gl_sep

#yy = y_pred[:, keep_indices]

yy = np.column_stack([
    SEC_sys,        # 0
    SEC_stack,      # 1
    SEC_lyecooler,  # 2
    SEC_comp,       # 3
    SEC_pumps,      # 4
    SEC_H2purif,    # 5
    SEC_gl_sep,     # 6
    y_pred[:,16],   # eta_volt
    y_pred[:,17],   # eta_curr_dens
    y_pred[:,0],    # 7  maxT
    y_pred[:,2],    # 8  H2sys
    y_pred[:,3],    # 9  HTO
])


'''j_samples = param_vals[:, 5]
print("Current density samples: min =", j_samples.min(), "max =", j_samples.max())

# Optional: quick sanity histogram (won't go in paper, just for you)

plt.hist(j_samples, bins=30)
plt.xlabel('currentDensity [A/m^2]')
plt.ylabel('Count')
plt.title('Sobol samples for j')
plt.show()
plt.pause(0.01)'''

n_out = yy.shape[1]
S1_matrix = np.zeros((n_out, problem['num_vars']))
ST_matrix = np.zeros_like(S1_matrix)

for i in range(n_out):
    Si = sobol.analyze(
        problem,
        yy[:, i],                  # use SEC / derived output i
        calc_second_order=True,
        print_to_console=False
    )
    S1_matrix[i, :] = Si['S1']
    ST_matrix[i, :] = Si['ST']


# Calculate Pearson correlation matrix
pearson_matrix = np.zeros_like(S1_matrix)

for i in range(n_out):
    for j in range(problem['num_vars']):
        pearson_matrix[i, j] = np.corrcoef(param_vals[:, j], yy[:, i])[0, 1]

# Modulate Sobol indices with Pearson sign for color encoding
S1_colored = S1_matrix * np.sign(pearson_matrix)
ST_colored = ST_matrix * np.sign(pearson_matrix)

fig, axes = plt.subplots(1, 2, figsize=(20, 14))

'''sns.heatmap(
    S1_colored, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
    xticklabels=input_labels, yticklabels=output_labels, ax=axes[0],
    linecolor='black', linewidths=0.5,
    vmin=-1, vmax=1
)
# Access the colorbar axis for the first heatmap
cbar_ax0 = axes[0].figure.axes[-1]
for label in cbar_ax0.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(11)

axes[0].set_title('First Order Sobol Indices (S1) with Pearson Direction', fontsize=16)
axes[0].set_xlabel('Inputs', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Outputs', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)'''

sns.heatmap(
    ST_colored, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
    xticklabels=input_labels, yticklabels=output_labels,ax=axes[1],
    linecolor='black', linewidths=0.5,
    vmin=-1, vmax=1
)
# Access the colorbar axis for the second heatmap
cbar_ax1 = axes[1].figure.axes[-1]
for label in cbar_ax1.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(11)

axes[1].set_title('Total Order Sobol Indices (ST) with Pearson Direction', fontsize=16)
axes[1].set_xlabel('Inputs', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Outputs', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
import pandas as pd

# Clean labels (no LaTeX) for CSV
input_names  = ['Wgde', 'P', 'Vin', 'dpore', 'Wsep', 'j']
output_names_kept = output_labels1 
#output_names_kept = keep_names  # or make a nicer list for Origin

# 1) Raw total Sobol indices
df_ST = pd.DataFrame(ST_matrix,
                     index=output_names_kept,
                     columns=input_names)
df_ST.to_csv('Sobol_ST_raw.csv')

# 2) Pearson-signed total indices (for colored plots)
df_ST_signed = pd.DataFrame(ST_colored,
                            index=output_names_kept,
                            columns=input_names)
#df_ST_signed.to_csv('Sobol_ST_signed.csv')

df_ST_rev = df_ST_signed.iloc[::-1, :]

# if you still want the transpose for Origin:
df_ST_rev.to_csv('Sobol_ST_raw_rev_T.csv')

# Select nominal (center) values for fixed inputs (mean of min and max)
nominal_vals = (minIn + maxIn) / 2
print("Nominal input values used for fixed parameters:")
for idx, val in enumerate(nominal_vals):
    print(f"{input_labels[idx]}: {val:.4f}")
fig, axes = plt.subplots(5, 3, figsize=(20, 25))
levels = 40  # number of contour levels

for i, ax in enumerate(axes.flatten()):
    # Get indices of 2 most influential inputs by ST for this output
    top2_idx = np.argsort(ST_matrix[i, :])[::-1][:2]

    # Generate 2D grid for these two inputs
    x_vals = np.linspace(minIn[top2_idx[0]], maxIn[top2_idx[0]], 50)
    y_vals = np.linspace(minIn[top2_idx[1]], maxIn[top2_idx[1]], 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare input matrix for predictions, fixed nominal values except top 2
    inp = np.tile(nominal_vals, (X.size, 1))
    inp[:, top2_idx[0]] = X.ravel()
    inp[:, top2_idx[1]] = Y.ravel()

    # Scale inputs
    inp_scaled = scale_inputs(inp)

    # Predict scaled outputs and unscale
    pred_scaled = model.predict(inp_scaled, verbose=0)
    pred = unscale_outputs(pred_scaled)[:, keep_indices]
    Z = pred[:, i].reshape(X.shape)

    # Plot contour lines only with color map from blue (low) to red (high)
    contour = ax.contour(
        X, Y, Z, levels=levels, cmap='coolwarm', linewidths=1.5
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt="%.2f")

    ax.set_xlabel(input_labels[top2_idx[0]], fontsize=12)
    ax.set_ylabel(input_labels[top2_idx[1]], fontsize=12)
    ax.set_title(output_labels1[i], fontsize=14)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(5, 3, figsize=(20, 25))
levels = 30  # number of contour levels

for i, ax in enumerate(axes.flatten()):
    # Get indices of 2 most influential inputs by ST for this output
    top2_idx = np.argsort(ST_matrix[i, :])[::-1][:2]

    # Generate 2D grid for these two inputs
    x_vals = np.linspace(minIn[top2_idx[0]], maxIn[top2_idx[0]], 50)
    y_vals = np.linspace(minIn[top2_idx[1]], maxIn[top2_idx[1]], 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Prepare input matrix for predictions, fixed at nominal except top 2
    inp = np.tile(nominal_vals, (X.size, 1))
    inp[:, top2_idx[0]] = X.ravel()
    inp[:, top2_idx[1]] = Y.ravel()

    # Scale and predict
    inp_scaled = scale_inputs(inp)
    pred_scaled = model.predict(inp_scaled, verbose=0)
    pred = unscale_outputs(pred_scaled)[:, keep_indices]
    Z = pred[:, i].reshape(X.shape)

    # Plot filled contours
    cf = ax.contourf(
        X, Y, Z, levels=levels, cmap='coolwarm'
    )
    #ax.set_xlabel(input_labels1[top2_idx[0]], fontsize=12)
    #ax.set_ylabel(input_labels1[top2_idx[1]], fontsize=12)
    #ax.set_title(output_labels1[i], fontsize=14)
    # Add colorbar for each subplot
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical')
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)
            # Axis labels and title
    ax.set_xlabel(input_labels1[top2_idx[0]], fontsize=14)
    ax.set_ylabel(input_labels1[top2_idx[1]], fontsize=14)
    ax.set_title(output_labels1[i], fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
plt.tight_layout()
plt.show()
plt.pause(0.01)