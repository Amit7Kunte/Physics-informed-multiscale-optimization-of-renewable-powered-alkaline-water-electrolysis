import os
#os.chdir(r"D:\Comsol_Tut\EquationBasedModelling\HTO_paper_models\latest_model_tcd\Neom_report_models\Publication Models\modified_model\Optimization_study\saved data\Shaheen")
#print(os.getcwd())  # Optionally verify


import numpy as np
import scipy.io
import tensorflow as tf
import time
import multiprocessing
from multiprocessing import Pool
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import PySAM.Windpower as windpower
import PySAM.Pvwattsv8 as pv
import scipy.io
import math
import statsmodels.api as sm
import inspect
jobid = os.getenv("JOBID_ENV", "noid")
#print("JOBID_ENV in Python:", repr(jobid), flush=True)
#----- physical constants---------

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
lam = -0.5
alpha_delPHX = 0.2
alpha_delPlyect =0.2
heater_frac_ratedpower = 0.05 # fraction of rated power used to design the heater rating
#----- stack degradation and plant model constants-----
degrade_stack = 10 # max degradation over stack lifetime
bj=0.0005 
#Sensitivity parameters
#m_degrate = 0.002# degradation rate ################################################################################################################################################################################### 
rated_load = 75 # rated current density ###############################################################################################################################################################################
genratio = 3.0 # ratio of hybrid generation rated cc to AWE system rated cc ###############################################################################################################################################
#cdeta_sweep_vals =  [0.0595]  #,0.08,0.1 ,0.11 ,0.13 ,0.15 ,0.18 ,0.2 ,0.22 ,0.25,0.28,0.3,0.35]
m_degrate_sweep = [0.001]
nj_sweep        = [0.5]
#genratio_sweep  = [1.2, 1.5, 1.8]
Tmax_constraint_value = 120
#nj = 0.2
sw = 1
t_sys = 40; # plant operation life in years
eta_pow = 0.98 # power electronics efficiency
T_thresh = 0 # threshold Tmax above which degradation starts
base_degrate = 0.125*8.760 # base stack degradation rate, % per year

r = .08 # disc rate
CRF = (r * (1 + r) ** t_sys) / ((1 + r) ** t_sys - 1)
gc =1000000 #scale-up factor, one Million times scale-up is used here
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

k_isen = 1.41 # as given in the The Techno- Economics of
#Hydrogen
#Compression
#TECHNICAL BRIEF
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
#rc_max1 = ((T_max_comp/T_amb-1)*eta_isen_comp+1)**(k_isen/(k_isen-1))
rc_max = ((T_max_comp/T_suc_comp-1)*eta_isen_comp+1)**(k_isen/(k_isen-1))
T_glsepHXout = 298




T_deoxo_out=273.15+150 # deoxidizer outlet temperature

# --- GA settings (best for pop=300) ---
pop_size = 1200
crossover_eta = 10
mutation_eta = 5
n_generations = 150

n_workers = 24 # Use all available cores, as per your workstation #####################################################################################################################################################
def log_callback(algorithm, *args, **kwargs):
    # getattr + default keeps it robust across versions
    n_eval = getattr(algorithm, "n_eval", None)
    # Many older pymoo versions store the current generation as `algorithm.n_gen`
    gen = getattr(algorithm, "n_gen", None)

    #print(f">>> GA progress: gen {gen} / {n_generations}, n_eval={n_eval}",
    #      flush=True)

# CAPEX calculation settings:

base = 397 #CEPCI cost index for base year in 2001.
now = 800
CI = now/base # cost index factor
conv_rate = 1.17 # Eur to USD conversion rate
Nc = 5# % No. of cells in stack
gc = 10**6 # This factor is used to multiply each functional unit, such as heat loads, shaft power etc, to bring the minimum size above the cost estimate bound. For the paper
rho_steel= 7850 # density of kg/m^3;
Cp_steel = 475 # J/kg/K, Cp CS is also same as Cp SS
rho_Ni = 8900
Cp_Ni = 440 # J/kg/K
sd = 1 # BOP cost scale down factor
#scale exponents for various equipments: Reference turnton
nc= 1 # compressor
nhe=1 # HE
nv = 1 # vessels
nf = 1 # blower
npp = 1 # pumps
# all vessels use SS clad construction

AllWsys70=[]


scaling_SEC = scipy.io.loadmat('preparedData_BoxCox_scaled.mat')
minIn_SEC     = scaling_SEC['minIn'].flatten()
maxIn_SEC     = scaling_SEC['maxIn'].flatten()
minIn_SEC     = np.delete(minIn_SEC, 4)
maxIn_SEC     = np.delete(maxIn_SEC, 4)
minOut_SEC    = scaling_SEC['minOut'][0]
maxOut_SEC    = scaling_SEC['maxOut'][0]
maxIn_SEC[1] =12

dt    = 1  
maxT_IDX            = 0
SEC_stack_IDX       = 1
vap_h2_pdt_IDX      = 2
H_T_O_IDX           = 3
w_KOH_angl_out_IDX  = 4
w_KOH_gl_out_IDX    = 5
Q_gl_out_IDX        = 6
Q_angl_out_IDX      = 7
glsep_O2_IDX        = 8
H2_mixedToHTO_IDX   = 9
Q_cond_h2cooler_IDX = 10
Q_cond_ads_IDX      = 11
Q_cond_deoxo_IDX    = 12
T_gl_out_IDX        = 13
T_angl_out_IDX      = 14
cell_delP_IDX       = 15
ancell_delP_IDX     = 16

J_kWh = 2.7778E-7
MH2 = 0.002
HHV_H2 = 285.8 * 1000 * J_kWh / MH2
#xl = np.array([0.7, 5.5711, 0.01096, 360, 0.101, minIn_SEC[-1]+600])
xu = np.array([2.0, 10, 0.05, 800, 0.485, maxIn_SEC[-1]])
#xu = np.array([0.7, 5.5711, 0.01096, 360, 0.101, maxIn_SEC[-1]])
xl = np.array([0.7, 1, 0.0075, 200, 0.1, minIn_SEC[-1]+600])

################################### CALCULATE THE HYBRID PV-WIND POWER GENERATION PROFILE ##################################
default_model = pv.default("PVWattsNone")
default_model.SystemDesign.system_capacity = 1.0 # temporary 1 unit of power
default_model.SystemDesign.dc_ac_ratio = 1
default_model.SystemDesign.array_type  = 0
default_model.SystemDesign.inv_eff     = 99.5
#default_model.SystemDesign.system_capacity = generatio/2 * (W_sys75)
default_model.SolarResource.solar_resource_file = "phoenix_az_33.450495_-111.983688_psmv3_60_tmy.csv"

default_model.execute(0)
pv_powerout = np.array(default_model.Outputs.dc)  # kW, 8760

# Optionally save for MATLAB
#np.savetxt("pvgen_profile.csv", pv_powerout, delimiter=",")

default_wind = windpower.default("WindPowerNone")
default_wind.Resource.wind_resource_filename= "AZ Eastern-Rolling Hills.srw"
#efault_model.Farm.wind_farm_sizing_mode
n_turbines = len(default_wind.Farm.wind_farm_xCoordinates)
P_farm_current = default_wind.Farm.system_capacity  # kW
P_turbine = P_farm_current / n_turbines
#n_turb = int(round(genratio/2*W_sys75/ P_turbine)) # in kW
n_turb = 1
default_wind.Farm.wind_farm_xCoordinates= [0.0]
default_wind.Farm.wind_farm_yCoordinates = [0.0]

default_wind.execute(0)
wind_powerout = default_wind.Outputs.gen
wind_powerout = np.array(wind_powerout)*1000
#np.savetxt("windgen_profile.csv", wind_powerout, delimiter=",")

SEC_model = None

def inv_boxcox(y, lam, c=0.0):
    """Inverse of x_bc = ((x+c)**lam - 1)/lam  (lam != 0)
       or x_bc = log(x+c) for lam == 0.
    """
    if lam == 0:
        return np.exp(y) - c
    else:
        return np.power(lam * y + 1.0, 1.0 / lam) - c
    

def get_sec_model():
    global SEC_model
    if SEC_model is None:
        #SEC_model = tf.keras.models.load_model('trainedANN_seed1273.keras')
         SEC_model = tf.keras.models.load_model('trainedANN_seed7084_boxcox_3.keras')
    return SEC_model

class AWEMultiObjVarConst(ElementwiseProblem):
    def __init__(self, minIn_SEC, maxIn_SEC, minOut_SEC, maxOut_SEC,
                 #minIn_CAPEX, maxIn_CAPEX, minOut_CAPEX, maxOut_CAPEX,
                maxT_IDX,          # 0
                SEC_stack_IDX,     # 1
                vap_h2_pdt_IDX,    # 2
                H_T_O_IDX,         # 3
                w_KOH_angl_out_IDX,# 4
                w_KOH_gl_out_IDX,  # 5
                Q_gl_out_IDX,      # 6
                Q_angl_out_IDX,    # 7
                glsep_O2_IDX,      # 8
                H2_mixedToHTO_IDX, # 9
                Q_cond_h2cooler_IDX,#10
                Q_cond_ads_IDX,    #11
                Q_cond_deoxo_IDX,  #12
                T_gl_out_IDX,      #13
                T_angl_out_IDX,    #14
                cell_delP_IDX, 
                ancell_delP_IDX,
                HHV_H2, xl, xu, Tmax_constraint_value=120.0, cd_eta75_base_ratio=0.1, m_degrate=0.002, nj=0.0, elementwise_runner=None, **kwargs):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=3, xl=xl, xu=xu, elementwise_runner=elementwise_runner, **kwargs)

        self.minIn_SEC = minIn_SEC
        self.maxIn_SEC = maxIn_SEC
        self.minOut_SEC = minOut_SEC
        self.maxOut_SEC = maxOut_SEC
        
        #self.SEC_sys_IDX = SEC_sys_IDX
        self.vap_h2_pdt_IDX = vap_h2_pdt_IDX
        self.SEC_stack_IDX = SEC_stack_IDX
        self.maxT_IDX = maxT_IDX
        self.HTO_IDX = H_T_O_IDX
        self.w_KOH_angl_out_IDX = w_KOH_angl_out_IDX
        self.w_KOH_gl_out_IDX= w_KOH_gl_out_IDX
        self.Q_gl_out_IDX = Q_gl_out_IDX
        self.Q_angl_out_IDX = Q_angl_out_IDX
        self.glsep_O2_IDX= glsep_O2_IDX
        self.H2_mixedToHTO_IDX = H2_mixedToHTO_IDX
        self.Q_cond_h2cooler_IDX = Q_cond_h2cooler_IDX
        self.Q_cond_ads_IDX = Q_cond_ads_IDX
        self.Q_cond_deoxo_IDX = Q_cond_deoxo_IDX
        #self.W_refrcomp_IDX = W_refrcomp_IDX 
        self.T_gl_out_IDX = T_gl_out_IDX
        self.T_angl_out_IDX = T_angl_out_IDX
        self.cell_delP_IDX = cell_delP_IDX
        self.ancell_delP_IDX = ancell_delP_IDX
        #self.T_reg_IDX = T_reg_IDX
        
        self.HHV_H2 = HHV_H2
        self.Tmax_constraint_value = Tmax_constraint_value
        self.cd_eta75_base_ratio = cd_eta75_base_ratio
        self.m_degrate = m_degrate
        self.nj = nj
       # self.genratio = genratio
        self.xk = self.xb = self.xc = self.xd = self.xe =self.xf =self.xg =self.xh =self.xiii =self.xjjj =self.xll =self.xm =self.xn =0
        #xa=0;xb=0;xc=0;xd=0;xe=0;xf=0;xg=0;xh=0;xi=0;xj=0;xl=0;xm=0;xn=0
    def _evaluate(self, x, out, *args, **kwargs):
      if not hasattr(self, 'SEC_model'):
        self.SEC_model  = get_sec_model() 
      #x = np.array([0.7, 1, 0.05, 500, 0.1, maxIn_SEC[-1]])

      SEC_model = self.SEC_model
      #CAPEX_model = self.CAPEX_model
      minIn_SEC, maxIn_SEC = self.minIn_SEC, self.maxIn_SEC
      minOut_SEC, maxOut_SEC = self.minOut_SEC, self.maxOut_SEC
      #minIn_CAPEX, maxIn_CAPEX = self.minIn_CAPEX, self.maxIn_CAPEX
      #minOut_CAPEX, maxOut_CAPEX = self.minOut_CAPEX, self.maxOut_CAPEX
    
      x_sec_scaled = 2 * (x - minIn_SEC) / (maxIn_SEC - minIn_SEC) - 1
      x_sec_scaled = x_sec_scaled.reshape(1, -1)
      #x_capex_scaled = 2 * (x[:5] - minIn_CAPEX) / (maxIn_CAPEX - minIn_CAPEX) - 1
      #x_capex_scaled = x_capex_scaled.reshape(1, -1)

      # run ANN and extract relevant values to calculate sec_avg_sys
      #pred_sec = SEC_model.predict(x_sec_scaled, verbose=0)
      #pred_sec_unscaled = 0.5 * (pred_sec + 1) * (maxOut_SEC - minOut_SEC) + minOut_SEC
      #SEC_sys = pred_sec_unscaled[0, self.SEC_sys_IDX]/eta_pow
      #SEC_sta = pred_sec_unscaled[0, self.SEC_stack_IDX]
      #vap_h2_pdt = pred_sec_unscaled[0, self.vap_h2_pdt_IDX]*gc
      #maxT = pred_sec_unscaled[0, self.maxT_IDX]
      
      #g1 = maxT - self.Tmax_constraint_value
    #=====contsraint II========
      cd_grid = np.linspace(100, maxIn_SEC[-1], 500)
      param_mat = np.tile(x[:5], (len(cd_grid), 1))
      grid_in = np.hstack([param_mat, cd_grid[:, None]])
      grid_in_scaled = 2 * (grid_in - minIn_SEC) / (maxIn_SEC - minIn_SEC) - 1
      y_grid = SEC_model.predict(grid_in_scaled, verbose=0)
      y_grid_unscaled = 0.5 * (y_grid + 1) * (maxOut_SEC - minOut_SEC) + minOut_SEC
      j = cd_grid
      j = j.reshape(-1)           # current density, length N
      frac = 1    
      #SEC_lyeheaters = y_grid_unscaled[:,self.SEC_lyeheaters_IDX]
      #SEC_system        = y_grid_unscaled[:, self.SEC_sys_IDX]/eta_pow
      maxT_s           = y_grid_unscaled[:, self.maxT_IDX]
      SEC_stack_s      = y_grid_unscaled[:, self.SEC_stack_IDX]
      vap_h2_pdts      = y_grid_unscaled[:, self.vap_h2_pdt_IDX]*gc
      HTO_s            = y_grid_unscaled[:, self.HTO_IDX]
      w_KOH_angl_out_s = y_grid_unscaled[:, self.w_KOH_angl_out_IDX]
      w_KOH_gl_out_s   = y_grid_unscaled[:, self.w_KOH_gl_out_IDX]
      Q_gl_out_s       = y_grid_unscaled[:, self.Q_gl_out_IDX]

      Q_gl_out_s = (inv_boxcox(Q_gl_out_s, lam, 0.0))*gc
      Q_gl_out_s = sm.nonparametric.lowess(
          endog=Q_gl_out_s, exog=j,
          frac=frac,         # larger -> smoother, smaller -> more local detail
          it=1,              # robustifying iterations; increase if needed
          return_sorted=False
      )

      Q_angl_out_s     = y_grid_unscaled[:, self.Q_angl_out_IDX]
      Q_angl_out_s = (inv_boxcox(Q_angl_out_s, lam, 0.0))*gc
      Q_angl_out_s = sm.nonparametric.lowess(
          endog=Q_angl_out_s, exog=j,
          frac=frac,         # larger -> smoother, smaller -> more local detail
          it=1,              # robustifying iterations; increase if needed
          return_sorted=False
      )

      H2_mixedToHTO_s  = y_grid_unscaled[:, self.H2_mixedToHTO_IDX]
      Q_cond_h2cooler_s  = y_grid_unscaled[:, self.Q_cond_h2cooler_IDX]*gc
      Q_cond_ads_s       = y_grid_unscaled[:, self.Q_cond_ads_IDX]*gc
      Q_cond_deoxo_s     = y_grid_unscaled[:, self.Q_cond_deoxo_IDX]*gc
      #W_refrcomp     = y_grid_unscaled[:, self.W_refrcomp_IDX]*gc
      T_gl_out_s         = y_grid_unscaled[:, self.T_gl_out_IDX]
      T_angl_out_s       = y_grid_unscaled[:, self.T_angl_out_IDX]
      glsep_O2_s         = y_grid_unscaled[:, self.glsep_O2_IDX]
      glsep_O2_s = (inv_boxcox(glsep_O2_s, lam, 0.0))*gc

      ancell_delP_s  = y_grid_unscaled[:, self.ancell_delP_IDX]
      ancell_delP_s = (inv_boxcox(ancell_delP_s, lam, 0.0))
      ancell_delP_s = sm.nonparametric.lowess(
          endog=ancell_delP_s, exog=j,
          frac=frac,         # larger -> smoother, smaller -> more local detail
          it=1,              # robustifying iterations; increase if needed
          return_sorted=False
      )
      cell_delP_s    = y_grid_unscaled[:, self.cell_delP_IDX]
      cell_delP_s = (inv_boxcox(cell_delP_s, lam, 0.0))
      cell_delP_s = sm.nonparametric.lowess(
        endog=cell_delP_s, exog=j,
        frac=frac,         # larger -> smoother, smaller -> more local detail
        it=1,              # robustifying iterations; increase if needed
        return_sorted=False
      )

      #T_reg          = y_grid_unscaled[:, self.T_reg_IDX]
              # fraction of data used in each local fit (tune this)

        #y_smooth = y_pred.copy()    # y_pred is after inv_boxcox

      #interp_SEC_sys        = interp1d(cd_grid, SEC_system,     kind='linear', fill_value="extrapolate")
      interp_maxT           = interp1d(cd_grid, maxT_s,           kind='linear', fill_value="extrapolate")
      interp_SEC_stack      = interp1d(cd_grid, SEC_stack_s,      kind='linear', fill_value="extrapolate")
      interp_vap_h2_pdt     = interp1d(cd_grid, vap_h2_pdts,     kind='linear', fill_value="extrapolate")
      interp_H_T_O          = interp1d(cd_grid, HTO_s,          kind='linear', fill_value="extrapolate")
      interp_w_KOH_angl_out = interp1d(cd_grid, w_KOH_angl_out_s, kind='linear', fill_value="extrapolate")
      interp_w_KOH_gl_out   = interp1d(cd_grid, w_KOH_gl_out_s,   kind='linear', fill_value="extrapolate")
      interp_Q_gl_out       = interp1d(cd_grid, Q_gl_out_s,       kind='linear', fill_value="extrapolate")
      interp_Q_angl_out     = interp1d(cd_grid, Q_angl_out_s,     kind='linear', fill_value="extrapolate")
      interp_H2_mixedToHTO  = interp1d(cd_grid, H2_mixedToHTO_s,  kind='linear', fill_value="extrapolate")
      interp_Q_cond_h2cooler = interp1d(cd_grid, Q_cond_h2cooler_s, kind='linear', fill_value="extrapolate")
      interp_Q_cond_ads     = interp1d(cd_grid, Q_cond_ads_s,     kind='linear', fill_value="extrapolate")
      interp_Q_cond_deoxo   = interp1d(cd_grid, Q_cond_deoxo_s,   kind='linear', fill_value="extrapolate")
      #interp_W_refrcomp     = interp1d(cd_grid, W_refrcomp,     kind='linear', fill_value="extrapolate")
      interp_T_gl_out       = interp1d(cd_grid, T_gl_out_s,       kind='linear', fill_value="extrapolate")
      interp_T_angl_out     = interp1d(cd_grid, T_angl_out_s,     kind='linear', fill_value="extrapolate")
      #interp_T_reg          = interp1d(cd_grid, T_reg,          kind='linear', fill_value="extrapolate")
      interp_glsep_O2       = interp1d(cd_grid, glsep_O2_s,     kind='linear', fill_value="extrapolate")
      interp_ancell_delP      = interp1d(cd_grid, ancell_delP_s,     kind='linear', fill_value="extrapolate")
      interp_cell_delP      = interp1d(cd_grid, cell_delP_s,     kind='linear', fill_value="extrapolate")
      

#      W_sys75 = SEC_sys_at_eta75 * vap_h2_pdt_eta75 * MH2 * 3600 # rated input power to system, 'rated' therefore no consdieration of degradation in the calculation
 #     W_sys_peak = SEC_sys_at_peak * vap_h2_pdt_peak * MH2 * 3600 # peak allowable input power to system, 10% above the rated curr dens.


      eta_stack = (self.HHV_H2 / SEC_stack_s) * 100
      idx = np.where(np.diff(np.sign(eta_stack - rated_load)))[0]

      cd = x[-1]

      maxT          = interp_maxT(cd)
      SEC_stack     = interp_SEC_stack(cd)
      vap_h2_pdt    = interp_vap_h2_pdt(cd)
      HTO         = interp_H_T_O(cd)
      w_KOH_angl_out = interp_w_KOH_angl_out(cd)
      w_KOH_gl_out   = interp_w_KOH_gl_out(cd)
      Q_gl_out      = interp_Q_gl_out(cd)
      Q_angl_out    = interp_Q_angl_out(cd)
      H2_mixedToHTO = interp_H2_mixedToHTO(cd)
      Q_cond_h2cooler = interp_Q_cond_h2cooler(cd)
      Q_cond_ads    = interp_Q_cond_ads(cd)
      Q_cond_deoxo  = interp_Q_cond_deoxo(cd)
      T_gl_out      = interp_T_gl_out(cd)
      T_angl_out    = interp_T_angl_out(cd)
      glsep_O2      = interp_glsep_O2(cd)
      ancell_delP   = interp_ancell_delP(cd)
      cell_delP   =   interp_cell_delP(cd)

      Cp_KOH_lye = (4.101*10**3-3.526*10**3*w_KOH_gl_out  + 9.644*10**-1*(T_gl_out-273.15)+1.776*(T_gl_out-273.15)*w_KOH_gl_out)
      Cp_KOH_anlye = (4.101*10**3-3.526*10**3*w_KOH_angl_out + 9.644*10**-1*(T_angl_out-273.15)+1.776*(T_angl_out-273.15)*w_KOH_angl_out)

      rho_KOH_gl_out = (1001.53 - 0.08343*(T_gl_out - 273.15) - 0.004*(T_gl_out-273.15)**2 + 5.51232*10**-6*(T_gl_out-273.15)**3 - 8.21*10**-10*(T_gl_out-273.15)**4)*np.exp(0.86*w_KOH_gl_out)
      rho_KOH_angl_out = (1001.53 - 0.08343*(T_angl_out - 273.15) - 0.004*(T_angl_out-273.15)**2 + 5.51232*10**-6*(T_angl_out-273.15)**3 - 8.21*10**-10*(T_angl_out-273.15)**4)*np.exp(0.86*w_KOH_angl_out)

      Q_lyecooler = np.maximum((T_gl_out-273.15-80)*Cp_KOH_lye*Q_gl_out*rho_KOH_gl_out,0)  + np.maximum((T_angl_out-273.15-80)*Cp_KOH_anlye*Q_angl_out*rho_KOH_angl_out,0) # lye cooler duty at candidate load
      

      p_vapor_h2cool_out = 0.61121*np.exp((18.678-(T_glsepHXout-273.15)/234.5)*((T_glsepHXout-273.15)/(257.14+T_glsepHXout-273.15)))*1000 # vapor pressure of water at 25C and ambient conditions

      Q_lyeheater = 0.001*((np.maximum((273.15+80-T_gl_out)*Cp_KOH_lye*Q_gl_out*rho_KOH_gl_out,0)  + np.maximum((273.15+80-T_angl_out)*Cp_KOH_anlye*Q_angl_out*rho_KOH_angl_out,0))) # lye heater power , in kW

      

          # ==============================
    
      if len(idx) == 0:
        g2 = 1e6
      #  print("entered g2 block")
      else:
        i = idx[-1]
        cd1, cd2 = cd_grid[i], cd_grid[i + 1]
        eta1, eta2 = eta_stack[i], eta_stack[i + 1]
        cd_eta75 = cd1 + (rated_load - eta1) * (cd2 - cd1) / (eta2 - eta1)
        
        hto_idxs = np.where(HTO_s > 2)[0]
        if len(hto_idxs) == 0:
            g2 = 1e6
       #     print("entered g2 block")
        else:
            cd_hto = cd_grid[hto_idxs[-1]]
            g2 = cd_hto / cd_eta75 - self.cd_eta75_base_ratio

    #  print("g2 is =",g2)
      SEC_stack_at_eta75 = interp_SEC_stack(cd_eta75)
      vap_h2_pdt_eta75 = interp_vap_h2_pdt(cd_eta75)
      #SEC_sys_at_peak = interp_SEC_stack(cd_eta75*1.1)
      vap_h2_pdt_peak = interp_vap_h2_pdt(cd_eta75*1.1)
      Q_gl_out_at_eta75 = interp_Q_gl_out(cd_eta75)
      Q_angl_out_at_eta75 = interp_Q_angl_out(cd_eta75)
      T_angl_out_at_eta75 = interp_T_angl_out(cd_eta75)
      T_gl_out_at_eta75 = interp_T_gl_out(cd_eta75)
      w_gl_out_at_eta75 = interp_w_KOH_gl_out(cd_eta75)
      w_angl_out_at_eta75 = interp_w_KOH_angl_out(cd_eta75)
      Q_cond_h2cooler_at_eta75 = interp_Q_cond_h2cooler(cd_eta75)
      Q_cond_ads_at_eta75 = interp_Q_cond_ads(cd_eta75)
      Q_cond_deoxo_at_eta75 = interp_Q_cond_deoxo(cd_eta75)
      #glsep_O2_eta75  =   interp_glsep_O2(cd_eta75)

      glsep_O2_at_eta75 = interp_glsep_O2(cd_eta75)
      ancell_delP_at_eta75 = interp_ancell_delP(cd_eta75)
      cell_delP_at_eta75 = interp_cell_delP(cd_eta75)

      Cp_KOH_lye_at_eta75 = (4.101*10**3-3.526*10**3*w_gl_out_at_eta75 + 9.644*10**-1*(T_gl_out_at_eta75-273.15)+1.776*(T_gl_out_at_eta75-273.15)*w_gl_out_at_eta75)
      Cp_KOH_anlye_at_eta75 = (4.101*10**3-3.526*10**3*w_angl_out_at_eta75 + 9.644*10**-1*(T_angl_out_at_eta75-273.15)+1.776*(T_angl_out_at_eta75-273.15)*w_angl_out_at_eta75)

      rho_KOH_gl_out_at_eta75 = (1001.53 -0.08343*(T_gl_out_at_eta75-273.15) - 0.004*(T_gl_out_at_eta75-273.15)**2 + 5.51232*10**-6*(T_gl_out_at_eta75-273.15)**3 - 8.21*10**-10*(T_gl_out_at_eta75-273.15)**4)*np.exp(0.86*w_gl_out_at_eta75)
      rho_KOH_angl_out_at_eta75 = (1001.53 -0.08343*(T_angl_out_at_eta75-273.15) - 0.004*(T_angl_out_at_eta75-273.15)**2 + 5.51232*10**-6*(T_angl_out_at_eta75-273.15)**3 - 8.21*10**-10*(T_angl_out_at_eta75-273.15)**4)*np.exp(0.86*w_angl_out_at_eta75)

      Q_lyecooler_at_eta75 = np.maximum((T_gl_out_at_eta75-273.15-80)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0)  + np.maximum((T_angl_out_at_eta75-273.15-80)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0) # lye cooler duty at rated load    


      

      # Check thermoneutral condition: at rated point lye should require cooling
      if (T_gl_out_at_eta75 < 353.15) and (T_angl_out_at_eta75 < 353.15):
          big = 1e8  # large penalty value
          out["F"] = np.array([big, big])
          # mark constraints as violated so GA clearly discards this point
          out["G"] = np.array([1.0, 1.0, 1.0])
          out["aux"] = {"W_sys75": 0.0}
          return
        

    # bottom-up cost estimates of major BOP equipment for capex model          

      '''def plot_capex_pie(capex_components, title, x):
          labels = list(capex_components.keys())
          values = list(capex_components.values())

          fig, ax = plt.subplots()
          ax.pie(values, labels=labels, autopct='%1.1f%%')
          ax.set_title(title)

    # show x vector in the figure
          txt = "\n".join([f"x{i+1} = {val:.3g}" for i, val in enumerate(x)])
          fig.text(0.02, 0.02, txt, fontsize=8, va='bottom', ha='left')

          plt.show()'''



     # STACK
      Ni_foam = 0.5/10**-6; # $0.5/cm^3,https://shop.nanografi.com/battery-equipment/nickel-foam-for-battery-cathode-substrate-size-1000-mm-x-300-mm-x-1-6-mm/
      zf_sep = gc*150*0.05*0.008*conv_rate*Nc; # 150 Euro/m^2 from AGFA Data 'Present and future cost of alkaline and PEM electrolyser stacks'
      steel = gc*(0.05*0.008*0.003*2+0.05*0.008*0.001*(Nc-1))*rho_steel*4.5;    # 0.9 $ /kg for carbon steel, 4.5$/kg for SS316L. SS used for longer stack life and in advanced stack designs operating at higher cd.
      Ni =  gc*2*(0.05*0.008*x[0]*0.001*(Nc))*Ni_foam; # 15 $/kg Ni price as on 4/10/2025, based on volume of electrode
      Fp_stack = 1+0.2*(x[1]-1)/(15-1);# Saba etal 2018, reported around 20% increase in stack costs for a pressure increase from 1 bar to 15 bar, assuming a linear relationship between operating pressure and stack material costs
      stack_mat_cost = 1.3*(zf_sep+steel+Ni)*Fp_stack;  # $/kW, 30% additional for balance of stack components such as gaskets etc.
      dir_stack_cost = stack_mat_cost+ 0.5*stack_mat_cost+0.125*stack_mat_cost; # ratio is as 8:4:1 from 'Present and future cost of alkaline and PEM electrolyser stacks'.

      Tot_stack_cost = 2*dir_stack_cost # % including overheads $/kW

      #Calc. of cooling water flowrate at rated conditions
      #Cp_KOH

      Q_lyeheater_at_eta75 = 0.001*(np.maximum((273.15+80-T_gl_out_at_eta75)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0)  + np.maximum((273.15+80-T_angl_out_at_eta75)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0)) # lye heater duty at rated load in kW

      m_cw_circ_at_eta75 =   (Q_lyecooler_at_eta75 + Q_cond_ads_at_eta75 + Q_cond_deoxo_at_eta75 + Q_cond_h2cooler_at_eta75)/(Cp_cw*(T_cwr-T_cws)) # cooling water circulation rate in kg/s

      m_cw_circ =   (Q_lyecooler + Q_cond_ads + Q_cond_deoxo + Q_cond_h2cooler)/(Cp_cw*(T_cwr-T_cws)) # cooling water circulation rate in kg/s

 
      # Gas-Liquid Separator vessels
           # Pres vessels 2 min resid time considered....... 
      od_glsep = 1.15
      #Q_gl_out = x[2]*0.008*x[0]*0.001*Nc*gc # in m^3/s
      Pg = x[1]-1 # guage pressure
      vol_liq_glsep = 120 * (Q_gl_out_at_eta75 + Q_angl_out_at_eta75)                                   # m^3, 2 mins residence time
      vol_glsep     = od_glsep * vol_liq_glsep                      # overdesigned volume, 50% fill level
      glsep_D       = ((4 * vol_glsep) / (3 * np.pi)) ** (1.0 / 3.0)   # diameter [m]
      glsep_L       = 3 * glsep_D                                      # length [m]

      # Purchase cost
      CP_glsep = 2 * 10 ** (3.5565
                            + 0.3776 * np.log10(max(vol_glsep, 0.1))
                            + 0.09 * (np.log10(max(vol_glsep, 0.1))) ** 2)

      CP_glsep_min = 2 * 10 ** (3.5565
                                + 0.3776 * np.log10(0.1)
                                + 0.09 * (np.log10(0.1)) ** 2)

      # Pressure vessel factor
      Fp_glsep = max(((Pg + 1) * glsep_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0)

      FBM_glsep = 1.49 + 1.52 * 1.7 * Fp_glsep

      BM_glsep  =  FBM_glsep * CP_glsep * CI
      BM_0_glsep = CP_glsep*(1.49+1.52)*CI
        
      # 6/10 rule if minimum capacity governs
      if (CP_glsep == CP_glsep_min) or (vol_glsep > 650):
          BM_glsep_min = CP_glsep * FBM_glsep 
          BM_glsep     = 2 * sd * BM_glsep_min * (vol_glsep / 0.1) ** nv
          BM_0_glsep   = CP_glsep_min*(1.49+1.52) * CI
          self.xb += 1
      
      # Volume and geometry (now also includes thermal calc. based on either GHSV, or catalyst mass)
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
      b_T = b_ref*np.exp(H_ads/R_const*(1/T_glsepHXout-1/293.15)) #; % b value at 25C
      p_H2O_adspec = np.mean(adspec_H2O*x[1]) #; % par_pres of moisture in ads outlet h2 stream at 25C ads temperature
      alpha_purge = 0.5  #;
      x_H2O_h2cool_out = p_vapor_h2cool_out/(x[1]*10**5)
      vap_h2_pdt = interp_vap_h2_pdt(x[-1]) # product gas flow rate at the candidate current density
      dry_gas = vap_h2_pdt*(1-adspec_H2O)
      vap_ads = vap_h2_pdt * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber

      ads_H2O_in = (dry_gas*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
      ads_H2O_out = (dry_gas*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;

      ads_cap_rate = ads_H2O_in-ads_H2O_out #; % rate of moisture capture 
      vel_ads = (vap_ads*R_const*T_glsepHXout/(x[1]*10**5))/(np.pi*ads_D**2/4.0)
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
      xH2O_equil = 101.325/(x[1]*10**2) # equil moisture content at 100C and pressure of operation
      p_H2O_reg_out = x[1]*10**5*(x_H2O_h2cool_out+1.0/rp)/(1.0+1.0/rp)
      p_vap_reg_eff = x[1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out-x_H2O_h2cool_out))

      b_reg = ((rq**0.2472)/(1.0-rq**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff)

      T_reg = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg/b_ref))**-1)

      T_base_vec = T_glsepHXout  # replicate as in MATLAB

      thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

      m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)
      therm_cat_ads = m_ads*Cp_ads
      therm_st_ads = m_steel_ads*Cp_steel
      heater_ads = 0.001*(purge_ads * Cp_H2_80 * (T_reg - T_base_vec) + des_rate * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg - T_base_vec) / t_des / eta_heater) * t_des / t_b) # in kW
      condensate_H2O_regen_cond = (purge_ads * (1.0 - x_H2O_h2cool_out) * xH2O_reg_out / (1.0 - xH2O_reg_out)- x_H2O_h2cool_out * purge_ads)
      cond_ads = (condensate_H2O_regen_cond*lat_heat_water+ purge_ads * Cp_H2_80 * (T_reg - T_glsepHXout)) * t_des / t_b 
      m_cw_cond_ads = cond_ads / (Cp_cw * (T_cwr - T_cws))
      # Ergun equation terms for calc. of delP in the regen bed
      vel_ads = (purge_ads*R_const*T_glsepHXout/(x[1]*10**5))/(np.pi*ads_D**2/4.0)
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
      eta_blower = 0.7 # desorber blower efficiency
      delP_des = 1.3*(term1+term2) # factor of 1.3 for calc. of other pressure drop conponnets in the recirc loop of purge gas
      flow_purge_ads = purge_ads*R_const*(T_reg)/(x[1]*10**5) # flowrate of purge adsorber
      W_blower = flow_purge_ads*delP_des/eta_blower # blower power us typically much lower
      # Purchase cost
      CP_ads = 2 * 10 ** (3.4974
                          + 0.4485 * np.log10(max(vol_ads, 0.3))
                          + 0.1074 * (np.log10(max(2 * vol_ads, 0.3)))**2)

      CP_ads_min = 2 * 10 ** (3.4974
                              + 0.4485 * np.log10(0.3)
                              + 0.1074 * (np.log10(0.3))**2)

      # Pressure vessel factor
      Fp_ads = max(((Pg + 1) * ads_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0)

      FBM_ads = 2.25 + 1.82 * 1.7 * Fp_ads

      BM_ads = FBM_ads * CP_ads  * CI
      BM_0_ads = CP_ads*(2.25 + 1.82) * CI

      # 6/10 rule for minimum capacity
      if (CP_ads_min == CP_ads) or (vol_ads > 520):
          BM_ads_min = CP_ads_min * FBM_ads
          BM_ads = sd * BM_ads_min * (vol_ads / 0.3)**nv*CI
          BM_0_ads = CP_ads_min *(2.25 + 1.82)*CI
          self.xc += 1
      
      

      od_deoxo = 1.2
      GHSV = 2000.0   # per hr, reference ......
      tcycle_deoxo = 8*3600 # number of seconds per cycle (8 hrs)
      # Volume and geometry
      vol_deoxo = od_deoxo * max(0.001* vap_h2_pdt_eta75 * 22.414 * 3600.0 / GHSV, 0.0)
      D_deoxo = 1.0   # m
      L_deoxo = vol_deoxo / (np.pi * D_deoxo**2 / 4.0)
      thick_deoxo = max(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness
      mass_deoxo = vol_deoxo*650 # bulk density of deoxo catalyst is 650 kg/m^3
      Cp_deoxo = 900
      mass_st_deoxo = (np.pi*(D_deoxo+thick_deoxo)**2.0/4.0*L_deoxo-np.pi*(D_deoxo)**2.0/4.0*L_deoxo)*rho_steel  #%2mm thickness
      therm_st_deoxo = mass_st_deoxo * Cp_steel / tcycle_deoxo #* np.ones(len(deoxo_heat)) thermal absorption rate of steel in the deoxidizer
      therm_cat_deoxo = mass_deoxo * Cp_deoxo / tcycle_deoxo #* np.ones(len(deoxo_heat)) thermal absorption rate of catalyst in the deoxidizer



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

      heater_deoxo = 0.001*((deoxo_T > T_deoxo).astype(float)* vap_h2_pdt* Cp_vap_h2cool* (deoxo_T - T_deoxo)/ eta_heater)
      heater_regen = ((therm_st_deoxo + therm_cat_deoxo)* (723.15 - deoxo_T)/ eta_heater) # regen temp of deoxo is 450C

      deoxo_H2O_eta75 = 2.0 * glsep_O2_at_eta75
      deoxo_H2reac_eta75 = 2.0 * glsep_O2_at_eta75
      deoxo_heat_eta75   = 2.0 * glsep_O2_at_eta75 * 244.9 * 10.0**3   # 244.9 kJ/mol → J/mol
      Cp_vap_h2cool = (
          Cp_watvap_80 * x_H2O_h2cool_out          
          + Cp_H2_80 * (1-x_H2O_h2cool_out)
      )

      T_deoxo_at_eta75   = ( deoxo_heat_eta75 
      + therm_cat_deoxo * (T_glsepHXout + 125.0)
      + therm_st_deoxo * (T_glsepHXout + 125.0)
      + vap_h2_pdt_eta75 * Cp_vap_h2cool * T_glsepHXout) / (vap_h2_pdt_eta75 * Cp_vap_h2cool + therm_st_deoxo + therm_cat_deoxo)

      deoxo_T = 150 +273.15 # deoxo maintains temp. at 150C
      heater_deoxo_at_eta75 = 0.001*((deoxo_T > T_deoxo_at_eta75 ).astype(float)* vap_h2_pdt_eta75* Cp_vap_h2cool* (deoxo_T - T_deoxo_at_eta75)/ eta_heater)
      heater_regen = 0.001*((therm_st_deoxo + therm_cat_deoxo)* (723.15 - deoxo_T)/ eta_heater) # regen temp of deoxo is 450C



      # Purchase cost (2 desorbers)
      CP_de = 2 * 10 ** (3.4974
                        + 0.4485 * np.log10(max(vol_deoxo, 0.3))
                        + 0.1074 * (np.log10(max(vol_deoxo, 0.3)))**2)

      CP_de_min = 2 * 10 ** (3.4974
                            + 0.4485 * np.log10(0.3)
                            + 0.1074 * (np.log10(0.3))**2)

      # Pressure vessel factor
      Fp_deoxo = max(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0)

      FBM_de = 2.25 + 1.82 * 1.7 * Fp_deoxo

      BM_de = sd * FBM_de * CP_de* CI
      BM_0_de = (2.25 + 1.82)*CP_de*CI

      # 6/10 rule when minimum capacity governs
      if (CP_de_min == CP_de) or (vol_deoxo/2 > 520):
          BM_de_min = CP_de_min * FBM_de
          BM_de = sd * BM_de_min * (vol_deoxo / 0.3)**nv
          BM_0_de = CP_de_min * CI*(2.25 + 1.82)
          self.xd += 1

      # H2 compressor
      Ncomp = math.ceil(math.log(P_final / (x[1])) / math.log(rc_max)) 
      Ncomp = max(Ncomp, 1)       # at least one stage

      od_h2comp = 1.15

      if Ncomp <= 0:
        # physically impossible, punish the design
        rc = 1.0
      elif Ncomp == 1:
        # single‑stage compression: all ratio in one step
        rc = rc_max #P_final / (x[1] * rc_max1)
      else:
        # multi‑stage: equal ratio per stage
        rc = (P_final / (x[1]))**(1.0 / (Ncomp))

      T_dis_comp = T_suc_comp * (1 + (rc**((k_isen - 1.0) / k_isen) - 1) / eta_isen_comp);  
      '''if P_final / x[1] > 1.0:
          term_over = (k_isen / (k_isen - 1.0) *
          R_const * T_amb * Z_comp *
          (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0) *
          vap_h2_pdt_eta75 / (eta_isen_comp  * eta_mech_comp))
      else:'''
      term_over = 0.0

      W_shaft_h2comp = 0.001*(max(Ncomp, 0) * k_isen / (k_isen - 1.0) *R_const * T_suc_comp * Z_comp *(rc ** ((k_isen - 1.0) / k_isen) - 1.0) *vap_h2_pdt_eta75  / (eta_isen_comp *  eta_mech_comp)+ term_over) # in kW, at rated current density for CAPEX calc.
      W_comp_SEC = 0.001*(max(Ncomp , 0) * k_isen / (k_isen - 1.0))* R_const * T_suc_comp * Z_comp * (rc ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt / (eta_isen_comp * eta_mech_motor * eta_mech_comp) #+ (P_final / x[1] > 1.0)* (1.0 * k_isen / (k_isen - 1.0) * R_const * T_amb * Z_comp* (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt / (eta_isen_comp * eta_mech_motor * eta_mech_comp)))  # [web:1], in kW
      intercool_comp_SEC = max(Ncomp, 0) * (T_dis_comp - T_suc_comp) * vap_h2_pdt * Cp_H2_80 #+ 1.0 * (T_max_comp - T_amb) * vap_h2_pdt * Cp_H2_80)  # [web:1]

      air_fan_SEC = intercool_comp_SEC / (delT_air * Cp_air * rho_air)  # [web:1]

      W_fan_SEC = 0.001*air_fan_SEC * delP_fan / eta_fan  # [web:1]



      W_h2comp = W_shaft_h2comp/eta_mech_motor # in kW
      CP_shaft_h2comp=Ncomp*(10**(2.2897+1.36*np.log10(max((W_shaft_h2comp*od_h2comp)/Ncomp,20))-0.1027*(np.log10(max((W_shaft_h2comp*od_h2comp)/Ncomp,20)))**2)) #min. input pow is 450kW reciprocating, max is 3000kW per stage
      CP_shaft_h2comp_min =  Ncomp*(10**(2.2897+1.36*np.log10(np.max(20))-0.1027*(np.log10(np.max(20) ))**2)) # min. input pow is 450kW as per turton, but using limit of 300, as the function is smooth
      FBM_shaft_h2comp = 7
      BM_shaft_h2comp = sd * FBM_shaft_h2comp * CP_shaft_h2comp * CI
      BM_0_shaft_h2comp = BM_shaft_h2comp / FBM_shaft_h2comp
      if (CP_shaft_h2comp_min == CP_shaft_h2comp) or (W_shaft_h2comp*od_h2comp/Ncomp > 3000):
        BM_shaft_h2comp_min = CP_shaft_h2comp_min * FBM_shaft_h2comp * CI
        BM_shaft_h2comp = (sd* BM_shaft_h2comp_min* ((W_shaft_h2comp*od_h2comp / Ncomp) / 20.0) ** 0.86)
        BM_0_shaft_h2comp = BM_shaft_h2comp / FBM_shaft_h2comp * CI
        self.xf += 1


      od_refrcomp =1.20
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
      W_refrcomp_SEC = 0.001*(delH_comp*m_refr/(eta_mech_comp*eta_mech_motor))
      W_refrfan_SEC = 0.001*((m_refr*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan
  
      #W_refrcompshaft_at_eta75 = W_refrcomp_at_eta75*eta_mech_motor
      m_refr_at_eta75 = m_cw_circ_at_eta75*Cp_cw*(T_cwr-T_cws)/refr_lat_evap
      W_refrcomp_at_eta75 = 0.001*(delH_comp*m_refr_at_eta75/(eta_mech_comp*eta_mech_motor))
      W_refrfan_at_eta75 = 0.001*((m_refr_at_eta75*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan
      W_refrcompshaft_at_eta75 = W_refrcomp_at_eta75*eta_mech_motor # in kW

      CP_shaft_refrcomp = 10 ** (2.2897+ 1.36 * np.log10(max(od_refrcomp*W_refrcompshaft_at_eta75, 50.0))- 0.1027 * (np.log10(max(od_refrcomp*W_refrcompshaft_at_eta75, 50.0))) ** 2)
      CP_shaft_refrcomp_min = 10 ** (2.2897+ 1.36 * np.log10(100.0)- 0.1027 * (np.log10(100.0)) ** 2)
      FBM_shaft_refrcomp = 3.3
      # Base shaft costs
      BM_shaft_refrcomp = sd * FBM_shaft_refrcomp * CP_shaft_refrcomp* CI
      BM_0_shaft_refrcomp = BM_shaft_refrcomp / FBM_shaft_refrcomp
      # Refr. shaft min‑capacity check
      if (CP_shaft_refrcomp_min == CP_shaft_refrcomp) or (od_refrcomp*W_refrcompshaft_at_eta75 > 3000):
          BM_shaft_refrcomp_min = CP_shaft_refrcomp_min * FBM_shaft_refrcomp* CI
          BM_shaft_refrcomp = (sd * BM_shaft_refrcomp_min * (od_refrcomp*W_refrcompshaft_at_eta75 / 50.0) ** 0.86)
          BM_0_shaft_refrcomp = BM_shaft_refrcomp / FBM_shaft_refrcomp
          self.xe += 1
      
      # comp drive
      # Purchase costs
      CP_refrcomp = 10 ** (2.9308+ 1.0688 * np.log10(max(od_refrcomp*W_refrcomp_at_eta75, 5.0))- 0.1315 * (np.log10(max(od_refrcomp*W_refrcomp_at_eta75, 5.0))) ** 2)
      CP_refrcomp_min = 10 ** (2.9308+ 1.0688 * np.log10(5.0)- 0.1315 * (np.log10(5.0)) ** 2)
      CP_h2comp = Ncomp * 10 ** (2.9308+ 1.0688 * np.log10(max(od_h2comp*W_h2comp / Ncomp, 5.0))- 0.1315 * (np.log10(max(od_h2comp*W_h2comp / Ncomp, 5.0))) ** 2) # min. drive power is 75kW,2600kW max
      CP_h2comp_min = Ncomp * 10 ** (2.9308+ 1.0688 * np.log10(5.0)- 0.1315 * (np.log10(5.0)) ** 2)
      FBM_comp = 1.5
      BM_refrcomp = sd * FBM_comp * CP_refrcomp*CI
      BM_h2comp   = sd * FBM_comp * CP_h2comp*CI
      BM_0_h2comp   = BM_h2comp   / FBM_comp
      BM_0_refrcomp = BM_refrcomp / FBM_comp
      # Refrigeration compressor min-capacity scaling
      if (CP_refrcomp == CP_refrcomp_min) or (od_refrcomp*W_refrcomp_at_eta75> 4000):
          BM_refrcomp_min = CP_refrcomp_min * FBM_comp
          BM_refrcomp = sd * BM_refrcomp_min * (od_refrcomp*W_refrcomp_at_eta75 / 5.0) ** 0.6 * CI
          BM_0_refrcomp = BM_refrcomp / FBM_comp
          self.xg += 1
      # H2 compressor min-capacity scaling
      if (CP_h2comp == CP_h2comp_min) or (od_h2comp*W_h2comp / Ncomp > 4000):
          BM_h2comp_min = CP_h2comp_min * FBM_comp
          BM_h2comp = sd * BM_h2comp_min * (od_h2comp*W_h2comp / 5.0) ** 0.6*CI
          BM_0_h2comp = BM_h2comp / FBM_comp
          self.xh += 1
       
      # intercoolers , air cooled condensers and fans 
      # Intercooler duty
      od_aircool = 1.20
      intercool_comp =  (T_dis_comp - T_suc_comp) * vap_h2_pdt_eta75 * Cp_H2_80 
     
      U_intercool = 30 #%%%%%%%%%%%%%%% GAS TO GAS is 30
      delTlm_intercool = ((T_dis_comp - T_aircooler) - (T_suc_comp - T_amb)) / np.log((T_dis_comp - T_aircooler) / (T_suc_comp - T_amb))

      area_intercool = intercool_comp / (U_intercool * delTlm_intercool*FF) # area of each set of intercoolers per compressor


      # Number of intercoolers
      N_intercool = max(1, int(np.floor(od_aircool*area_intercool / 20000.0 + 0.8))) # No. of intercool, per compressor
      #N_intercoolobj[k] = N_intercool

      # Purchase cost
      CP_intercool = (Ncomp)*(N_intercool * 10 ** (4.0336+ 0.2341 * np.log10(max(od_aircool*area_intercool / N_intercool, 1.0))+ 0.0497 * (np.log10(max(od_aircool*area_intercool / N_intercool, 1.0))) ** 2))

      CP_intercool_min = (Ncomp)*(N_intercool * 10 ** (4.0336+ 0.2341 * np.log10(1.0)+ 0.0497 * (np.log10(1.0)) ** 2))

      FBM_intercool = 0.96 + 1.21 * 2.9   # SS construction

      BM_intercool = sd * FBM_intercool * CP_intercool*CI
      BM_0_intercool = CP_intercool*(0.96 + 1.21)*CI
      
      if (CP_intercool == CP_intercool_min) or (od_aircool*area_intercool / N_intercool > 11000):
          BM_intercool_min = CP_intercool_min * FBM_intercool * CI
          BM_intercool = sd * BM_intercool_min * (od_aircool*area_intercool / 10.0) ** nhe
          BM_0_intercool = CP_intercool_min*(0.96 + 1.21)*CI
          self.xll += 1
      
      # aircooled refr condenser
      T_cond = 273.15 + 50 #condenser saturation, ie 11C approach over ambient
      T_refrcomp = 273.15 + 65 # refr comp outlet temp , 55 C is the actual comp o/l temp from the Freon thermo tables, at the condernser pressure, and the eta_isen of 0.6
      Q_desup = delH_desup * m_refr_at_eta75
      Q_cond_refr = refr_lat_heat_cond * m_refr_at_eta75
      U_desup = 30
      U_cond_refr = 40 # both are close, as airside controls HT

      delTlm_cond_refr = ((T_cond - (T_amb+9)) - (T_cond - T_amb)) / np.log((T_cond - (T_amb+9)) / (T_cond - T_amb)) # Temp. rise of 1K is assumed across the desuperheater

      delTlm_desup = ((T_refrcomp - (T_aircooler)) - (T_cond - (T_amb+9))) / np.log((T_refrcomp - (T_aircooler)) / (T_cond - (T_amb+9)))

      area_desup = Q_desup / (U_desup * delTlm_desup*FF)
      area_cond_refr = Q_cond_refr / (U_cond_refr * delTlm_cond_refr*FF)
      area_refr = (area_desup + area_cond_refr) # total area of the air cooled refrigerant condenser

      N_refr = max(1, int(np.floor(od_aircool*area_refr / 20000.0 + 0.5))) # max area for air cooler is 10000 m^2
      CP_refr_min = N_refr * 10 ** (4.0336+ 0.2341 * np.log10(1.0)+ 0.0497 * (np.log10(1.0)) ** 2)

      CP_refr = N_refr * 10 ** (4.0336+ 0.2341 * np.log10(max(od_aircool*area_refr / N_refr, 1.0))+ 0.0497 * (np.log10(max(od_aircool*area_refr / N_refr, 1.0))) ** 2)

      Fp_refr = 10 ** (-0.125+ 0.15361 * np.log10(29.0 + np.finfo(float).eps)- 0.02861 * (np.log10(29.0 + np.finfo(float).eps)) ** 2) # 21 bar is the guage pressure of the refrigerant inside the vapor compression cycle

      FBM_refr = 0.96 + 1.21 * 1.0 * Fp_refr   # CS construction, 30 bar pressure

      BM_refr = sd * FBM_refr * CP_refr* CI
      BM_0_refr = (0.96+1.21)*CP_refr*CI

      if (CP_refr_min == CP_refr) or (od_aircool*area_refr / N_refr > 30000):
          BM_refr_min = CP_refr_min * FBM_refr * CI
          BM_refr = sd * BM_refr_min * (od_aircool*area_refr / 1.0) ** nhe
          BM_0_refr = (0.96+1.21)*CP_refr_min*CI
          self.xk += 1

      # Air flowrate and fan power
     
      od_fans = 1.20
      air_fan = intercool_comp / (delT_air * Cp_air * rho_air) # division by od_aircool because it being factored inside intercool_comp
      W_compfan = 0.001*air_fan * delP_fan / eta_fan
      N_fan = max(1, int(np.floor(air_fan / 100.0 + 0.8)))
      CP_fan = N_fan * 10 ** (3.5391- 0.3533 * np.log10(max(od_fans*air_fan / N_fan, 0.01))+ 0.4477 * (np.log10(max(od_fans*air_fan / N_fan, 0.01))) ** 2)

      CP_fan_min = N_fan * 10 ** (3.5391- 0.3533 * np.log10(0.01)+ 0.4477 * (np.log10(0.01)) ** 2)

      FBM_fan = 2.7
      BM_fan = sd * FBM_fan * CP_fan * CI
      BM_0_fan = BM_fan / FBM_fan

      if (CP_fan == CP_fan_min) or (od_fans*air_fan / N_fan > 110):
          BM_fan_min = CP_fan_min * FBM_fan* CI
          BM_fan = sd * BM_fan_min * (od_fans*air_fan / 0.01) ** nf
          BM_0_fan = BM_fan / FBM_fan
          self.xn += 1

      air_refrfan = m_refr_at_eta75*(delH_desup+refr_lat_heat_cond)/(delT_air*Cp_air*rho_air)# %in m^3/s
      W_refrfan = air_refrfan*delP_fan/eta_fan
      # Number of refrigeration fans
      N_refrfan = max(1, int(np.floor(od_fans*air_refrfan / 100.0 + 0.9)))

      # Fan purchase cost
      CP_refrfan = N_refrfan * 10 ** (3.5391- 0.3533 * np.log10(max(od_fans*air_refrfan / N_refrfan, 0.1))+ 0.4477 * (np.log10(max(od_fans*air_refrfan / N_refrfan, 0.1))) ** 2)

      CP_refrfan_min = N_refrfan * 10 ** (3.5391- 0.3533 * np.log10(0.1)+ 0.4477 * (np.log10(0.1)) ** 2)

      BM_refrfan = sd * FBM_fan * CP_refrfan* CI
      BM_0_refrfan = BM_refrfan / FBM_fan

      if (CP_refrfan == CP_refrfan_min) or (od_fans*air_refrfan / N_refrfan > 110):
          BM_refrfan_min = CP_refrfan_min * FBM_fan* CI
          BM_refrfan = sd * BM_refrfan_min * (od_fans*air_refrfan / 0.1) ** nf
          BM_0_refrfan = BM_refrfan / FBM_fan
          self.xm += 1

      BM_f = BM_fan+BM_refrfan
      BM_0_f = BM_0_fan+BM_0_refrfan
      
      # refr evaporator HX (shell and tube)
      od_HX = 1.15 
      Q_evap_refr = refr_lat_evap*m_refr_at_eta75
      U_evap = 850 # OHTC in W/m^2K;  
      delTlm_evap = ((T_cwr-T_evap) - (T_cws-T_evap))/(np.log((T_evap- T_cwr)/(T_evap-T_cws)))
      area_evap = Q_evap_refr / (U_evap * delTlm_evap * FF)
      N_evap = max(1, int(np.floor(od_HX*area_evap / 1000.0 + 0.9)))
      

      CP_evap_min = N_evap * 10 ** (4.1884- 0.2503 * np.log10(2)+ 0.1974 * (np.log10(2)) ** 2)

      CP_evap = N_evap * 10 ** (4.1884- 0.2503 * np.log10(max(od_HX*area_evap / N_evap, 2))+ 0.1974 * (np.log10(max(od_HX*area_evap / N_evap, 2))) ** 2)

      Fp_evap = 10 ** (0.03881- 0.11272 * np.log10(12.0 + np.finfo(float).eps)+ 0.08183 * (np.log10(12.0 + np.finfo(float).eps)) ** 2)

      FBM_evap = 1.63 + 1.66 * 1.0 * Fp_evap

      BM_evap = sd * FBM_evap * CP_evap * CI
      BM_0_evap = (1.63 + 1.66)*CP_evap*CI

      if (CP_evap == CP_evap_min) or (od_HX*area_evap / N_evap > 1100):
          BM_evap_min = CP_evap_min * FBM_evap* CI
          BM_evap = sd * BM_evap_min * (od_HX*area_evap / 2.0) ** nhe
          BM_0_evap = (1.63 + 1.66)*CP_evap_min*CI
          self.xjjj += 1
      
      ######################################### lye cooler HX (shell and tube) 
      
      #SEC_chiller_lyecooler_at_eta75 = interp_func5(cd_eta75)

      #Q_lyecool_eta75 = od_HX*1000*SEC_chiller_lyecooler_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 # must be in watts
      U_lyecooler = 280 #OHTC in W/m^2K; %%%%%%%%%%%Liquid to Liquid
      
      # Q_lyecooler_at_eta75

      #T_angl_rated = interp_func9(cd_eta75)
      #T_gl_rated = interp_func10(cd_eta75)

      Q_catlyecool_eta75 = np.maximum((T_gl_out_at_eta75-273.15-80)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0) 
      Q_anlyecool_eta75 = np.maximum((T_angl_out_at_eta75-273.15-80)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0) 

      #T_lyecooler_in = np.mean([T_angl_out_at_eta75, T_gl_out_at_eta75])

      delTlm_catlyecooler_at_eta75 = ((T_gl_out_at_eta75 - T_cwr) - (353.15 - T_cws)) / np.log((T_gl_out_at_eta75 - T_cwr) / (353.15 - T_cws))
      delTlm_anlyecooler_at_eta75 = ((T_angl_out_at_eta75- T_cwr) - (353.15 - T_cws)) / np.log((T_angl_out_at_eta75 - T_cwr) / (353.15 - T_cws))


      area_catlyecooler_at_eta75 = Q_catlyecool_eta75 / (U_lyecooler * delTlm_catlyecooler_at_eta75 * FF)

      area_anlyecooler_at_eta75 = Q_anlyecool_eta75 / (U_lyecooler * delTlm_anlyecooler_at_eta75 * FF)

      area_lyecooler_at_eta75 = max(area_catlyecooler_at_eta75,area_anlyecooler_at_eta75 ) # the higher of the 2 areas is taken for sizing of the lye cooler HX
      # calc. of lye pump power and capex
      


      L_tube = 6 # per pass
      flow_area = np.pi*D_tube**2/4.0  # per tube flow area

      N_tube_lyecooler = np.ceil(od_HX*area_lyecooler_at_eta75/(np.pi*D_tube*L_tube))  #/N_lye # Total No. of tubes per lyecool HX , total number of 2 passes
      #N_tube_anlyecooler = np.ceil(area_anlyecooler_at_eta75/(np.pi*D_tube*L_tube)) 

      N_pass = 2 # number of passes
      N_tube_pass_lyecooler = N_tube_lyecooler/N_pass # calc. number of parallel tubes per pass
      #N_tube_pass_anlyecooler = N_tube_anlyecooler/N_pass # calc. number of parallel tubes per pass

      area_lyecooler_flow_pass = N_tube_pass_lyecooler* flow_area
      #area_anlyecooler_flow_pass = N_tube_pass_anlyecooler* flow_area

      #W_pump75 = m_refr_at_eta75*(0.4)/eta_pump
                        
      vel_tubes_lyecooler = np.maximum(Q_gl_out_at_eta75,Q_angl_out_at_eta75) /area_lyecooler_flow_pass # velocity of lye in HX tubes at rated cc
      #vel_tubes_anlyecooler = (Q_angl_out) /area_anlyecooler_flow_pass

      vel_ratio = vel_tubes_lyecooler/V_target
      #vel_anratio = vel_tubes_anlyecooler/V_target
      N_tube_pass_lyecooler = N_tube_pass_lyecooler*np.maximum(vel_ratio,1)
      area_lyecooler_flow_pass = N_tube_pass_lyecooler* flow_area
      area_lyecooler_at_eta75 = (np.pi*D_tube*L_tube)*N_tube_pass_lyecooler*N_pass
      N_tube_lyecooler = N_tube_pass_lyecooler*N_pass
      val = np.floor(area_lyecooler_at_eta75 / 1100 + 0.5)

      if np.isnan(val):
          print("NaN in area_lyecooler_at_eta75, x =", x)  # x = decision vars
          raise ValueError("NaN in area_lyecooler")        # or return huge penalty

      N_lye = np.maximum(1, int(val))
      #N_lye = max(1, int(np.floor(area_lyecooler_at_eta75 / 1000 + 0.5)))# No. of lyecoolers per circuit, the no. of lyecoolers based on the max alloable HT area is 1000 m^2, velocity ratio is also equated to the N_lye
      #N_anlye = max(1, int(np.floor(area_anlyecooler_at_eta75 / 1000 + 0.9)),np.floor(vel_anratio))# No. of lyecoolers per circuit, the no. of lyecoolers based on the max alloable HT area is 1000 m^2, velocity ratio is also equated to the N_lye

      vel_tubes_lyecooler_eta75 = (np.maximum(Q_gl_out_at_eta75,Q_angl_out_at_eta75)) /area_lyecooler_flow_pass # recalculate the velocity in cathode lyecooler tubes after calc. of N_lye  , at rated operation.
      vel_tubes_lyecooler = (np.maximum(Q_gl_out,Q_angl_out)) /area_lyecooler_flow_pass
      
      #vel_tubes_anlyecooler_eta75  = (Q_anlyecool_eta75/N_lye) /area_anlyecooler_flow_pass # velocity in anode lyecooler tubes at candidate cd


       
      


      
      
      #N_lye = 2*N_lye
      #area_lyecooler_at_eta75  = area_catlyecooler_at_eta75 + area_anlyecooler_at_eta75

      CP_lyecooler = 2*N_lye * 10 ** (4.1884- 0.2503 * np.log10(max(2*area_lyecooler_at_eta75 /(2* N_lye), 2.0))+ 0.1974 * (np.log10(max(2*area_lyecooler_at_eta75 / (2*N_lye), 2.0))) ** 2)

      CP_lyecooler_min = 2*N_lye * 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

      eps = np.finfo(float).eps
      Fp_HX = (1.0 * (Pg == 0)+ (Pg > 0)* ((Pg < 5)+ (Pg >= 5)* 10 ** (0.03881- 0.11272 * np.log10(Pg + eps)+ 0.08183 * (np.log10(Pg + eps)) ** 2)))

      FBM_lyecool = 1.63 + 1.66 * 1.8 * Fp_HX
      BM_lyecool = sd * FBM_lyecool * CP_lyecooler* CI
      BM_0_lyecool = CP_lyecooler*(1.63+1.66)*CI

      if (CP_lyecooler == CP_lyecooler_min) or (area_lyecooler_at_eta75 / (N_lye) > 1100):
          BM_lyecool_min = CP_lyecooler_min * FBM_lyecool* CI
          BM_lyecool = sd * BM_lyecool_min * (area_lyecooler_at_eta75 / 2.0) ** nhe
          BM_0_lyecool = CP_lyecooler_min*(1.63+1.66)*CI
          self.xiii += 1

      # H2cooler, ads_cooler, deoxo_cooler all shell and tube

      #SEC_chiller_h2cooler_eta75 = interp_func4(cd_eta75)

      #Q_h2cool_eta75 = od_HX*1000*SEC_chiller_h2cooler_eta75* vap_h2_pdt_eta75 * MH2 * 3600 # must be in watts

      #Q_cond_ads_eta75 = od_HX*interp_func6(cd_eta75) # must be in watts

      #Q_cond_deoxo_eta75 = od_HX*interp_func7(cd_eta75) # must be in watts

      #Calculation of T_reg at eta75
      dry_gas_at_eta75 = vap_h2_pdt_eta75*(1-adspec_H2O) #dry gas flowrate at rated h2 production rate
      vap_ads_at_eta75 = vap_h2_pdt_eta75 * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber

      ads_H2O_in_at_eta75 = (dry_gas_at_eta75*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
      ads_H2O_out_at_eta75 = (vap_ads_at_eta75*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;

      ads_cap_rate_at_eta75 = ads_H2O_in_at_eta75-ads_H2O_out_at_eta75 #; % rate of moisture capture required
      qcc_at_eta75 = ads_cap_rate_at_eta75*t_b/(m_ads*eta_ads) #; % anticipated ads cc density in mol/kg
      ads_cc_at_eta75 = m_ads*qcc_at_eta75*eta_ads #; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required
      ads_cap_rate_at_eta75 = ads_cc_at_eta75/t_b
      q_res_at_eta75 = q_max-qcc_at_eta75 # excess residual capacity above the reqd capture rate
      rq_at_eta75 = q_res_at_eta75 / q_max # fraction of reduction in capacity after reqd capture capacity of the adsorbent
      des_rate_at_eta75 = ads_cc_at_eta75 / t_des #; % the reqd des_rate increases with current density as more mositure is in gas stream.

      rp_at_eta75 = purge_ads/(des_rate_at_eta75) #; % ratio of purge flow rate to des rate (based on desorb time)

      xH2O_reg_out_at_eta75 = (x_H2O_h2cool_out+1.0/rp_at_eta75)/(1.0+1.0/rp_at_eta75) #; % mositure moefrac at regen exit

      p_H2O_reg_out_at_eta75 = x[1]*10**5*(x_H2O_h2cool_out+1.0/rp_at_eta75)/(1.0+1.0/rp_at_eta75)

      p_vap_reg_eff_at_eta75 = x[1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out_at_eta75-x_H2O_h2cool_out))

      b_reg_at_eta75 = ((rq_at_eta75**0.2472)/(1.0-rq_at_eta75**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff_at_eta75)

      T_reg_at_eta75 = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg_at_eta75/b_ref))**-1) # T_reg is much more sensitive to the rq (residual ads cc after required adsorption as per specifications) than the purge gas circulation rate.

      T_base_vec_eta_75 = T_glsepHXout #* np.ones(len(T_reg_at_eta75))   # replicate as in MATLAB

      thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

      m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)

      heater_ads_at_eta75 = 0.001*(purge_ads * Cp_H2_80 * (T_reg_at_eta75 - T_base_vec_eta_75) + des_rate_at_eta75 * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg_at_eta75- T_base_vec_eta_75) / t_des / eta_heater) * t_des / t_b)

      #condensate_H2O_regen_cond = (purge_ads * (1.0 - x_H2O_h2cool_out) * xH2O_reg_out / (1.0 - xH2O_reg_out)- x_H2O_h2cool_out * purge_ads)

      #cond_ads = (condensate_H2O_regen_cond*lat_heat_water+ purge_ads * Cp_H2_80 * (T_reg - T_glsepHXout)) * t_des / t_b 

      #m_cw_cond_ads = cond_ads / (Cp_cw * (T_cwr - T_cws))




      U_purif= 60# %%%%%%%%%%%%%%% LIQUID TO GAS is 60, common for these HX

      # LMTDs
      delTlm_h2cool_at_eta75 = ((T_gl_out_at_eta75 - T_cwr) - (T_amb - T_cws)) / np.log((T_gl_out_at_eta75 - T_cwr) / (T_amb - T_cws))

      delTlm_deoxo_at_eta75 = ((T_deoxo_out - T_cwr) - (T_amb - T_cws)) / np.log((T_deoxo_out - T_cwr) / (T_amb - T_cws))

      delTlm_des_cool_at_eta75 = ((T_reg_at_eta75 - T_cwr) - (T_amb - T_cws)) / np.log((T_reg_at_eta75 - T_cwr) / (T_amb - T_cws))

      # HT Areas, shell dias, cross-sectional areas etc
      area_h2cool_at_eta75      = od_HX*Q_cond_h2cooler_at_eta75 / (U_purif * delTlm_h2cool_at_eta75 * FF)
      area_deoxo_cool_at_eta75  = od_HX*Q_cond_deoxo_at_eta75 / (U_purif * delTlm_deoxo_at_eta75 * FF)
      area_des_cool_at_eta75    =  od_HX*Q_cond_ads_at_eta75 / (U_purif * delTlm_des_cool_at_eta75 * FF)

      N_tube_h2cooler = np.ceil(area_h2cool_at_eta75/(np.pi*D_tube*L_tube))
      N_tube_deoxocool = np.ceil(area_deoxo_cool_at_eta75/(np.pi*D_tube*L_tube))  
      N_tube_descool = np.ceil(area_des_cool_at_eta75/(np.pi*D_tube*L_tube))

      Ds_h2cooler = ((4*N_tube_h2cooler*(3/4)**0.5*pt**2)/np.pi)**0.5 # shell dia of h2 cooler estimate for delP calculations, based on a vornoi cell area belonging and surrounding each tube
      Ds_deoxocool = ((4*N_tube_deoxocool*(3/4)**0.5*pt**2)/np.pi)**0.5
      Ds_descool = ((4*N_tube_descool*(3/4)**0.5*pt**2)/np.pi)**0.5
      Ds_lyecool = ((4*N_tube_lyecooler*(3/4)**0.5*pt**2)/np.pi)**0.5
      
      As_h2cooler =   Ds_h2cooler*(0.4*Ds_h2cooler)*(pt-D_tube)/pt
      As_deoxocool =   Ds_deoxocool*(0.4*Ds_deoxocool)*(pt-D_tube)/pt
      As_descool =   Ds_descool*(0.4*Ds_descool)*(pt-D_tube)/pt
      As_lyecool =   Ds_lyecool*(0.4*Ds_lyecool)*(pt-D_tube)/pt

      ##### CW circ power calc.
      m_circ_lyecool_at_eta75 = (1/(2*N_lye))*Q_lyecooler_at_eta75/(Cp_cw*(T_cwr-T_cws)) # per lye cooler unit
      m_circ_h2cooler_at_eta75 = Q_cond_h2cooler_at_eta75/(Cp_cw*(T_cwr-T_cws))  
      m_circ_deoxocool_at_eta75 = Q_cond_deoxo_at_eta75/(Cp_cw*(T_cwr-T_cws))
      m_circ_descool_at_eta75 = Q_cond_ads_at_eta75/(Cp_cw*(T_cwr-T_cws))

      m_circ_lyecool = (1/(2*N_lye))*Q_lyecooler/(Cp_cw*(T_cwr-T_cws)) # per lye cooler
      m_circ_h2cooler = Q_cond_h2cooler/(Cp_cw*(T_cwr-T_cws))  
      m_circ_deoxocool = Q_cond_deoxo/(Cp_cw*(T_cwr-T_cws))
      m_circ_descool = Q_cond_ads/(Cp_cw*(T_cwr-T_cws))


      Re_lyecool_at_eta75 = (m_circ_lyecool_at_eta75/As_lyecool )*de/mu_cw
      Re_h2cooler_at_eta75 = (m_circ_h2cooler_at_eta75/As_h2cooler  )*de/mu_cw
      Re_deoxocool_at_eta75 = (m_circ_deoxocool_at_eta75/As_deoxocool )*de/mu_cw
      Re_descool_at_eta75 = (m_circ_descool_at_eta75/As_descool )*de/mu_cw
      
      Re_lyecool = (m_circ_lyecool/As_lyecool )*de/mu_cw
      Re_h2cooler = (m_circ_h2cooler/As_h2cooler  )*de/mu_cw
      Re_deoxocool = (m_circ_deoxocool/As_deoxocool )*de/mu_cw
      Re_descool = (m_circ_descool/As_descool )*de/mu_cw

      shell_delP_lyecool_at_eta75 = (L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(np.exp(0.576 - 0.19 * np.log(Re_lyecool_at_eta75))*(m_circ_lyecool_at_eta75/As_lyecool )**2/(2*9.81*rho_cw*de))*N_lye*2
      shell_delP_h2cooler_at_eta75 = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(np.exp(0.576 - 0.19 * np.log(Re_h2cooler_at_eta75))*(m_circ_h2cooler_at_eta75/As_h2cooler  )**2/(2*9.81*rho_cw*de))
      shell_delP_deoxocool_at_eta75 = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(np.exp(0.576 - 0.19 * np.log(Re_deoxocool_at_eta75))*(m_circ_deoxocool_at_eta75 /As_deoxocool )**2/(2*9.81*rho_cw*de))
      shell_delP_descool_at_eta75 = (L_tube/(0.4*Ds_descool))*Ds_descool*(np.exp(0.576 - 0.19 * np.log(Re_descool_at_eta75))*(m_circ_descool_at_eta75 /As_descool )**2/(2*9.81*rho_cw*de))

      shell_delP_lyecool = (L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(np.exp(0.576 - 0.19 * np.log(np.maximum(Re_lyecool,1e-06)))*(m_circ_lyecool/As_lyecool )**2/(2*9.81*rho_cw*de))*N_lye*2
      
      shell_delP_h2cooler = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(np.exp(0.576 - 0.19 * np.log(Re_h2cooler))*(m_circ_h2cooler/As_h2cooler  )**2/(2*9.81*rho_cw*de))
      shell_delP_deoxocool = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(np.exp(0.576 - 0.19 * np.log(Re_deoxocool))*(m_circ_deoxocool /As_deoxocool )**2/(2*9.81*rho_cw*de))
      shell_delP_descool = (L_tube/(0.4*Ds_descool))*Ds_descool*(np.exp(0.576 - 0.19 * np.log(Re_descool))*(m_circ_descool /As_descool )**2/(2*9.81*rho_cw*de))  



      delP_cw_circ_at_eta75 = (shell_delP_lyecool_at_eta75+ shell_delP_h2cooler_at_eta75 + shell_delP_deoxocool_at_eta75 + shell_delP_descool_at_eta75)*1.2
      delP_cw_circ = (shell_delP_lyecool+ shell_delP_h2cooler + shell_delP_deoxocool + shell_delP_descool)*1.2

      #Number of H2 coolers
      N_h2cool = max(1, int(np.floor(area_h2cool_at_eta75 / 3000.0 + 0.5)))


      # Cost correlations
      CP_h2cool = N_h2cool * 10 ** (4.1884- 0.2503 * np.log10(max(area_h2cool_at_eta75 / N_h2cool, 2.0))+ 0.1974 * (np.log10(max(area_h2cool_at_eta75  / N_h2cool, 2.0))) ** 2)

      CP_h2cool_min = N_h2cool * 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

      CP_deoxo_cool = 10 ** (4.1884- 0.2503 * np.log10(max(area_deoxo_cool_at_eta75, 2.0))+ 0.1974 * (np.log10(max(area_deoxo_cool_at_eta75, 2.0))) ** 2)

      CP_deoxo_min_cool = 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

      CP_des_cool = 10 ** (4.1884- 0.2503 * np.log10(max(area_des_cool_at_eta75, 2.0))+ 0.1974 * (np.log10(max(area_des_cool_at_eta75, 2.0))) ** 2)

      CP_des_cool_min = 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

      # Bounds check
      if (area_h2cool_at_eta75   / N_h2cool) > 3000 or area_des_cool_at_eta75 > 1000 or area_deoxo_cool_at_eta75 > 1000:
          raise ValueError("HX area above correlation upper bound")

      # Bare-module costs
      FBM_purif = 1.63 + 1.66 * 1.8 * Fp_HX   # SS tube, CS shell

      BM_h2cool   = sd * FBM_purif * CP_h2cool * CI
      BM_0_h2cool = sd *(1.63 + 1.66) * CP_h2cool * CI
      BM_deoxocool    = sd * FBM_purif * CP_deoxo_cool * CI
      BM_0_deoxocool = sd * (1.63 + 1.66) * CP_deoxo_cool * CI
      BM_des_cool = sd * FBM_purif * CP_des_cool * CI
      BM_0_des_cool = sd * (1.63 + 1.66)* CP_des_cool * CI

      #BM_purif   = BM_h2cool + BM_deoxo + BM_des_cool
      #BM_0_purif = (BM_h2cool + BM_deoxo + BM_des_cool) * CI / FBM_purif
     
            ############################# PUMPS (ALL CENTRRIFUGAL due to high flowrate, calculation of energy and capex based on total HX pressure drop and fittings)

      od_pump = 1.15 # 15 % overdeisgn factor for pumps
      eta_pump =0.6
      ###### lye cooler tube side delP and lye circ pump power calc.
      Re_tube = 1260*  vel_tubes_lyecooler*D_tube/0.00093 # Re number of tubes inside the HX
      ff = (0.316*Re_tube**-0.25) # friction factor for smooth tubes turbulent flow , Blasius equation
      Re_tube_at_eta75 = 1260*  vel_tubes_lyecooler_eta75*D_tube/0.00093 # Re number of tubes inside the HX
      ff_at_eta75 = (0.316*Re_tube_at_eta75**-0.25)

      delP_catlyecooler = (1+alpha_delPHX)*ff*L_tube*1260*vel_tubes_lyecooler**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
      delP_catlyect  = (1+alpha_delPlyect)*delP_catlyecooler # total pressure drop in individual circuits

      delP_catlyecooler_at_eta75 = (1+alpha_delPHX)*ff_at_eta75 *L_tube*1260*vel_tubes_lyecooler_eta75**2/(2*D_tube)*N_lye
      delP_catlyect_at_eta75 = (1+alpha_delPlyect)*delP_catlyecooler_at_eta75 
      
      delP_anlyecooler = (1+alpha_delPHX)*ff*L_tube*1260*vel_tubes_lyecooler**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
      delP_anlyect  = (1+alpha_delPlyect)*delP_anlyecooler # total pressure drop in individual circuits

      delP_anlyecooler_at_eta75 = (1+alpha_delPHX)*ff_at_eta75 *L_tube*1260*vel_tubes_lyecooler_eta75**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
      delP_anlyect_at_eta75  = (1+alpha_delPlyect)*delP_anlyecooler_at_eta75 # total pressure drop in individual circuits  

      W_catlyepump =  0.001*(delP_catlyect + cell_delP)*Q_gl_out /eta_pump 

      W_anlyepump_eta75 = 0.001*(delP_anlyect_at_eta75 + ancell_delP_at_eta75)*Q_angl_out_at_eta75 /eta_pump # in kW

      W_anlyepump =   0.001*(delP_anlyect + ancell_delP)*Q_angl_out /eta_pump 

      W_catlyepump_eta75 = 0.001*(delP_catlyect_at_eta75 + cell_delP_at_eta75)*Q_gl_out_at_eta75 /eta_pump # in kW

      W_lyepump75 = np.maximum(W_anlyepump_eta75 , W_catlyepump_eta75) # maximum value is selected for lye pump sizing, in kW

      W_cwpump75 = 0.001*delP_cw_circ_at_eta75*(m_cw_circ_at_eta75/rho_cw)/eta_pump    # in kW  

      W_cwpump = 0.001*delP_cw_circ*(m_cw_circ/rho_cw)/eta_pump

      W_pump =  (W_anlyepump + W_catlyepump+ W_cwpump)# total pumping power in kW

      W_pump75 = (W_anlyepump_eta75 + W_catlyepump_eta75 + W_cwpump75)
      #W_pump75 = od_pump*SEC_pump_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 # unit is kW

      if W_lyepump75 > 0:
      # Pump cost
        CP_pump_lye = 2*10 ** (3.3892
                     + 0.0536 * np.log10(max(od_pump*W_lyepump75, 0.05))
                     + 0.1538 * (np.log10(max(od_pump*W_lyepump75, 0.05)))**2) # factor of 2 because of 2 identical lye pumps

        CP_pump_lye_min = 2*10 ** (3.3892
                         + 0.0536 * np.log10(0.05)
                         + 0.1538 * (np.log10(0.05))**2)   # 1 kW

        FBM_pump_lye = 1.89 + 1.35 * (np.mean([2.3, 1.4]) *
                              ((x[1] < 11) * 1 + (x[1] > 11) * 1.157))

        BM_lyepumps = sd * CP_pump_lye * FBM_pump_lye * CI
        BM_0_lyepumps = CP_pump_lye*(1.89+1.35)* CI
      

      # apply 6/10 rule if min. cc > calc. cc
        if (CP_pump_lye == CP_pump_lye_min) or (od_pump*W_lyepump75 > 500):
          BM_lyepumps_min = CP_pump_lye_min * FBM_pump_lye* CI
          BM_lyepumps = sd * BM_lyepumps_min * (od_pump*W_lyepump75 / 0.05) ** 0.6
          BM_0_lyepumps = CP_pump_lye_min*(1.89+1.35)* CI
          #xa += 1
      else:
         BM_lyepumps = BM_0_lyepumps =0

      if W_cwpump75 > 0:
      # Pump cost
        CP_cwpump = 10 ** (3.3892
                     + 0.0536 * np.log10(max(od_pump*W_cwpump75, 0.05))
                     + 0.1538 * (np.log10(max(od_pump*W_cwpump75, 0.05)))**2) # factor of 2 because of 2 identical lye pumps

        CP_cwpump_min = 10 ** (3.3892
                         + 0.0536 * np.log10(0.05)
                         + 0.1538 * (np.log10(0.05))**2)   # 1 kW

        FBM_cwpump = 1.89 + 1.35 * (np.mean([2.3, 1.4]) *
                              ((x[1] < 11) * 1 + (x[1] > 11) * 1.157))

        BM_cwpump = sd * CP_cwpump * FBM_cwpump * CI
        BM_0_cwpumps = CP_cwpump*(1.89+1.35)* CI
      

      # apply 6/10 rule if min. cc > calc. cc
        if (CP_cwpump == CP_cwpump_min) or (od_pump*W_cwpump75> 500):
          BM_cwpump_min = CP_cwpump_min * FBM_cwpump* CI
          BM_cwpump = sd * BM_cwpump_min * (od_pump*W_cwpump75 / 0.05) ** 0.6
          BM_0_cwpumps = CP_cwpump_min*(1.89+1.35)* CI
          #xa += 1
      else:
         BM_cwpump = BM_0_cwpumps =0  
         BM_0_pumps = BM_0_cwpumps + BM_0_lyepumps
      ################################################################################### calc. of Wsys and SEC_sys ###################################################################################################

      # ===== INITIALIZE W_sys75 =====
      W_sys75 = 0.0

            
            # ===== COMPUTE W_sys75 HERE (inside valid condition) =====


      #W_sys75 = SEC_sys_at_eta75 * vap_h2_pdt_eta75 * MH2 * 3600 # rated input power to system, 'rated' therefore no consdieration of degradation in the calculation
      #W_sys_peak = SEC_sys_at_peak * vap_h2_pdt_peak * MH2 * 3600 # peak allowable input power to system, 10% above the rated curr dens.

      W_sys75 = W_pump75 + SEC_stack_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 + (heater_ads_at_eta75 + heater_deoxo_at_eta75 + heater_regen + Q_lyeheater_at_eta75)/eta_heater +W_h2comp +W_refrcomp_at_eta75 +W_refrfan_at_eta75 +W_compfan
      W_sys75 = W_sys75/eta_pow
      W_sys = W_pump + SEC_stack* vap_h2_pdt * MH2 * 3600 + (heater_ads + heater_deoxo + heater_regen + Q_lyeheater)/eta_heater + W_refrcomp_SEC + W_refrfan_SEC + W_comp_SEC + W_fan_SEC
      W_sys = W_sys/eta_pow  
      W_sys_exstack = W_pump + (heater_ads + heater_deoxo + heater_regen + Q_lyeheater)/eta_heater + W_refrcomp_SEC + W_refrfan_SEC + W_comp_SEC + W_fan_SEC # system power draw ex of stack, ie only BOP
            # ========================================================  

      Pow_lyeheater =  heater_frac_ratedpower*W_sys75*1000 # rated lye heater power is heater_frac_ratedpower% of the W_sys75 in Watt
      P_rated_el   = eta_heater * Pow_lyeheater # this is the power actually transferred to the heat of the lye
      BM_heaters = 7.5 * 3*W_sys75      # 7.5 $/kW, %7.5$/kW , 3 heaters in B.O.P,Grid connected Hydrogen production via large-scale water electrolysis{Nguyen, 2019 #186}, W_sys75 is in kW, od = 1.15
      BM_heaters_lye = Pow_lyeheater*0.001*150*(800/619) # heaters have to be sized higher when the AWE is totally reliant on renewable power compared to when a battery is available, so as  to maximize the full load hours ( only when renew power is used), and increase load factor. $150/kW
       

      BM_pow = 199*W_sys75            # 199 $/kW, power electronics includes buck converter, transfomer, DC to DC converter, 1.15 over design factor
      

      
      
   
      

      #------ calc. of t_oper based from actual PV+wind hybrid generation profiles-------
      
      #f_min = 0.3; #10% load factor
      min_load = self.cd_eta75_base_ratio*cd_eta75
      

      #SEC_sys_min_load = interp_SEC_sys(min_load)
      #SEC_sys_min_load1 = interp_func1(min_load1) # system SEC at min. load
      vap_h2_pdt_minload = interp_vap_h2_pdt(min_load)

      ################ calc. of W_sys at min. load ####################
      T_gl_out_minload = interp_T_gl_out(min_load)  
      T_angl_out_minload = interp_T_angl_out(min_load)
      w_KOH_gl_out_minload = interp_w_KOH_gl_out(min_load)
      w_KOH_angl_out_minload = interp_w_KOH_angl_out(min_load)

      Cp_KOH_lye_minload = (4.101*10**3-3.526*10**3*w_KOH_gl_out_minload  + 9.644*10**-1*(T_gl_out_minload-273.15)+1.776*(T_gl_out_minload-273.15)*w_KOH_gl_out_minload)
      Cp_KOH_anlye_minload = (4.101*10**3-3.526*10**3*w_KOH_angl_out_minload + 9.644*10**-1*(T_angl_out_minload-273.15)+1.776*(T_angl_out_minload-273.15)*w_KOH_angl_out_minload)
      Q_gl_out_minload      = interp_Q_gl_out(min_load)
      Q_angl_out_minload      = interp_Q_angl_out(min_load)
      vel_tubes_lyecooler_minload  = (np.maximum(Q_gl_out_minload,Q_angl_out_minload)/N_lye)/area_lyecooler_flow_pass # use maximum of the flowrate
      rho_KOH_gl_out_minload = (1001.53 - 0.08343*(T_gl_out_minload - 273.15) - 0.004*(T_gl_out_minload-273.15)**2 + 5.51232*10**-6*(T_gl_out_minload-273.15)**3 - 8.21*10**-10*(T_gl_out_minload-273.15)**4)*np.exp(0.86*w_KOH_gl_out_minload)
      rho_KOH_angl_out_minload = (1001.53 - 0.08343*(T_angl_out_minload - 273.15) - 0.004*(T_angl_out_minload-273.15)**2 + 5.51232*10**-6*(T_angl_out_minload-273.15)**3 - 8.21*10**-10*(T_angl_out_minload-273.15)**4)*np.exp(0.86*w_KOH_angl_out_minload)

      Re_tube_minload = np.maximum(rho_KOH_gl_out_minload,rho_KOH_angl_out_minload)*  vel_tubes_lyecooler_minload*D_tube/0.00093 # Re number of tubes inside the HX
      ff_minload = (0.316*Re_tube_minload**-0.25) # friction factor for smooth tubes turbulent flow , Blasius equation

      delP_catlyecooler_minload = (1+alpha_delPHX)*ff_minload*L_tube*rho_KOH_gl_out_minload*vel_tubes_lyecooler_minload**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
      delP_catlyect_minload  = (1+alpha_delPlyect)*delP_catlyecooler_minload # total pressure drop in individual circuits
      
      delP_anlyecooler_minload = (1+alpha_delPHX)*ff_minload*L_tube*rho_KOH_angl_out_minload*vel_tubes_lyecooler_minload**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
      delP_anlyect_minload  = (1+alpha_delPlyect)*delP_anlyecooler_minload # total pressure drop in individual circuits

      cell_delP_minload = interp_cell_delP(min_load)
      ancell_delP_minload = interp_ancell_delP(min_load)

      W_catlyepump_minload =  0.001*(delP_catlyect_minload + cell_delP_minload)*Q_gl_out_minload /eta_pump 
      W_anlyepump_minload =   0.001*(delP_anlyect_minload + ancell_delP_minload)*Q_angl_out_minload /eta_pump 



      Q_lyecooler_minload = np.maximum((T_gl_out_minload-273.15-80)*Cp_KOH_lye_minload*Q_gl_out_minload*rho_KOH_gl_out_minload,0)  + np.maximum((T_angl_out_minload-273.15-80)*Cp_KOH_anlye_minload*Q_angl_out_minload*rho_KOH_angl_out_minload,0) # lye cooler duty at candidate load
      Q_cond_h2cooler_minload = interp_Q_cond_h2cooler(min_load)
      Q_cond_deoxo_minload = interp_Q_cond_deoxo(min_load)
      Q_cond_ads_minload = interp_Q_cond_ads(min_load)

      m_circ_lyecool_minload = 0.5*Q_lyecooler_minload/(Cp_cw*(T_cwr-T_cws)) # per lye cooler
      m_circ_h2cooler_minload = Q_cond_h2cooler_minload/(Cp_cw*(T_cwr-T_cws))  
      m_circ_deoxocool_minload = Q_cond_deoxo_minload/(Cp_cw*(T_cwr-T_cws))
      m_circ_descool_minload = Q_cond_ads_minload/(Cp_cw*(T_cwr-T_cws))      
      Re_lyecool_minload = (m_circ_lyecool_minload/As_lyecool )*de/mu_cw
      Re_h2cooler_minload = (m_circ_h2cooler_minload/As_h2cooler  )*de/mu_cw
      Re_deoxocool_minload = (m_circ_deoxocool_minload/As_deoxocool )*de/mu_cw
      Re_descool_minload = (m_circ_descool_minload/As_descool )*de/mu_cw

      #shell_delP_lyecool_minload = (L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(np.exp(0.576 - 0.19 * np.log(np.maximum(Re_lyecool_minload,1e-06)))*(m_circ_lyecool_minload/As_lyecool )**2/(2*9.81*rho_cw*de))
      #shell_delP_h2cooler_minload = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(np.exp(0.576 - 0.19 * np.log(Re_h2cooler_minload))*(m_circ_h2cooler_minload/As_h2cooler  )**2/(2*9.81*rho_cw*de))
      #shell_delP_deoxocool_minload = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(np.exp(0.576 - 0.19 * np.log(Re_deoxocool_minload))*(m_circ_deoxocool_minload /As_deoxocool )**2/(2*9.81*rho_cw*de))
      #shell_delP_descool_minload = (L_tube/(0.4*Ds_descool))*Ds_descool*(np.exp(0.576 - 0.19 * np.log(Re_descool_minload))*(m_circ_descool_minload /As_descool )**2/(2*9.81*rho_cw*de))  

      #delP_cw_circ_minload = (shell_delP_lyecool_minload + shell_delP_h2cooler_minload + shell_delP_deoxocool_minload + shell_delP_descool_minload)*1.2

      #W_catlyepump_eta75 = 0.001*(delP_catlyect_at_eta75 + cell_delP_at_eta75)*Q_gl_out_at_eta75 /eta_pump # in kW


      #W_cwpump_minload = 0.001*delP_cw_circ_minload*(m_cw_circ/rho_cw)/eta_pump

      W_pump_minload =  (W_anlyepump_minload + W_catlyepump_minload)# total pumping power in kW
      #Calculation of T_reg at eta75
      dry_gas_at_minload = vap_h2_pdt_minload*(1-adspec_H2O) #dry gas flowrate at rated h2 production rate
      vap_ads_at_minload = vap_h2_pdt_minload * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber
      ads_H2O_in_at_minload = (dry_gas_at_minload*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
      ads_H2O_out_at_minload = (vap_ads_at_minload*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;
      ads_cap_rate_at_minload = ads_H2O_in_at_minload-ads_H2O_out_at_minload #; % rate of moisture capture required
      qcc_at_minload = ads_cap_rate_at_minload*t_b/(m_ads*eta_ads) #; % anticipated ads cc density in mol/kg
      ads_cc_at_minload = m_ads*qcc_at_minload*eta_ads #; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required
      ads_cap_rate_at_minload = ads_cc_at_minload/t_b
      q_res_at_minload = q_max-qcc_at_minload # excess residual capacity above the reqd capture rate
      rq_at_minload = q_res_at_minload / q_max # fraction of reduction in capacity after reqd capture capacity of the adsorbent
      des_rate_at_minload = ads_cc_at_minload / t_des #; % the reqd des_rate increases with current density as more mositure is in gas stream.
      rp_at_minload = purge_ads/(des_rate_at_minload) #; % ratio of purge flow rate to des rate (based on desorb time)
      xH2O_reg_out_at_minload = (x_H2O_h2cool_out+1.0/rp_at_minload)/(1.0+1.0/rp_at_minload) #; % mositure moefrac at regen exit
      p_H2O_reg_out_at_minload = x[1]*10**5*(x_H2O_h2cool_out+1.0/rp_at_minload)/(1.0+1.0/rp_at_minload)
      p_vap_reg_eff_at_minload = x[1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out_at_minload-x_H2O_h2cool_out))
      b_reg_at_minload = ((rq_at_minload**0.2472)/(1.0-rq_at_minload**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff_at_minload)
      T_reg_at_minload = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg_at_minload/b_ref))**-1) # T_reg is much more sensitive to the rq (residual ads cc after required adsorption as per specifications) than the purge gas circulation rate.
      T_base_vec_minload = T_glsepHXout #* np.ones(len(T_reg_at_eta75))   # replicate as in MATLAB
      thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness
      m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)
      heater_ads_minload = 0.001*(purge_ads * Cp_H2_80 * (T_reg_at_minload - T_base_vec_minload) + des_rate_at_minload * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg_at_minload- T_base_vec_minload) / t_des / eta_heater) * t_des / t_b)
      glsep_O2_minload = interp_glsep_O2(min_load)

      deoxo_H2O_minload     = 2.0 * glsep_O2_minload
      deoxo_H2reac_minload = 2.0 * glsep_O2_minload
      deoxo_heat_minload   = 2.0 * glsep_O2_minload * 244.9 * 10.0**3   # 244.9 kJ/mol → J/mol

      T_deoxo_minload = ( deoxo_heat_minload
      + therm_cat_deoxo * (T_glsepHXout + 125.0)
      + therm_st_deoxo * (T_glsepHXout + 125.0)
      + vap_h2_pdt_minload * Cp_vap_h2cool * T_glsepHXout) / (vap_h2_pdt_minload * Cp_vap_h2cool + therm_st_deoxo + therm_cat_deoxo)

      deoxo_T = 150 +273.15 # deoxo maintains temp. at 150C
      heater_deoxo_minload = 0.001*((deoxo_T > T_deoxo_minload).astype(float)* vap_h2_pdt_minload* Cp_vap_h2cool* (deoxo_T - T_deoxo_minload)/ eta_heater)
      Q_lyeheater_minload = 0.001*((np.maximum((273.15+80-T_gl_out_minload)*Cp_KOH_lye_minload*Q_gl_out_minload*rho_KOH_gl_out_minload,0)  + np.maximum((273.15+80-T_angl_out_minload)*Cp_KOH_anlye_minload*Q_angl_out_minload*rho_KOH_angl_out_minload,0)))

      W_comp_minload = 0.001*((max(Ncomp , 0) * k_isen / (k_isen - 1.0))* R_const * T_suc_comp * Z_comp * (rc ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt_minload / (eta_isen_comp * eta_mech_motor * eta_mech_comp)) #+ (P_final / x[1] > 1.0)* (1.0 * k_isen / (k_isen - 1.0) * R_const * T_amb * Z_comp* (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt_minload / (eta_isen_comp * eta_mech_motor * eta_mech_comp)))  # [web:1], in kW
      intercool_comp_minload = max(Ncomp, 0) * (T_dis_comp - T_suc_comp) * vap_h2_pdt_minload * Cp_H2_80 #+ 1.0 * (T_max_comp - T_amb) * vap_h2_pdt_minload * Cp_H2_80)  # [web:1]

      air_fan_minload = intercool_comp_minload / (delT_air * Cp_air * rho_air)  # [web:1]

      W_fan_minload = 0.001*air_fan_minload * delP_fan / eta_fan  # [web:1]

      m_cw_circ_minload =   (Q_lyecooler_minload + Q_cond_ads_minload + Q_cond_deoxo_minload + Q_cond_h2cooler_minload)/(Cp_cw*(T_cwr-T_cws))

      m_refr_minload = m_cw_circ_minload*Cp_cw*(T_cwr-T_cws)/refr_lat_evap    #; % total refireigenrant circulation rate inside the chiller
      W_refrcomp_minload = 0.001*(delH_comp*m_refr_minload/(eta_mech_comp*eta_mech_motor))
      W_refrfan_minload = 0.001*((m_refr_minload*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan

      SEC_stack_minload = interp_SEC_stack(min_load)

      P_min = 1000*(W_pump_minload + SEC_stack_minload* vap_h2_pdt_minload * MH2 * 3600 + (heater_ads_minload + heater_deoxo_minload + heater_regen + Q_lyeheater_minload)/eta_heater  + W_refrcomp_minload + W_refrfan_minload + W_comp_minload + W_fan_minload)/eta_pow # in Watts

      if P_min > W_sys*1000 :
        
         g2 = 1e6
      #in Watts, system power at min load
      #vap_h2_pdt_standby_load = interp_vap_h2_pdt(minIn_SEC[-1])
      
      

      #P_min_standby = interp_func14(minIn_SEC[-1])*vap_h2_pdt_standby_load * MH2 * 3600*1000 +  interp_func12(minIn_SEC[-1])*vap_h2_pdt_standby_load * MH2 * 3600*1000 # min. power for hot standby
      T_gl_out_standby = interp_T_gl_out(minIn_SEC[-1]) 
      T_angl_out_standby = interp_T_angl_out(minIn_SEC[-1]) 
      Q_angl_out_standby = interp_Q_angl_out(minIn_SEC[-1]) 
      Q_gl_out_standby = interp_Q_gl_out(minIn_SEC[-1]) 
      w_KOH_gl_out_standby = interp_w_KOH_gl_out(minIn_SEC[-1])
      w_KOH_angl_out_standby = interp_w_KOH_angl_out(minIn_SEC[-1])
      Cp_KOH_lye_standby =  (4.101*10**3-3.526*10**3*w_KOH_gl_out_standby +9.644*10**-1*(T_gl_out_standby-273)+1.776*(T_gl_out_standby-273)*w_KOH_gl_out_standby)
      Cp_KOH_anlye_standby =  (4.101*10**3-3.526*10**3*w_KOH_angl_out_standby +9.644*10**-1*(T_angl_out_standby-273)+1.776*(T_angl_out_standby-273)*w_KOH_angl_out_standby)
      rho_KOH_gl_out_standby = (1001.53 -0.08343*(T_gl_out_standby-273.15) -0.004*(T_gl_out_standby-273.15)**2 + 5.51232*10**-6*(T_gl_out_standby-273.15)**3 - 8.21*10**-10*(T_gl_out_standby-273.15)**4)*np.exp(0.86*w_KOH_gl_out_standby)
      rho_KOH_angl_out_standby = (1001.53 -0.08343*(T_angl_out_standby-273.15) -0.004*(T_angl_out_standby-273.15)**2 + 5.51232*10**-6*(T_angl_out_standby-273.15)**3 - 8.21*10**-10*(T_angl_out_standby-273.15)**4)*np.exp(0.86*w_KOH_angl_out_standby)

      vel_tubes_lyecooler_standby  = (np.maximum(Q_gl_out_standby  ,Q_angl_out_standby)/N_lye)/area_lyecooler_flow_pass # use maximum of the flowrate
      Re_tube_standby = 1260*  vel_tubes_lyecooler_standby*D_tube/0.00093 # Re number of tubes inside the HX
      ff_standby = (0.316*Re_tube_standby**-0.25) # friction factor for smooth tubes turbulent flow , Blasius equation
      delP_catlyecooler_standby = (1+alpha_delPHX)*ff_standby*L_tube*1260*vel_tubes_lyecooler_standby**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
      delP_catlyect_standby = (1+alpha_delPlyect)*delP_catlyecooler_standby # total pressure drop in individual circuits
      delP_anlyecooler_standby = (1+alpha_delPHX)*ff_minload*L_tube*1260*vel_tubes_lyecooler_standby**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.

      delP_anlyect_standby  = (1+alpha_delPlyect)*delP_anlyecooler_standby # total pressure drop in individual circuits
      cell_delP_standby = interp_cell_delP(minIn_SEC[-1])
      ancell_delP_standby= interp_ancell_delP(minIn_SEC[-1])

      W_catlyepump_standby =  0.001*(delP_catlyect_standby + cell_delP_standby)*Q_gl_out_standby /eta_pump 
      W_anlyepump_standby =   0.001*(delP_anlyect_standby + ancell_delP_standby)*Q_angl_out_standby /eta_pump 
    
      Pump_lye_standby = (W_anlyepump_minload + W_catlyepump_minload) # only the lye circ pump power is needed for standby
      P_controls = 1000 # control power input to plant in Watts

      P_min_standby = 1000*Pump_lye_standby/eta_pow + P_controls/eta_pow  # (273.15+80-T_gl_out_standby)*Cp_KOH*x[0]*x[2]*0.008*Nc*gc  + (273.15+80-T_angl_out_standby)*Cp_KOH*x[0]*x[2]*0.008*Nc*gc #interp_func12(minIn_SEC[-1])*vap_h2_pdt_standby_load * MH2 * 3600*1000 # min. renew power for hot standby

      Heater_lye_standby = (273.15+80-T_gl_out_standby)*Cp_KOH_lye_standby*Q_gl_out_standby*rho_KOH_gl_out_standby  + (273.15+80-T_angl_out_standby)*Cp_KOH_anlye_standby*Q_angl_out_standby*rho_KOH_angl_out_standby # min. heater duty to maintain temp at set point during warm standby, currently not part of min. standby power

      #P_min1 = SEC_sys_min_load1 * vap_h2_pdt_min_load1 * MH2 * 3600*1000
      #awebop_power = np.where(pv_powerout >= P_min, np.minimum(pv_powerout, W_sys75*1000), 0)
    
      global pv_powerout, wind_powerout
      
      #genratio = self.genratio
      pv_powerout_scaled = pv_powerout* genratio/2 * (W_sys75)
      wind_powerout_scaled = wind_powerout * genratio/2 * (W_sys75)*(1/1504.5) # to get the correct mean annual energy produced including wind effects


      hybrid_powerout = wind_powerout_scaled+pv_powerout_scaled
      P_min_vec = np.full(len(hybrid_powerout), P_min)
      below_min = hybrid_powerout < P_min
#      below_min1 = hybrid_powerout < P_min1
      n_true = np.sum(below_min)
      
      # Hot standby: P_min_standby <= P < P_min
      prod_mask = hybrid_powerout >= P_min
      standby_mask = (hybrid_powerout >= P_min_standby) & (hybrid_powerout < P_min) # standby mode active 
      # Idle: P < P_min_standby
      idle_mask = hybrid_powerout  < P_min_standby # idle mode active
      # --- Time in each mode (hours) ---------------------------------------------
      # A start occurs when we were NOT in production at t-1, but ARE in production at t

      
      ## Thermal RC model for calculation of temp. decay rate in idle and AWE ramp rate to set point temperature (80C)... all rho and Cp values calculated at 80C
      #  Calc. of glsep thickness and material mass
      od_glsep = 1.15
      #Q_gl_out = x[2]*0.008*x[0]*0.001*Nc*gc # in m^3/s
      Pg = x[1]-1 # guage pressure

      vol_liq_catglsep_standby = 120 * Q_gl_out_standby 
      vol_liq_anglsep_standby =  120 * Q_angl_out_standby  # m^3, 2 mins residence time, liquid volume inside the separator (sized at rated liq volume resid time), at standby conditions
      #vol_glsep     = od_glsep * 120 * Q_gl_out_at_eta75 *2                       # overdesigned volume, 50% fill level
     # glsep_D       = ((4 * vol_glsep) / (3 * np.pi)) ** (1.0 / 3.0)   # diameter [m]
     # glsep_L       = 3 * glsep_D                                      # length [m]

      # Pressure vessel thickness
      thick_glsep = max(((Pg + 1) * glsep_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

      vol_glsep_steel = np.pi* (glsep_D+thick_glsep)**2/4.0*glsep_L -  np.pi* (glsep_D)**2/4.0*glsep_L
      mass_glsep = 2*vol_glsep_steel* rho_steel # rho_steel is the density of the carbon steel (MOC og glsep), mult by 2 for 2 pieces of glsep
      rho_KOH = (1001.53 -0.08343*(80) -0.004*(80)**2 + 5.51232*10**-6*(80)**3 - 8.21*10**-10*(80)**4)*np.exp(0.86*.302) # density of KOH at nominal conditions ie 80C, and 30.2 wt%
      Cp_KOH = (4.101*10**3-3.526*10**3*0.302 +9.644*10**-1*(80)+1.776*(80)*0.302)

      mass_KOH  = vol_liq_catglsep_standby*rho_KOH_gl_out_standby + vol_liq_anglsep_standby*rho_KOH_angl_out_standby  + (gc*2*(0.05*0.008*x[0]*0.001*(Nc))*0.85 + gc*0.05*0.008*Nc*x[4]*0.001*0.57)*rho_KOH # rho_KOH is the KOH density at 80C, 30.2 wt%, 85 % porosity of electrode is assumed.
      therm_KOH = vol_liq_catglsep_standby*rho_KOH_gl_out_standby*Cp_KOH_lye_standby + vol_liq_anglsep_standby*rho_KOH_angl_out_standby*Cp_KOH_anlye_standby + (gc*2*(0.05*0.008*x[0]*0.001*(Nc))*0.85 + gc*0.05*0.008*Nc*x[4]*0.001*0.57)*rho_KOH*Cp_KOH # therm cc of KOH, J/kg

      mass_stack_SS = (1+0.2*(x[1]-1)/(15-1))*steel/4.5   # consists of SS316L
      mass_stack_Ni =  (Ni/Ni_foam)*rho_Ni # consists of Ni 
      heat_cc_elec = (mass_stack_Ni*Cp_Ni +  (mass_stack_SS + mass_glsep)*Cp_steel + therm_KOH)*1.2 # factor of 1.2 to account for heat cc of piping, valves, HX, pumps in the electrolyte circuit 
      
      Area_stack = 2*gc*(0.05*0.003*2+0.05*0.001*(Nc-1) + 2*(0.05*x[0]*0.001*(Nc))+ (0.05*x[4]*0.001*(Nc)) + 0.008*0.005 + 0.008*(2*(x[0]*0.001*(Nc))+ (x[4]*0.001*(Nc)))) # total stack exposed surface area

      Area_glsep = 2*(np.pi*(glsep_D+thick_glsep)*glsep_L + 2*np.pi*(glsep_D+thick_glsep)**2/4.0)   # total glsep exposed area assuming cylindrical shape
      Tot_Area = (Area_stack + Area_glsep)*1.3 # 1.3 factor to account for extra area from piping, valves HX etc.
      htc_elec = 5 # taken as const, in W/m^2*K
      therm_tau = heat_cc_elec/(Tot_Area*htc_elec)/3600 # thermal time constant in hours
      alpha = np.exp(-dt / therm_tau)   # dimensionless temp decay factor
      
      
      T_sp = 353.15 # set point at 80C
      T = T_sp        # assume hot at start, or set to T_amb
      T_hist = np.zeros_like(hybrid_powerout, dtype=float)
      hA = Tot_Area * htc_elec
      heater_rated_limited = np.zeros_like(hybrid_powerout, dtype=bool)
      
      # hours where heater is limiting AND still heating up


      for kk, P_ren in enumerate(hybrid_powerout):
       
        P_from_renew = eta_pow * eta_heater * np.maximum((P_ren - P_min_standby),0) # power from renewables available for heating the lye
        P_rated_el   = eta_heater * Pow_lyeheater # actual heater power available for heating the lye after accounting for heating inefficiencies

        if P_rated_el <= P_from_renew:
        # heater is at its rating; renewables could give more
          P_heater_el = P_rated_el
          heater_rated_limited[kk] = True
        else:
        # renewables are limiting
          P_heater_el = P_from_renew
          heater_rated_limited[kk] = False

        Q_heater = P_heater_el   
        #P_heater = np.minimum(eta_heater*Pow_lyeheater,eta_pow*eta_heater*P_ren)
        T_ss_heat = T_amb + Q_heater / hA
        if P_ren < P_min_standby:
        # idle cooling
          T = T_amb + (T - T_amb) * alpha

        elif (P_ren >= P_min_standby) and (P_ren < P_min):
        # heating only, no production
          T = T_ss_heat + (T - T_ss_heat) * alpha
          if T > T_sp: T = T_sp

        else:  # P >= P_min
          if T < T_sp:
            # still heating to setpoint (no production until T>=T_sp)
            T = T_ss_heat + (T - T_ss_heat) * alpha
            if T > T_sp: T = T_sp
          else:
            T = T_sp   # operate at setpoint
            
        T_hist[kk] = T

      below_Tsp        = T_hist < 353.15 
      heater_at_rating = heater_rated_limited 
      mask_heating_limited = heater_at_rating & below_Tsp
      hours_heating_limited = mask_heating_limited.sum() * dt 
      frac_heating_limited = hours_heating_limited/8760      

      startup_mask = (prod_mask) & (T_hist < T_sp)

      prod_mask_eff   = prod_mask & ~startup_mask   # truly productive
      standby_mask_eff = standby_mask | startup_mask

      prod_shift = np.roll(prod_mask_eff, 1)
      prod_shift[0] = False  # no start at first element by definition
      starts_mask = (~prod_shift) & prod_mask_eff
      n_starts = np.sum(starts_mask)

   
      hours_prod    = np.sum(prod_mask_eff)    * dt
      hours_standby = np.sum(standby_mask_eff) * dt

      hours_idle    = np.sum(idle_mask)    * dt
      startup_hours = startup_mask.sum() * dt

      E_standby = 0.001*np.sum(standby_mask_eff * P_min_standby * dt)  # kWh per year








      # Identify, for each start, whether the system was in standby or idle at t-1
      #prev_standby = np.roll(standby_mask, 1); prev_standby[0] = False
      #prev_idle    = np.roll(idle_mask, 1);    prev_idle[0]    = False
      

      #prod_shift = np.roll(prod_mask, 1)
      #prod_shift[0] = False  # no start at first element by definition

      #starts_mask = prod_mask & (~prod_shift)
      #n_starts = np.sum(starts_mask)

      #startup_hours_loss = (n_starts_standby * startup_time_from_standby +
      #                     n_starts_idle * startup_time_from_idle)
      #startup_mask = (prod_mask) & (T_hist < T_sp)
      #startup_hours = startup_mask.sum() * dt

      #E_standby = 0.001*np.sum(standby_mask_eff * P_min_standby * dt)  # kWh per year
#      starts1 = np.sum( (below_min1[:-1] == True) & (below_min1[1:] == False) )
      # startup time per hot start (h)
      # 2) Dips start where we go from ON (False) to OFF (True)
      dip_start_mask = (~below_min[:-1]) & (below_min[1:])   # ON -> OFF
      dip_stop_mask  = (below_min[:-1]) & (~below_min[1:])   # OFF -> ON
      
      dip_start_idx = np.where(dip_start_mask)[0] + 1
      dip_stop_idx  = np.where(dip_stop_mask)[0] + 1
      L = dip_stop_idx[1:] - dip_start_idx[:-1]
      short_cycles = np.sum(L == 1)       # 1 h dips
      long_cycles  = np.sum(L >  1)       # >1 h dips
      
      #start_mask = (below_min[:-1] == True) & (below_min[1:] == False)
#      start_mask1 = (below_min1[:-1] == True) & (below_min1[1:] == False)
      # Number of starts (scalar)
      #n_starts = np.sum(start_mask)
     # n_starts1 = np.sum(start_mask1)
      # total startup-lost hours per year
      #startup_lost_hours = starts * t_start
      # effective operating hours at/above min load (if you want it explicitly)
      hours_above_min = np.sum(~below_min)    # number of time steps with operation
      #effective_operating_hours = hours_above_min - startup_lost_hours
      #start_indices = np.where(start_mask)[0] + 1
#     start_indices1 = np.where(start_mask1)[0] + 1  
      P_clip = np.minimum(hybrid_powerout, W_sys * 1000)
      awebop_power = np.where(P_clip >= P_min, P_clip, 1e-6) # the W_sys is used here, as the further calc. load _factor is used find the annual h2 production, so the peak power is not used
      awebop_power_curt = np.where(P_clip >= P_min_standby, P_clip, 1e-6)
#      awebop_power1 = np.where(hybrid_powerout >= P_min1, np.minimum(hybrid_powerout, W_sys*1000), 0) # the used power of the awebop is calcuated based on the peak current density limit of the electrolyzer which is 10% above the rated current density
      #start_indices = np.where(starts)[0] + 1   # hour just after each start
      #P_first = awebop_power[start_indices]     # kW in that first hour
#      P_first1 = awebop_power[start_indices1]
 # kWh lost in startups
#     E_lost1 = np.sum(P_first1 * t_start)  
      P_lost       = awebop_power* startup_mask
      E_lost       = P_lost.sum() * dt
      total_energy_supplied = np.sum(awebop_power) # total energy supplied to AWE within its power band , wihtout considering lost prod hrs due to slow startup
      total_energy_supplied_curt = np.sum(awebop_power_curt) # total power supplied within the states (production + hot standby), used for calc. of curtailment loss
        # If array in kW, yields kWh for the year, is the total energy supplied by the pow gen within the load range of the awe electrolyzer
#      total_energy_supplied1 = np.sum(awebop_power1)  # If array in kW, yields kWh for the year, is the total energy supplied by the pow gen within the load range of the awe electrolyzer  
      E_effective = total_energy_supplied - E_lost # Efective energy from power produced utilized in actual faradaic h2 production
      total_energy_supplied = np.sum(E_effective)
      heater_mask = startup_mask | standby_mask_eff # actual energy supplied which is used for h2 production
      heater_start_pow = (np.minimum(Pow_lyeheater,np.maximum(awebop_power_curt-P_min_standby,0)))
      pump_start_pow =1000*Pump_lye_standby + 1000 # 2nd term for control unit power (1kW, fixed)
      theoretical_max = W_sys75*1000 * len(pv_powerout)   # MAx. theoretical load is the rated load, and the LF is calcuated is based on the rated power of the AWE
      #print(W_sys75)
      frac_stby_ener = (np.sum(heater_start_pow*heater_mask*0.001*dt)+np.sum(pump_start_pow*standby_mask_eff*0.001*dt))/(total_energy_supplied_curt*0.001)
      plant_availability = 0.97
       # this is the total energy utilized for production
      load_factor = E_effective / theoretical_max*100*plant_availability
#      load_factor1 = E_effective1 / theoretical_max*100*plant_availability
      #======== calculation of stack degradation, SEC_avg of platn life and replacement present value 
      m_degrate = self.m_degrate
      nj = self.nj   # or whatever you call the exponential coeff
       # term multiplier for current density in the expenential degradation dependence, bj=0.00004
      degrate = base_degrate + m_degrate*(np.maximum(maxT,1)**nj)*np.exp(bj*x[-1])  # base_degrate is 1% per annum
      alphaa = sw*2*5*10**-4 
      degrate_eff = degrate* (1.0 + alphaa * n_starts)  # degrate = 0.125  + m_degrate * max(maxT_interp - T_thresh, 0).^3.*exp(0.00001*xFine); degradation rate is % per annum
#      degrate_eff1 = degrate* (1.0 + alpha * starts1)
      #degrate = base_degrate + m_degrate * np.exp(bj*x[-1])
      dur_stack = degrade_stack / degrate_eff * 8760 # durability of the stack in hours  
      #print(degrate_eff)



  
      t_oper = load_factor*8760/100 # t_oper is the number of operating hours at rated load
      #t_oper = -0.6453*(self.cd_eta75_base_ratio*100) -13.571*(self.cd_eta75_base_ratio*100) + 3486.1 # operational plant hours per year , 40% CF

      t_total = t_oper * t_sys

      Nrep = int(np.floor(t_total / dur_stack))
      
      dur_rem = t_total - Nrep * dur_stack

      SEC_initial = SEC_stack                   # kWh/kg H₂ (BOL)
      SEC_end = SEC_initial * (1 + degrade_stack / 100) 
      SEC_avg_per_stack = 0.5 * (SEC_initial + SEC_end)
      if dur_rem > 0:
        frac_life = dur_rem / dur_stack  # Fraction of last stack used
        SEC_end_partial = SEC_initial + frac_life * (SEC_end - SEC_initial)
        SEC_avg_partial = 0.5 * (SEC_initial + SEC_end_partial)
      else:
        SEC_avg_partial = 0

      SEC_avg_stack = (Nrep * dur_stack * SEC_avg_per_stack + dur_rem * SEC_avg_partial) /t_total#  avg stack SEC over plant lifetime
      
      SEC_avg_sys = ((SEC_avg_stack* vap_h2_pdt * MH2 * 3600 + W_sys_exstack + np.sum(heater_start_pow*heater_mask*0.001*dt)/8760 +  np.sum(pump_start_pow*standby_mask_eff*0.001*dt)/8760))/(vap_h2_pdt*MH2*3600)/eta_pow
      #SEC_avg_sys=SEC_avg_sys; #% This value is currently not being used in the glob. optim GA code. This shud be ideally the value to minimize.

      #replacement_pv = Tot_stack_cost / (1 + r) ** (dur_stack / t_oper)
      repl_inter_yrs = dur_stack/t_oper
      PV_repl=0.0
      for k in range(1,Nrep+1):
         PV_repl+= Tot_stack_cost/(1+r)**(k*repl_inter_yrs)

      Tot_stack_cost_life = Tot_stack_cost+PV_repl # total stack costs over plant lifetime   
      ######################################### Total BM_capex calculation... ###########################################

      BM_capex = Tot_stack_cost_life+BM_pow + BM_heaters_lye + BM_heaters + BM_h2cool+ BM_ads + BM_de+BM_glsep + BM_cwpump+BM_lyepumps + BM_evap +BM_refr + BM_shaft_h2comp +BM_shaft_refrcomp+ BM_h2comp + BM_refrcomp + BM_intercool + BM_fan +BM_refrfan +BM_lyecool +BM_deoxocool + BM_des_cool
      BM_0_capex = Tot_stack_cost/Fp_stack +BM_pow + BM_heaters_lye + BM_heaters + BM_0_h2cool + BM_0_ads + BM_0_de + BM_0_glsep + BM_0_cwpumps + BM_0_lyepumps + BM_0_evap + BM_0_refr + BM_0_shaft_h2comp+BM_0_shaft_refrcomp + BM_0_h2comp +BM_0_refrcomp + BM_0_intercool + BM_0_fan + BM_0_refrfan + BM_0_lyecool +BM_0_deoxocool + BM_0_des_cool
      capex_aux = BM_0_capex*0.5# Auxilliaries
      capex_cont =  0.18*BM_capex # contingnecy as a percentage of total fixed costs
      TIC = 1.18*BM_capex + capex_aux

      BM_h2compression = BM_intercool + BM_shaft_h2comp + BM_h2comp + BM_fan
      BM_refrig = BM_refr+BM_evap+BM_shaft_refrcomp+BM_refrcomp+BM_refrfan
      BM_gasliqsep = BM_glsep+BM_h2cool
      BM_h2purif = BM_des_cool + BM_deoxocool + BM_ads + BM_de + BM_heaters #contains deoxo heater and ads heater

      BM_lyecool=BM_lyecool
      BM_heaters=BM_heaters_lye
      BM_pumps = BM_cwpump+BM_lyepumps
      BM_pow = BM_pow
      ############################### LCOE calculation ##########################
      base_cc = 100 # base plant cc in MW
      m_scale =0.9
      PV_basecapex = 900 # $/kW
      PV_capex = (PV_basecapex*(100*1000)*((genratio/2 * (W_sys75)*0.001)/(100))**(m_scale))/(genratio/2 * (W_sys75)) # scaled capex based on installed renew capacity
      wind_basecapex = 1500 # $/kW
      wind_capex = (wind_basecapex*(100*1000)*((genratio/2 * (W_sys75)*0.001)/(100))**(m_scale))/(genratio/2 * (W_sys75)) # scaled capex based on installed renew capacity
      PV_opex = 17
      wind_opex =44
      

      n_renew = 25
      CRF_renew = (r*(1+r)**n_renew)/((1+r)**n_renew-1)
                             # hours per step
       #% Cumulative energy produced per annum kWh/(kW installed)/yr, factor of 0.001 as the data from SAM is in watts / per unit kW insatlled capacity
      CF_wind = np.sum(wind_powerout_scaled)*0.001/((genratio/2 * (W_sys75))*8760) 
      CF_solar = np.sum(pv_powerout_scaled)*0.001/((genratio/2 * (W_sys75))*8760)
      LCOE_wind = (wind_capex*CRF_renew+wind_opex)/(CF_wind*8760)
      LCOE_solar = (PV_capex*CRF_renew+PV_opex)/(CF_solar*8760)
      E_gen   = np.sum(hybrid_powerout) # total energy generated by the hybrid renew system per annum kWh/yr
      elec_cost =0.5*(LCOE_wind+LCOE_solar)*(E_gen/total_energy_supplied_curt) # the total energy supplied is actually the energy used by the electrolyzer, ie the power profiles from the renew faciltiy falling within the electrolyzer's operating range.
      ############################################################################
      '''capex_components = {
        "Stack": Tot_stack_cost,
        "H2 comp": BM_h2compression,
        "Refrig": BM_refrig,
        "Gas-liq sep": BM_gasliqsep,
        "H2 purif": BM_h2purif,
        "Lye cool": BM_lyecool,
        "Heater": BM_heaters,
        "Pumps": BM_pumps,
        "Power": BM_pow,
      }
      plot_capex_pie(capex_components, "Candidate CAPEX breakdown",x)

      #print(TIC/W_sys75)
      #print(W_pump75)
      #print(self.xf)'''
      #capex = capex + PV_repl # replacement stack cost added to total capex
      cd = x[-1]

      g3 = cd_hto - cd
      g4 = cd/cd_eta75 - 1.1 # max. curr that can be drawn is 10% above the rated current
      out["F"] = np.array([SEC_avg_sys*elec_cost, TIC*CRF*1.03/(vap_h2_pdt* MH2* t_oper * 3600)]) # obj are energy costs in $/kg, and total capex in $/kg, 3% operational O&M
      out["G"] = np.array([g2, g3, g4])
      out["aux"] = {"W_sys75": W_sys75}
      #print (g2,g3,g4)  

if __name__ == "__main__":
  multiprocessing.set_start_method("spawn", force=True)
    # Sweep g1 (Tmax constraint), fix g2 base ratio 0.1
    

    # Sweep g2 (cd/eta75 ratio), fix Tmax constraint to 5
    
  Tmax_constraint_value = 120
  with multiprocessing.Pool(n_workers) as pool:
    runner = StarmapParallelization(pool.starmap)
    '''for m_val in m_degrate_sweep:
        for nj_val in nj_sweep:
           for ratio_val in cdeta_sweep_vals:
            #print(f"--- GA run for Tmax_constraint = 5.0, cd_eta75 base ratio = {ratio_val} ---")
                fname = (
                    f"ga_sens_cdeta{ratio_val:.3f}"
                    f"_m{m_val:.4f}"
                    f"_nj{nj_val:.1f}"
                    f"_PV_rated{rated_load:.1f}"
                    f"_gr{genratio:.1f}_lcoh"
                #    f"_heater{heater_frac_ratedpower:.2f}"
                    f"_job{jobid}_SECsys_.mat"
                )

                start_time = time.perf_counter()
                print(f'Starting run  {fname}')
                    #pool = multiprocessing.Pool(n_workers)
                #with multiprocessing.Pool(n_workers) as pool:
                # runner = StarmapParallelization(pool.starmap)
                problem = AWEMultiObjVarConst(
                        minIn_SEC, maxIn_SEC, minOut_SEC, maxOut_SEC,
                        maxT_IDX,
                        SEC_stack_IDX,
                        vap_h2_pdt_IDX,
                        H_T_O_IDX,
                        w_KOH_angl_out_IDX,
                        w_KOH_gl_out_IDX,
                        Q_gl_out_IDX,
                        Q_angl_out_IDX,
                        glsep_O2_IDX,
                        H2_mixedToHTO_IDX,
                        Q_cond_h2cooler_IDX,
                        Q_cond_ads_IDX,
                        Q_cond_deoxo_IDX,
                        T_gl_out_IDX,
                        T_angl_out_IDX,
                        cell_delP_IDX,
                        ancell_delP_IDX,
                        HHV_H2,
                        xl,
                        xu,
                        Tmax_constraint_value=Tmax_constraint_value,
                        cd_eta75_base_ratio=ratio_val,
                        m_degrate=m_val,
                        nj=nj_val,
                        elementwise_runner=runner
                )
                algorithm = NSGA2(
                        pop_size=pop_size,
                        sampling=FloatRandomSampling(),
                        crossover=SBX(prob=0.9, eta=10),
                        mutation=PM(eta=5),
                        eliminate_duplicates=True
                )
                    # more conservative hyperparam
                    #algorithm = NSGA2(
                    #    pop_size=pop_size,
                    #    sampling=FloatRandomSampling(),
                    #    crossover=SBX(prob=0.9, eta=15),  # a bit “tighter” offspring around parents
                    #    mutation=PM(eta=20),             # smaller perturbations but keep default prob
                    #    eliminate_duplicates=True
                    #)
                
                    #generations = []
                    #best_sec, mean_sec, nds_counts, diversity, feasible_log = [], [], [], [], []
                    #best_vap, mean_vap, mean_capex, best_capex = [], [], [], []
                    #history = []
                cd_eta75_base_ratioo = ratio_val
                m_vall = m_degrate_sweep
                njj = nj_val # to use in the post GA code for collecting non- optimized variables
                res = minimize(
                        problem, algorithm, ("n_gen", n_generations),
                        seed=1, verbose=True #, callba callback=log_callback ck=log_callback
                )'''

                    # -------- fixed min-load ratio for all runs --------
    cdeta_fixed = 0.28

    # -------- sensitivity lists over x[0]..x[5] --------
    # x order: [elec width, pressure, inlet vel, pore size, sep width, curr dens]
    sweep_lists = { 
       # POPULATE THE required parameter with values, for performing the parameter fixed GA optimization of the RES powered AWEBOP system.... example currently the current density is being fixed, while others are free to be adjusted by the GA optimizer
      #  0: list(np.arange(1.55, 1.625, 0.025)),  # upper bound is exclusive
         # electode width
      #  0:[1.55, 1.57 ,1.59,1.6]
      #  1: [9.1,9.2, 9.3,9.5,10.0] # 5.8 5.85 5.9 5.95],             # operating pressure
      #  2: [0.0075],  # inlet velocity
      #  3: [350] #[200, 250,300,350,360,400,450,500, 550, 600,650,700, 750, 800] #         # pore size
       # 4: [0.485],  # separator width
         5: [6000,6500,7000,7500,8000,8500,9000, 9500,10000,10500,11000,11455,11500,12000,12500,13000],  # current density
    }

    param_names = {
        0: "elecwidth",
        1: "pressure",
        2: "inletvel",
        3: "poresize",
        4: "sepwidth",
        5: "currdens",
    }

    # keep a copy of the original ANN-based bounds
    xl_base = xl.copy()
    xu_base = xu.copy()

    for i_param, values in sweep_lists.items():
        pname = param_names[i_param]

        for v in values:
            print(
                f"=== GA run: fix param {pname} (x[{i_param}]) = {v}, "
                f"cd_eta75_base_ratio = {cdeta_fixed} ===",
                flush=True,
            )

            # local bounds: fix this parameter, others free
            xl_local = xl_base.copy()
            xu_local = xu_base.copy()
            xl_local[i_param] = v
            xu_local[i_param] = v

            # file name for this sensitivity case
            fname = (
                f"ga_sens_{pname}{v:.3f}"
           #    f"ga_sens_sepwidth0.101"
                f"_cdeta{cdeta_fixed:.2f}"
                f"_m{m_degrate_sweep[0]:.4f}"
                f"_nj{nj_sweep[0]:.1f}"
                f"_PV_rated{rated_load:.1f}"
                f"_gr{genratio:.1f}_lcoh"
                f"_job{jobid}_SECsys_.mat"
            )
     #start_time = time.perf_counter()
     #print(f'Starting run  {fname}')

            # GA problem for this fixed parameter value
            problem = AWEMultiObjVarConst(
                minIn_SEC, maxIn_SEC, minOut_SEC, maxOut_SEC,
                maxT_IDX,
                SEC_stack_IDX,
                vap_h2_pdt_IDX,
                H_T_O_IDX,
                w_KOH_angl_out_IDX,
                w_KOH_gl_out_IDX,
                Q_gl_out_IDX,
                Q_angl_out_IDX,
                glsep_O2_IDX,
                H2_mixedToHTO_IDX,
                Q_cond_h2cooler_IDX,
                Q_cond_ads_IDX,
                Q_cond_deoxo_IDX,
                T_gl_out_IDX,
                T_angl_out_IDX,
                cell_delP_IDX,
                ancell_delP_IDX,
                HHV_H2,
                xl_local,
                xu_local,
                Tmax_constraint_value=Tmax_constraint_value,
                cd_eta75_base_ratio=cdeta_fixed,
                m_degrate=m_degrate_sweep[0],
                nj=nj_sweep[0],
                elementwise_runner=runner,
            )

            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=crossover_eta),
                mutation=PM(eta=mutation_eta),
                eliminate_duplicates=True,
            )

            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_generations),
                seed=1,
                verbose=True, callback=log_callback
            )

            # keep your existing Pareto processing & saving block here,
            # using `fname` as the output .mat name
            # (no change needed inside that block)

            if res is None or getattr(res, 'F', None) is None:
                print("Optimizer failed/no feasible solutions. Skipping save.")
                continue

                #pool.close(); pool.join()
            F = res.F
            nds = NonDominatedSorting()
            if F.ndim == 3 or (F.ndim == 2 and F.shape[1] == 2):
                    fronts = nds.do(res.F)
                    pareto_front_indices = fronts[0]
                    pareto_F = res.F[pareto_front_indices]
                    pareto_X = res.X[pareto_front_indices]
                    pareto_G = res.G[pareto_front_indices]
            else:
                    pareto_F, pareto_X, pareto_G = res.F, res.X, res.G



    #   end_time = time.perf_counter()
    #   print(f"cd_eta75 ratio {ratio_val}: elapsed time = {end_time - start_time:.2f} s",
    #      flush=True)
        
        # ===== RECALCULATE W_sys75 (since aux is not preserved) =====
            print("Recalculating W_sys75 for Pareto solutions...")
            #SEC_model_post = tf.keras.models.load_model('trainedANN_seed1273.keras')
            SEC_model_post = tf.keras.models.load_model('trainedANN_seed7084_boxcox_3.keras')
            pareto_maxT = np.zeros(len(pareto_front_indices ))
            pareto_curtail_factor = np.zeros(len(pareto_front_indices ))
            pareto_LCOE_awe = np.zeros(len(pareto_front_indices ))
            pareto_W_sys75 = np.zeros(len(pareto_front_indices ))
            pareto_load_factor = np.zeros(len(pareto_front_indices))
            pareto_vap_h2_pdt = np.zeros(len(pareto_front_indices))
            pareto_Nrep  = np.zeros(len(pareto_front_indices))
            pareto_degrate_eff = np.zeros(len(pareto_front_indices))
            pareto_n_starts = np.zeros(len(pareto_front_indices))
            #standby_time = np.zeros(len(pareto_front_indices))
            BMM_heaters = np.zeros(len(pareto_front_indices))
            BMM_pow = np.zeros(len(pareto_front_indices))
            BMM_h2compression = np.zeros(len(pareto_front_indices))
            BMM_refrig = np.zeros(len(pareto_front_indices))  
            BMM_gasliqsep = np.zeros(len(pareto_front_indices))
            BMM_h2purif = np.zeros(len(pareto_front_indices))
            BMM_lyecool = np.zeros(len(pareto_front_indices))
            BMM_pumps = np.zeros(len(pareto_front_indices))
            pareto_SEC_sta = np.zeros(len(pareto_front_indices))
            pareto_cd_eta75 = np.zeros(len(pareto_front_indices))
            pareto_frac_heating_limited = np.zeros(len(pareto_front_indices))
            pareto_hours_prod = np.zeros(len(pareto_front_indices))
            pareto_hours_standby = np.zeros(len(pareto_front_indices))
            pareto_hours_idle  = np.zeros(len(pareto_front_indices))
            pareto_therm_tau = np.zeros(len(pareto_front_indices))
            pareto_p_min_standby = np.zeros(len(pareto_front_indices))
            pareto_frac_standby_ener = np.zeros(len(pareto_front_indices))   
            n_pf = len(pareto_front_indices)
            pareto_T_hist = np.zeros((n_pf, 8760))
            pareto_SEC_avg_stack = np.zeros(len(pareto_front_indices))
            base_degrate = 0.125*8.760
            degrade_stack =10
            #m_vall = m_vall[0]
            for i in range(len(pareto_front_indices)):
                    x = pareto_X[i]



                    # Generate cd grid
                    cd_grid = np.linspace(100, maxIn_SEC[-1], 500)
                    param_mat = np.tile(x[:5], (len(cd_grid), 1))
                    grid_in = np.hstack([param_mat, cd_grid[:, None]])
                    grid_in_scaled = 2 * (grid_in - minIn_SEC) / (maxIn_SEC - minIn_SEC) - 1

                    # Predict
                    y_grid = SEC_model_post.predict(grid_in_scaled, verbose=0)
                    y_grid_unscaled = 0.5 * (y_grid + 1) * (maxOut_SEC - minOut_SEC) + minOut_SEC
                  
                    j = cd_grid
                    j = j.reshape(-1)           # current density, length N
                    frac = 1 
  #W_refrcomp     = y_grid_unscaled[:, self.W_refrcomp_IDX]*gc
                    # Extract outputs

                    maxT_s           = y_grid_unscaled[:, maxT_IDX]
                    SEC_stack_s      = y_grid_unscaled[:, SEC_stack_IDX]
                    vap_h2_pdts      = y_grid_unscaled[:, vap_h2_pdt_IDX]*gc
                    HTO_s            = y_grid_unscaled[:, H_T_O_IDX]
                    w_KOH_angl_out_s = y_grid_unscaled[:, w_KOH_angl_out_IDX]
                    w_KOH_gl_out_s   = y_grid_unscaled[:, w_KOH_gl_out_IDX]
                    Q_gl_out_s       = y_grid_unscaled[:, Q_gl_out_IDX]
                    Q_gl_out_s = (inv_boxcox(Q_gl_out_s, lam, 0.0))*gc

                    Q_gl_out_s = sm.nonparametric.lowess(
                      endog=Q_gl_out_s, exog=j,
                      frac=frac,         # larger -> smoother, smaller -> more local detail
                      it=1,              # robustifying iterations; increase if needed
                      return_sorted=False
                    )


                    Q_angl_out_s     = y_grid_unscaled[:, Q_angl_out_IDX]
                    Q_angl_out_s = (inv_boxcox(Q_angl_out_s, lam, 0.0))*gc

                    Q_angl_out_s = sm.nonparametric.lowess(
                      endog=Q_angl_out_s, exog=j,
                      frac=frac,         # larger -> smoother, smaller -> more local detail
                      it=1,              # robustifying iterations; increase if needed
                      return_sorted=False
                    )


                    H2_mixedToHTO_s  = y_grid_unscaled[:, H2_mixedToHTO_IDX]
                    Q_cond_h2cooler_s  = y_grid_unscaled[:, Q_cond_h2cooler_IDX]*gc
                    Q_cond_ads_s       = y_grid_unscaled[:, Q_cond_ads_IDX]*gc
                    Q_cond_deoxo_s     = y_grid_unscaled[:, Q_cond_deoxo_IDX]*gc
                    #W_refrcomp     = y_grid_unscaled[:, self.W_refrcomp_IDX]*gc
                    T_gl_out_s         = y_grid_unscaled[:, T_gl_out_IDX]
                    T_angl_out_s       = y_grid_unscaled[:, T_angl_out_IDX]
                    glsep_O2_s         = y_grid_unscaled[:, glsep_O2_IDX]
                    glsep_O2_s         = (inv_boxcox(glsep_O2_s, lam, 0.0))*gc # no polynomial smoothing required for this var
                    ancell_delP_s  = y_grid_unscaled[:, ancell_delP_IDX]
                    ancell_delP_s = (inv_boxcox(ancell_delP_s, lam, 0.0))

                    ancell_delP_s = sm.nonparametric.lowess(
                      endog=ancell_delP_s, exog=j,
                      frac=frac,         # larger -> smoother, smaller -> more local detail
                      it=1,              # robustifying iterations; increase if needed
                      return_sorted=False
                    )

                    cell_delP_s    = y_grid_unscaled[:, cell_delP_IDX]  
                    cell_delP_s = (inv_boxcox(cell_delP_s, lam, 0.0))

                    cell_delP_s = sm.nonparametric.lowess(
                      endog=cell_delP_s, exog=j,
                      frac=frac,         # larger -> smoother, smaller -> more local detail
                      it=1,              # robustifying iterations; increase if needed
                      return_sorted=False
                    )


                    # Interpolation
                    interp_maxT           = interp1d(cd_grid, maxT_s,           kind='linear', fill_value="extrapolate")
                    interp_SEC_stack      = interp1d(cd_grid, SEC_stack_s,      kind='linear', fill_value="extrapolate")
                    interp_vap_h2_pdt     = interp1d(cd_grid, vap_h2_pdts,     kind='linear', fill_value="extrapolate")
                    interp_H_T_O          = interp1d(cd_grid, HTO_s,          kind='linear', fill_value="extrapolate")
                    interp_w_KOH_angl_out = interp1d(cd_grid, w_KOH_angl_out_s, kind='linear', fill_value="extrapolate")
                    interp_w_KOH_gl_out   = interp1d(cd_grid, w_KOH_gl_out_s,   kind='linear', fill_value="extrapolate")
                    interp_Q_gl_out       = interp1d(cd_grid, Q_gl_out_s,       kind='linear', fill_value="extrapolate")
                    interp_Q_angl_out     = interp1d(cd_grid, Q_angl_out_s,     kind='linear', fill_value="extrapolate")
                    interp_H2_mixedToHTO  = interp1d(cd_grid, H2_mixedToHTO_s,  kind='linear', fill_value="extrapolate")
                    interp_Q_cond_h2cooler = interp1d(cd_grid, Q_cond_h2cooler_s, kind='linear', fill_value="extrapolate")
                    interp_Q_cond_ads     = interp1d(cd_grid, Q_cond_ads_s,     kind='linear', fill_value="extrapolate")
                    interp_Q_cond_deoxo   = interp1d(cd_grid, Q_cond_deoxo_s,   kind='linear', fill_value="extrapolate")
                    #interp_W_refrcomp     = interp1d(cd_grid, W_refrcomp,     kind='linear', fill_value="extrapolate")
                    interp_T_gl_out       = interp1d(cd_grid, T_gl_out_s,       kind='linear', fill_value="extrapolate")
                    interp_T_angl_out     = interp1d(cd_grid, T_angl_out_s,     kind='linear', fill_value="extrapolate")
                    #interp_T_reg          = interp1d(cd_grid, T_reg,          kind='linear', fill_value="extrapolate")
                    interp_glsep_O2       = interp1d(cd_grid, glsep_O2_s,     kind='linear', fill_value="extrapolate")

                    interp_ancell_delP      = interp1d(cd_grid, ancell_delP_s,     kind='linear', fill_value="extrapolate")
                    

                    interp_cell_delP      = interp1d(cd_grid, cell_delP_s,     kind='linear', fill_value="extrapolate")

                    # Find cd_eta75
                    eta_stack = (HHV_H2 / SEC_stack_s) * 100
                    idx = np.where(np.diff(np.sign(eta_stack - rated_load)))[0]

                    if len(idx) > 0:
                            i_idx = idx[-1]
                            cd1, cd2 = cd_grid[i_idx], cd_grid[i_idx + 1]
                            eta1, eta2 = eta_stack[i_idx], eta_stack[i_idx + 1]
                            cd_eta75 = cd1 + (rated_load - eta1) * (cd2 - cd1) / (eta2 - eta1)
                # Compute W_sys75
                            '''SEC_sys_at_eta75 = interp_func1(cd_eta75)
                            SEC_sys = interp_func1(x[-1])
                            pareto_SEC_sta[i] = interp_func14(x[-1])
                            vap_h2_pdt = interp_func2(x[-1])
                            pareto_maxT[i] = interp_func3(x[-1])
                            W_sys  = SEC_sys * vap_h2_pdt * MH2 * 3600
                            vap_h2_pdt_eta75 = interp_func2(cd_eta75)
                            SEC_sys_at_peak = interp_func1(cd_eta75*1.1)
                            vap_h2_pdt_peak = interp_func2(cd_eta75*1.1)
                            pareto_W_sys75[i] = SEC_sys_at_eta75 * vap_h2_pdt_eta75 * MH2 * 3600
                            W_sys_peak = SEC_sys_at_peak * vap_h2_pdt_peak * MH2 * 3600'''
                    cd = x[-1]
                    maxT          = interp_maxT(cd)
                    pareto_maxT[i]= interp_maxT(cd)
                    SEC_stack     = interp_SEC_stack(cd)
                    pareto_SEC_sta[i] = SEC_stack
                    vap_h2_pdt    = interp_vap_h2_pdt(cd)
                    HTO         = interp_H_T_O(cd)
                    w_KOH_angl_out = interp_w_KOH_angl_out(cd)
                    w_KOH_gl_out   = interp_w_KOH_gl_out(cd)
                    Q_gl_out      = interp_Q_gl_out(cd)
                    Q_angl_out    = interp_Q_angl_out(cd)
                    H2_mixedToHTO = interp_H2_mixedToHTO(cd)
                    Q_cond_h2cooler = interp_Q_cond_h2cooler(cd)
                    Q_cond_ads    = interp_Q_cond_ads(cd)
                    Q_cond_deoxo  = interp_Q_cond_deoxo(cd)
                    T_gl_out      = interp_T_gl_out(cd)
                    T_angl_out    = interp_T_angl_out(cd)
                    glsep_O2      = interp_glsep_O2(cd)
                    ancell_delP   = interp_ancell_delP(cd)
                    cell_delP   =   interp_cell_delP(cd)
                    Cp_KOH_lye = (4.101*10**3-3.526*10**3*w_KOH_gl_out  + 9.644*10**-1*(T_gl_out-273.15)+1.776*(T_gl_out-273.15)*w_KOH_gl_out)
                    Cp_KOH_anlye = (4.101*10**3-3.526*10**3*w_KOH_angl_out + 9.644*10**-1*(T_angl_out-273.15)+1.776*(T_angl_out-273.15)*w_KOH_angl_out)

                    rho_KOH_gl_out = (1001.53 - 0.08343*(T_gl_out - 273.15) - 0.004*(T_gl_out-273.15)**2 + 5.51232*10**-6*(T_gl_out-273.15)**3 - 8.21*10**-10*(T_gl_out-273.15)**4)*np.exp(0.86*w_KOH_gl_out)
                    rho_KOH_angl_out = (1001.53 - 0.08343*(T_angl_out - 273.15) - 0.004*(T_angl_out-273.15)**2 + 5.51232*10**-6*(T_angl_out-273.15)**3 - 8.21*10**-10*(T_angl_out-273.15)**4)*np.exp(0.86*w_KOH_angl_out)

                    Q_lyecooler = np.maximum((T_gl_out-273.15-80)*Cp_KOH_lye*Q_gl_out*rho_KOH_gl_out,0)  + np.maximum((T_angl_out-273.15-80)*Cp_KOH_anlye*Q_angl_out*rho_KOH_angl_out,0) # lye cooler duty at candidate load
                    

                    p_vapor_h2cool_out = 0.61121*np.exp((18.678-(T_glsepHXout-273.15)/234.5)*((T_glsepHXout-273.15)/(257.14+T_glsepHXout-273.15)))*1000 # vapor pressure of water at 25C and ambient conditions

                    Q_lyeheater = 0.001*((np.maximum((273.15+80-T_gl_out)*Cp_KOH_lye*Q_gl_out*rho_KOH_gl_out,0)  + np.maximum((273.15+80-T_angl_out)*Cp_KOH_anlye*Q_angl_out*rho_KOH_angl_out,0))) # lye heater power , in kW

                    SEC_stack_at_eta75 = interp_SEC_stack(cd_eta75)
                    vap_h2_pdt_eta75 = interp_vap_h2_pdt(cd_eta75)
                    #SEC_sys_at_peak = interp_SEC_stack(cd_eta75*1.1)
                    vap_h2_pdt_peak = interp_vap_h2_pdt(cd_eta75*1.1)
                    Q_gl_out_at_eta75 = interp_Q_gl_out(cd_eta75)
                    Q_angl_out_at_eta75 = interp_Q_angl_out(cd_eta75)
                    T_angl_out_at_eta75 = interp_T_angl_out(cd_eta75)
                    T_gl_out_at_eta75 = interp_T_gl_out(cd_eta75)
                    w_gl_out_at_eta75 = interp_w_KOH_gl_out(cd_eta75)
                    w_angl_out_at_eta75 = interp_w_KOH_angl_out(cd_eta75)
                    Q_cond_h2cooler_at_eta75 = interp_Q_cond_h2cooler(cd_eta75)
                    Q_cond_ads_at_eta75 = interp_Q_cond_ads(cd_eta75)
                    Q_cond_deoxo_at_eta75 = interp_Q_cond_deoxo(cd_eta75)
                    #glsep_O2_eta75  =   interp_glsep_O2(cd_eta75)

                    glsep_O2_at_eta75 = interp_glsep_O2(cd_eta75)
                    ancell_delP_at_eta75 = interp_ancell_delP(cd_eta75)
                    cell_delP_at_eta75 = interp_cell_delP(cd_eta75)

                    Cp_KOH_lye_at_eta75 = (4.101*10**3-3.526*10**3*w_gl_out_at_eta75 + 9.644*10**-1*(T_gl_out_at_eta75-273.15)+1.776*(T_gl_out_at_eta75-273.15)*w_gl_out_at_eta75)
                    Cp_KOH_anlye_at_eta75 = (4.101*10**3-3.526*10**3*w_angl_out_at_eta75 + 9.644*10**-1*(T_angl_out_at_eta75-273.15)+1.776*(T_angl_out_at_eta75-273.15)*w_angl_out_at_eta75)

                    rho_KOH_gl_out_at_eta75 = (1001.53 -0.08343*(T_gl_out_at_eta75-273.15) - 0.004*(T_gl_out_at_eta75-273.15)**2 + 5.51232*10**-6*(T_gl_out_at_eta75-273.15)**3 - 8.21*10**-10*(T_gl_out_at_eta75-273.15)**4)*np.exp(0.86*w_gl_out_at_eta75)
                    rho_KOH_angl_out_at_eta75 = (1001.53 -0.08343*(T_angl_out_at_eta75-273.15) - 0.004*(T_angl_out_at_eta75-273.15)**2 + 5.51232*10**-6*(T_angl_out_at_eta75-273.15)**3 - 8.21*10**-10*(T_angl_out_at_eta75-273.15)**4)*np.exp(0.86*w_angl_out_at_eta75)

                    Q_lyecooler_at_eta75 = np.maximum((T_gl_out_at_eta75-273.15-80)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0)  + np.maximum((T_angl_out_at_eta75-273.15-80)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0) # lye cooler duty at rated load     
                            
                ###########################################################################     AUXILLIARY CAPEX CALCULATION       ###########################################

                    # STACK
                    Ni_foam = 0.5/10**-6; # $0.5/cm^3,https://shop.nanografi.com/battery-equipment/nickel-foam-for-battery-cathode-substrate-size-1000-mm-x-300-mm-x-1-6-mm/
                    zf_sep = gc*150*0.05*0.008*conv_rate*Nc; # 150 Euro/m^2 from AGFA Data 'Present and future cost of alkaline and PEM electrolyser stacks'
                    steel = gc*(0.05*0.008*0.003*2+0.05*0.008*0.001*(Nc-1))*rho_steel*4.5;    # 0.9 $ /kg for carbon steel, 4.5$/kg for SS316L. SS used for longer stack life and in advanced stack designs operating at higher cd.
                    Ni =  gc*2*(0.05*0.008*x[0]*0.001*(Nc))*Ni_foam; # 15 $/kg Ni price as on 4/10/2025, based on volume of electrode
                    Fp_stack = 1+0.2*(x[1]-1)/(15-1);# Saba etal 2018, reported around 20% increase in stack costs for a pressure increase from 1 bar to 15 bar, assuming a linear relationship between operating pressure and stack material costs
                    stack_mat_cost = 1.3*(zf_sep+steel+Ni)*Fp_stack;  # $/kW, 30% additional for balance of stack components such as gaskets etc.
                    dir_stack_cost = stack_mat_cost+ 0.5*stack_mat_cost+0.125*stack_mat_cost; # ratio is as 8:4:1 from 'Present and future cost of alkaline and PEM electrolyser stacks'.

                    Tot_stack_cost = 2*dir_stack_cost # % including overheads $/kW

                    #Calc. of cooling water flowrate at rated conditions
                    #Cp_KOH

                    Q_lyeheater_at_eta75 = 0.001*(np.maximum((273.15+80-T_gl_out_at_eta75)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0)  + np.maximum((273.15+80-T_angl_out_at_eta75)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0)) # lye heater duty at rated load in kW

                    m_cw_circ_at_eta75 =   (Q_lyecooler_at_eta75 + Q_cond_ads_at_eta75 + Q_cond_deoxo_at_eta75 + Q_cond_h2cooler_at_eta75)/(Cp_cw*(T_cwr-T_cws)) # cooling water circulation rate in kg/s

                    m_cw_circ =   (Q_lyecooler + Q_cond_ads + Q_cond_deoxo + Q_cond_h2cooler)/(Cp_cw*(T_cwr-T_cws)) # cooling water circulation rate in kg/s

              
                    # Gas-Liquid Separator vessels
                        # Pres vessels 2 min resid time considered....... 
                    od_glsep = 1.15
                    #Q_gl_out = x[2]*0.008*x[0]*0.001*Nc*gc # in m^3/s
                    Pg = x[1]-1 # guage pressure
                    vol_liq_glsep = 120 * (Q_gl_out_at_eta75 + Q_angl_out_at_eta75)                                   # m^3, 2 mins residence time
                    vol_glsep     = od_glsep * vol_liq_glsep                      # overdesigned volume, 50% fill level
                    glsep_D       = ((4 * vol_glsep) / (3 * np.pi)) ** (1.0 / 3.0)   # diameter [m]
                    glsep_L       = 3 * glsep_D                                      # length [m]

                    # Purchase cost
                    CP_glsep = 2 * 10 ** (3.5565
                                          + 0.3776 * np.log10(max(vol_glsep, 0.1))
                                          + 0.09 * (np.log10(max(vol_glsep, 0.1))) ** 2)

                    CP_glsep_min = 2 * 10 ** (3.5565
                                              + 0.3776 * np.log10(0.1)
                                              + 0.09 * (np.log10(0.1)) ** 2)

                    # Pressure vessel factor
                    Fp_glsep = max(((Pg + 1) * glsep_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0)

                    FBM_glsep = 1.49 + 1.52 * 1.7 * Fp_glsep

                    BM_glsep  =  FBM_glsep * CP_glsep * CI
                    BM_0_glsep = CP_glsep*(1.49+1.52)*CI
                      
                    # 6/10 rule if minimum capacity governs
                    if (CP_glsep == CP_glsep_min) or (vol_glsep > 650):
                        BM_glsep_min = CP_glsep * FBM_glsep 
                        BM_glsep     = 2 * sd * BM_glsep_min * (vol_glsep / 0.1) ** nv
                        BM_0_glsep   = CP_glsep_min*(1.49+1.52) * CI
                        #self.xb += 1
                    
                    # Volume and geometry (now also includes thermal calc. based on either GHSV, or catalyst mass)
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
                    b_T = b_ref*np.exp(H_ads/R_const*(1/T_glsepHXout-1/293.15)) #; % b value at 25C
                    p_H2O_adspec = np.mean(adspec_H2O*x[1]) #; % par_pres of moisture in ads outlet h2 stream at 25C ads temperature
                    alpha_purge = 0.5  #;
                    x_H2O_h2cool_out = p_vapor_h2cool_out/(x[1]*10**5)
                    vap_h2_pdt = interp_vap_h2_pdt(x[-1]) # product gas flow rate at the candidate current density
                    dry_gas = vap_h2_pdt*(1-adspec_H2O)
                    vap_ads = vap_h2_pdt * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber

                    ads_H2O_in = (dry_gas*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
                    ads_H2O_out = (dry_gas*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;

                    ads_cap_rate = ads_H2O_in-ads_H2O_out #; % rate of moisture capture 
                    vel_ads = (vap_ads*R_const*T_glsepHXout/(x[1]*10**5))/(np.pi*ads_D**2/4.0)
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
                    xH2O_equil = 101.325/(x[1]*10**2) # equil moisture content at 100C and pressure of operation
                    p_H2O_reg_out = x[1]*10**5*(x_H2O_h2cool_out+1.0/rp)/(1.0+1.0/rp)
                    p_vap_reg_eff = x[1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out-x_H2O_h2cool_out))

                    b_reg = ((rq**0.2472)/(1.0-rq**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff)

                    T_reg = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg/b_ref))**-1)

                    T_base_vec = T_glsepHXout  # replicate as in MATLAB

                    thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

                    m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)
                    therm_cat_ads = m_ads*Cp_ads
                    therm_st_ads = m_steel_ads*Cp_steel
                    heater_ads = 0.001*(purge_ads * Cp_H2_80 * (T_reg - T_base_vec) + des_rate * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg - T_base_vec) / t_des / eta_heater) * t_des / t_b) # in kW
                    condensate_H2O_regen_cond = (purge_ads * (1.0 - x_H2O_h2cool_out) * xH2O_reg_out / (1.0 - xH2O_reg_out)- x_H2O_h2cool_out * purge_ads)
                    cond_ads = (condensate_H2O_regen_cond*lat_heat_water+ purge_ads * Cp_H2_80 * (T_reg - T_glsepHXout)) * t_des / t_b 
                    m_cw_cond_ads = cond_ads / (Cp_cw * (T_cwr - T_cws))
                    # Ergun equation terms for calc. of delP in the regen bed
                    vel_ads = (purge_ads*R_const*T_glsepHXout/(x[1]*10**5))/(np.pi*ads_D**2/4.0)
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
                    eta_blower = 0.7 # desorber blower efficiency
                    delP_des = 1.3*(term1+term2) # factor of 1.3 for calc. of other pressure drop conponnets in the recirc loop of purge gas
                    flow_purge_ads = purge_ads*R_const*(T_reg)/(x[1]*10**5) # flowrate of purge adsorber
                    W_blower = flow_purge_ads*delP_des/eta_blower # blower power us typically much lower
                    # Purchase cost
                    CP_ads = 2 * 10 ** (3.4974
                                        + 0.4485 * np.log10(max(vol_ads, 0.3))
                                        + 0.1074 * (np.log10(max(2 * vol_ads, 0.3)))**2)

                    CP_ads_min = 2 * 10 ** (3.4974
                                            + 0.4485 * np.log10(0.3)
                                            + 0.1074 * (np.log10(0.3))**2)

                    # Pressure vessel factor
                    Fp_ads = max(((Pg + 1) * ads_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0)

                    FBM_ads = 2.25 + 1.82 * 1.7 * Fp_ads

                    BM_ads = FBM_ads * CP_ads  * CI
                    BM_0_ads = CP_ads*(2.25 + 1.82) * CI

                    # 6/10 rule for minimum capacity
                    if (CP_ads_min == CP_ads) or (vol_ads > 520):
                        BM_ads_min = CP_ads_min * FBM_ads
                        BM_ads = sd * BM_ads_min * (vol_ads / 0.3)**nv*CI
                        BM_0_ads = CP_ads_min *(2.25 + 1.82)*CI
                        #self.xc += 1
                    
                    

                    od_deoxo = 1.2
                    GHSV = 2000.0   # per hr, reference ......
                    tcycle_deoxo = 8*3600 # number of seconds per cycle (8 hrs)
                    # Volume and geometry
                    vol_deoxo = od_deoxo * max(0.001* vap_h2_pdt_eta75 * 22.414 * 3600.0 / GHSV, 0.0)
                    D_deoxo = 1.0   # m
                    L_deoxo = vol_deoxo / (np.pi * D_deoxo**2 / 4.0)
                    thick_deoxo = max(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness
                    mass_deoxo = vol_deoxo*650 # bulk density of deoxo catalyst is 650 kg/m^3
                    Cp_deoxo = 900
                    mass_st_deoxo = (np.pi*(D_deoxo+thick_deoxo)**2.0/4.0*L_deoxo-np.pi*(D_deoxo)**2.0/4.0*L_deoxo)*rho_steel  #%2mm thickness
                    therm_st_deoxo = mass_st_deoxo * Cp_steel / tcycle_deoxo #* np.ones(len(deoxo_heat)) thermal absorption rate of steel in the deoxidizer
                    therm_cat_deoxo = mass_deoxo * Cp_deoxo / tcycle_deoxo #* np.ones(len(deoxo_heat)) thermal absorption rate of catalyst in the deoxidizer



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

                    heater_deoxo = 0.001*((deoxo_T > T_deoxo).astype(float)* vap_h2_pdt* Cp_vap_h2cool* (deoxo_T - T_deoxo)/ eta_heater)
                    heater_regen = ((therm_st_deoxo + therm_cat_deoxo)* (723.15 - deoxo_T)/ eta_heater) # regen temp of deoxo is 450C

                    deoxo_H2O_eta75 = 2.0 * glsep_O2_at_eta75
                    deoxo_H2reac_eta75 = 2.0 * glsep_O2_at_eta75
                    deoxo_heat_eta75   = 2.0 * glsep_O2_at_eta75 * 244.9 * 10.0**3   # 244.9 kJ/mol → J/mol
                    Cp_vap_h2cool = (
                        Cp_watvap_80 * x_H2O_h2cool_out          
                        + Cp_H2_80 * (1-x_H2O_h2cool_out)
                    )

                    T_deoxo_at_eta75   = ( deoxo_heat_eta75 
                    + therm_cat_deoxo * (T_glsepHXout + 125.0)
                    + therm_st_deoxo * (T_glsepHXout + 125.0)
                    + vap_h2_pdt_eta75 * Cp_vap_h2cool * T_glsepHXout) / (vap_h2_pdt_eta75 * Cp_vap_h2cool + therm_st_deoxo + therm_cat_deoxo)

                    deoxo_T = 150 +273.15 # deoxo maintains temp. at 150C
                    heater_deoxo_at_eta75 = 0.001*((deoxo_T > T_deoxo_at_eta75 ).astype(float)* vap_h2_pdt_eta75* Cp_vap_h2cool* (deoxo_T - T_deoxo_at_eta75)/ eta_heater)
                    heater_regen = 0.001*((therm_st_deoxo + therm_cat_deoxo)* (723.15 - deoxo_T)/ eta_heater) # regen temp of deoxo is 450C



                    # Purchase cost (2 desorbers)
                    CP_de = 2 * 10 ** (3.4974
                                      + 0.4485 * np.log10(max(vol_deoxo, 0.3))
                                      + 0.1074 * (np.log10(max(vol_deoxo, 0.3)))**2)

                    CP_de_min = 2 * 10 ** (3.4974
                                          + 0.4485 * np.log10(0.3)
                                          + 0.1074 * (np.log10(0.3))**2)

                    # Pressure vessel factor
                    Fp_deoxo = max(((Pg + 1) * D_deoxo / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315) / 0.0063, 1.0)

                    FBM_de = 2.25 + 1.82 * 1.7 * Fp_deoxo

                    BM_de = sd * FBM_de * CP_de* CI
                    BM_0_de = (2.25 + 1.82)*CP_de*CI

                    # 6/10 rule when minimum capacity governs
                    if (CP_de_min == CP_de) or (vol_deoxo/2 > 520):
                        BM_de_min = CP_de_min * FBM_de
                        BM_de = sd * BM_de_min * (vol_deoxo / 0.3)**nv
                        BM_0_de = CP_de_min * CI*(2.25 + 1.82)
                        #self.xd += 1

                    # H2 compressor
                    Ncomp = math.ceil(math.log(P_final / (x[1])) / math.log(rc_max)) 
                    Ncomp = max(Ncomp, 1)       # at least one stage

                    od_h2comp = 1.15

                    if Ncomp <= 0:
                      # physically impossible, punish the design
                      rc = 1.0
                    elif Ncomp == 1:
                      # single‑stage compression: all ratio in one step
                      rc = rc_max #P_final / (x[1] * rc_max1)
                    else:
                      # multi‑stage: equal ratio per stage
                      rc = (P_final / (x[1] ))**(1.0 / (Ncomp)) # assuming uniform compression ratio, and compr size 

                    T_dis_comp = T_suc_comp * (1 + (rc**((k_isen - 1.0) / k_isen) - 1) / eta_isen_comp);  
                    '''if P_final / x[1] > 1.0:
                        term_over = (k_isen / (k_isen - 1.0) *
                        R_const * T_amb * Z_comp *
                        (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0) *
                        vap_h2_pdt_eta75 / (eta_isen_comp  * eta_mech_comp))
                    else:'''
                    term_over = 0.0

                    W_shaft_h2comp = 0.001*(max(Ncomp, 0) * k_isen / (k_isen - 1.0) *R_const * T_suc_comp * Z_comp *(rc ** ((k_isen - 1.0) / k_isen) - 1.0) *vap_h2_pdt_eta75  / (eta_isen_comp *  eta_mech_comp)+ term_over) # in kW, at rated current density for CAPEX calc.
                    W_comp_SEC = 0.001*(max(Ncomp, 0) * k_isen / (k_isen - 1.0))* R_const * T_suc_comp * Z_comp * (rc ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt / (eta_isen_comp * eta_mech_motor * eta_mech_comp) #+ (P_final / x[1] > 1.0)* (1.0 * k_isen / (k_isen - 1.0) * R_const * T_amb * Z_comp* (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt / (eta_isen_comp * eta_mech_motor * eta_mech_comp)))  # [web:1], in kW
                    intercool_comp_SEC = max(Ncomp, 0) * (T_dis_comp - T_suc_comp) * vap_h2_pdt * Cp_H2_80 #+ 1.0 * (T_max_comp - T_amb) * vap_h2_pdt * Cp_H2_80)  # [web:1]

                    air_fan_SEC = intercool_comp_SEC / (delT_air * Cp_air * rho_air)  # [web:1]

                    W_fan_SEC = 0.001*air_fan_SEC * delP_fan / eta_fan  # [web:1]



                    W_h2comp = W_shaft_h2comp/eta_mech_motor # in kW
                    CP_shaft_h2comp=Ncomp*(10**(2.2897+1.36*np.log10(max((od_h2comp*W_shaft_h2comp)/Ncomp,20))-0.1027*(np.log10(max((od_h2comp*W_shaft_h2comp)/Ncomp,20)))**2)) #min. input pow is 450kW reciprocating, max is 3000kW per stage
                    CP_shaft_h2comp_min =  Ncomp*(10**(2.2897+1.36*np.log10(np.max(20))-0.1027*(np.log10(np.max(20) ))**2)) # min. input pow is 450kW as per turton, but using limit of 300, as the function is smooth
                    FBM_shaft_h2comp = 7
                    BM_shaft_h2comp = sd * FBM_shaft_h2comp * CP_shaft_h2comp * CI
                    BM_0_shaft_h2comp = BM_shaft_h2comp / FBM_shaft_h2comp
                    if (CP_shaft_h2comp_min == CP_shaft_h2comp) or (od_h2comp*W_shaft_h2comp/Ncomp > 3000):
                      BM_shaft_h2comp_min = CP_shaft_h2comp_min * FBM_shaft_h2comp * CI
                      BM_shaft_h2comp = (sd* BM_shaft_h2comp_min* ((od_h2comp*W_shaft_h2comp / Ncomp) / 20.0) ** 0.86)
                      BM_0_shaft_h2comp = BM_shaft_h2comp / FBM_shaft_h2comp * CI
                      # self.xf += 1


                    od_refrcomp =1.20
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
                    W_refrcomp_SEC = 0.001*(delH_comp*m_refr/(eta_mech_comp*eta_mech_motor))
                    W_refrfan_SEC = 0.001*((m_refr*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan
                
                    #W_refrcompshaft_at_eta75 = W_refrcomp_at_eta75*eta_mech_motor
                    m_refr_at_eta75 = m_cw_circ_at_eta75*Cp_cw*(T_cwr-T_cws)/refr_lat_evap
                    W_refrcomp_at_eta75 = 0.001*(delH_comp*m_refr_at_eta75/(eta_mech_comp*eta_mech_motor))
                    W_refrfan_at_eta75 = 0.001*((m_refr_at_eta75*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan
                    W_refrcompshaft_at_eta75 = W_refrcomp_at_eta75*eta_mech_motor # in kW

                    CP_shaft_refrcomp = 10 ** (2.2897+ 1.36 * np.log10(max(od_refrcomp*W_refrcompshaft_at_eta75, 50.0))- 0.1027 * (np.log10(max(od_refrcomp*W_refrcompshaft_at_eta75, 50.0))) ** 2)
                    CP_shaft_refrcomp_min = 10 ** (2.2897+ 1.36 * np.log10(100.0)- 0.1027 * (np.log10(100.0)) ** 2)
                    FBM_shaft_refrcomp = 3.3
                    # Base shaft costs
                    BM_shaft_refrcomp = sd * FBM_shaft_refrcomp * CP_shaft_refrcomp* CI
                    BM_0_shaft_refrcomp = BM_shaft_refrcomp / FBM_shaft_refrcomp
                    # Refr. shaft min‑capacity check
                    if (CP_shaft_refrcomp_min == CP_shaft_refrcomp) or (od_refrcomp*W_refrcompshaft_at_eta75 > 3000):
                        BM_shaft_refrcomp_min = CP_shaft_refrcomp_min * FBM_shaft_refrcomp* CI
                        BM_shaft_refrcomp = (sd * BM_shaft_refrcomp_min * (od_refrcomp*W_refrcompshaft_at_eta75 / 50.0) ** 0.86)
                        BM_0_shaft_refrcomp = BM_shaft_refrcomp / FBM_shaft_refrcomp
                      #   self.xe += 1
                    
                    # comp drive
                    # Purchase costs
                    CP_refrcomp = 10 ** (2.9308+ 1.0688 * np.log10(max(od_refrcomp*W_refrcomp_at_eta75, 5.0))- 0.1315 * (np.log10(max(od_refrcomp*W_refrcomp_at_eta75, 5.0))) ** 2)
                    CP_refrcomp_min = 10 ** (2.9308+ 1.0688 * np.log10(5.0)- 0.1315 * (np.log10(5.0)) ** 2)
                    CP_h2comp = Ncomp * 10 ** (2.9308+ 1.0688 * np.log10(max(W_h2comp / Ncomp, 5.0))- 0.1315 * (np.log10(max(W_h2comp / Ncomp, 5.0))) ** 2) # min. drive power is 75kW,2600kW max
                    CP_h2comp_min = Ncomp * 10 ** (2.9308+ 1.0688 * np.log10(5.0)- 0.1315 * (np.log10(5.0)) ** 2)
                    FBM_comp = 1.5
                    BM_refrcomp = sd * FBM_comp * CP_refrcomp*CI
                    BM_h2comp   = sd * FBM_comp * CP_h2comp*CI
                    BM_0_h2comp   = BM_h2comp   / FBM_comp
                    BM_0_refrcomp = BM_refrcomp / FBM_comp
                    # Refrigeration compressor min-capacity scaling
                    if (CP_refrcomp == CP_refrcomp_min) or (od_refrcomp*W_refrcomp_at_eta75 > 4000):
                        BM_refrcomp_min = CP_refrcomp_min * FBM_comp
                        BM_refrcomp = sd * BM_refrcomp_min * (od_refrcomp*W_refrcomp_at_eta75 / 5.0) ** 0.6 * CI
                        BM_0_refrcomp = BM_refrcomp / FBM_comp
                    #    self.xg += 1
                    # H2 compressor min-capacity scaling
                    if (CP_h2comp == CP_h2comp_min) or (od_h2comp*W_h2comp / Ncomp > 4000):
                        BM_h2comp_min = CP_h2comp_min * FBM_comp
                        BM_h2comp = sd * BM_h2comp_min * (od_h2comp*W_h2comp / 5.0) ** 0.6*CI
                        BM_0_h2comp = BM_h2comp / FBM_comp
                    #    self.xh += 1
                    
                    # intercoolers , air cooled condensers and fans 
                    # Intercooler duty
                    od_aircool = 1.20
                    intercool_comp = (T_dis_comp - T_suc_comp) * vap_h2_pdt_eta75 * Cp_H2_80 
                    #intercool_comp_last = od_aircool * (T_max_comp - T_amb) * vap_h2_pdt_eta75 * Cp_H2_80 # comp intercooler duty a
                    U_intercool = 30 #%%%%%%%%%%%%%%% GAS TO GAS is 30
                    delTlm_intercool = ((T_dis_comp - T_aircooler) - (T_suc_comp - T_amb)) / np.log((T_dis_comp - T_aircooler) / (T_suc_comp - T_amb))

                    area_intercool = intercool_comp / (U_intercool * delTlm_intercool*FF) # area of each set of intercoolers per compressor


                    # Number of intercoolers
                    N_intercool = max(1, int(np.floor(od_aircool*area_intercool / 30000.0 + 0.8))) # No. of intercool, per compressor
                    #N_intercoolobj[k] = N_intercool

                    # Purchase cost
                    CP_intercool = (Ncomp)*(N_intercool * 10 ** (4.0336+ 0.2341 * np.log10(max(od_aircool*area_intercool  / N_intercool, 10.0))+ 0.0497 * (np.log10(max(od_aircool*area_intercool / N_intercool, 10.0))) ** 2))

                    CP_intercool_min = (Ncomp)*(N_intercool * 10 ** (4.0336+ 0.2341 * np.log10(10.0)+ 0.0497 * (np.log10(10.0)) ** 2))

                    FBM_intercool = 0.96 + 1.21 * 2.9   # SS construction

                    BM_intercool = sd * FBM_intercool * CP_intercool*CI
                    BM_0_intercool = CP_intercool*(0.96 + 1.21)*CI
                    
                    if (CP_intercool == CP_intercool_min) or (od_aircool*area_intercool  / N_intercool > 30000):
                        BM_intercool_min = CP_intercool_min * FBM_intercool * CI
                        BM_intercool = sd * BM_intercool_min * (od_aircool*area_intercool / 10.0) ** nhe
                        BM_0_intercool = CP_intercool_min*(0.96 + 1.21)*CI
                    #    self.xll += 1
                    
                    # aircooled refr condenser
                    T_cond = 273.15 + 50 #condenser saturation, ie 11C approach over ambient
                    T_refrcomp = 273.15 + 65 # refr comp outlet temp , 55 C is the actual comp o/l temp from the Freon thermo tables, at the condernser pressure, and the eta_isen of 0.6
                    Q_desup = delH_desup * m_refr_at_eta75
                    Q_cond_refr = refr_lat_heat_cond * m_refr_at_eta75
                    U_desup = 30
                    U_cond_refr = 40 # both are close, as airside controls HT

                    delTlm_cond_refr = ((T_cond - (T_amb+9)) - (T_cond - T_amb)) / np.log((T_cond - (T_amb+9)) / (T_cond - T_amb)) # Temp. rise of 1K is assumed across the desuperheater

                    delTlm_desup = ((T_refrcomp - (T_aircooler)) - (T_cond - (T_amb+9))) / np.log((T_refrcomp - (T_aircooler)) / (T_cond - (T_amb+9)))

                    area_desup = Q_desup / (U_desup * delTlm_desup*FF)
                    area_cond_refr = Q_cond_refr / (U_cond_refr * delTlm_cond_refr*FF)
                    area_refr = (area_desup + area_cond_refr) # total area of the air cooled refrigerant condenser

                    N_refr = max(1, int(np.floor(od_aircool*area_refr / 30000.0 + 0.5))) # max area for air cooler is 10000 m^2
                    CP_refr_min = N_refr * 10 ** (4.0336+ 0.2341 * np.log10(1.0)+ 0.0497 * (np.log10(1.0)) ** 2)

                    CP_refr = N_refr * 10 ** (4.0336+ 0.2341 * np.log10(max(od_aircool*area_refr / N_refr, 1.0))+ 0.0497 * (np.log10(max(od_aircool*area_refr / N_refr, 1.0))) ** 2)

                    Fp_refr = 10 ** (-0.125+ 0.15361 * np.log10(29.0 + np.finfo(float).eps)- 0.02861 * (np.log10(29.0 + np.finfo(float).eps)) ** 2) # 21 bar is the guage pressure of the refrigerant inside the vapor compression cycle

                    FBM_refr = 0.96 + 1.21 * 1.0 * Fp_refr   # CS construction, 30 bar pressure

                    BM_refr = sd * FBM_refr * CP_refr* CI
                    BM_0_refr = (0.96+1.21)*CP_refr*CI

                    if (CP_refr_min == CP_refr) or (od_aircool*area_refr / N_refr > 30000):
                        BM_refr_min = CP_refr_min * FBM_refr * CI
                        BM_refr = sd * BM_refr_min * (od_aircool*area_refr / 1.0) ** nhe
                        BM_0_refr = (0.96+1.21)*CP_refr_min*CI
                    #     self.xk += 1

                    # Air flowrate and fan power
                  
                    od_fans = 1.20
                    air_fan = intercool_comp / (delT_air * Cp_air * rho_air)
                    W_compfan = 0.001*air_fan * delP_fan / eta_fan
                    N_fan = max(1, int(np.floor(od_fans*air_fan / 100.0 + 0.8)))
                    CP_fan = N_fan * 10 ** (3.5391- 0.3533 * np.log10(max(od_fans*air_fan / N_fan, 0.05))+ 0.4477 * (np.log10(max(od_fans*air_fan / N_fan, 0.05))) ** 2)

                    CP_fan_min = N_fan * 10 ** (3.5391- 0.3533 * np.log10(0.05)+ 0.4477 * (np.log10(0.05)) ** 2)

                    FBM_fan = 2.7
                    BM_fan = sd * FBM_fan * CP_fan * CI
                    BM_0_fan = BM_fan / FBM_fan

                    if (CP_fan == CP_fan_min) or (od_fans*air_fan / N_fan > 110):
                        BM_fan_min = CP_fan_min * FBM_fan* CI
                        BM_fan = sd * BM_fan_min * (od_fans*air_fan / 0.05) ** nf
                        BM_0_fan = BM_fan / FBM_fan
                    #    self.xn += 1

                    air_refrfan = m_refr_at_eta75*(delH_desup+refr_lat_heat_cond)/(delT_air*Cp_air*rho_air)# %in m^3/s
                    W_refrfan = air_refrfan*delP_fan/eta_fan
                    # Number of refrigeration fans
                    N_refrfan = max(1, int(np.floor(od_fans*air_refrfan / 100.0 + 0.9)))

                    # Fan purchase cost
                    CP_refrfan = N_refrfan * 10 ** (3.5391- 0.3533 * np.log10(max(od_fans*air_refrfan / N_refrfan, 0.1))+ 0.4477 * (np.log10(max(od_fans*air_refrfan / N_refrfan, 0.1))) ** 2)

                    CP_refrfan_min = N_refrfan * 10 ** (3.5391- 0.3533 * np.log10(0.1)+ 0.4477 * (np.log10(0.1)) ** 2)

                    BM_refrfan = sd * FBM_fan * CP_refrfan* CI
                    BM_0_refrfan = BM_refrfan / FBM_fan

                    if (CP_refrfan == CP_refrfan_min) or (od_fans*air_refrfan / N_refrfan > 110):
                        BM_refrfan_min = CP_refrfan_min * FBM_fan* CI
                        BM_refrfan = sd * BM_refrfan_min * (od_fans*air_refrfan / 0.1) ** nf
                        BM_0_refrfan = BM_refrfan / FBM_fan
                    #    self.xm += 1

                    BM_f = BM_fan+BM_refrfan
                    BM_0_f = BM_0_fan+BM_0_refrfan
                    
                    # refr evaporator HX (shell and tube)
                    od_HX = 1.15 
                    Q_evap_refr = refr_lat_evap*m_refr_at_eta75
                    U_evap = 850 # OHTC in W/m^2K;  
                    delTlm_evap = ((T_cwr-T_evap) - (T_cws-T_evap))/(np.log((T_evap- T_cwr)/(T_evap-T_cws)))
                    area_evap = Q_evap_refr / (U_evap * delTlm_evap * FF)
                    N_evap = max(1, int(np.floor(od_HX*area_evap / 3000.0 + 0.9)))
                    

                    CP_evap_min = N_evap * 10 ** (4.1884- 0.2503 * np.log10(2)+ 0.1974 * (np.log10(2)) ** 2)

                    CP_evap = N_evap * 10 ** (4.1884- 0.2503 * np.log10(max(od_HX*area_evap / N_evap, 2))+ 0.1974 * (np.log10(max(od_HX*area_evap / N_evap, 2))) ** 2)

                    Fp_evap = 10 ** (0.03881- 0.11272 * np.log10(12.0 + np.finfo(float).eps)+ 0.08183 * (np.log10(12.0 + np.finfo(float).eps)) ** 2)

                    FBM_evap = 1.63 + 1.66 * 1.0 * Fp_evap

                    BM_evap = sd * FBM_evap * CP_evap * CI
                    BM_0_evap = (1.63 + 1.66)*CP_evap*CI

                    if (CP_evap == CP_evap_min) or (od_HX*area_evap / N_evap > 3000):
                        BM_evap_min = CP_evap_min * FBM_evap* CI
                        BM_evap = sd * BM_evap_min * (od_HX*area_evap / 2.0) ** nhe
                        BM_0_evap = (1.63 + 1.66)*CP_evap_min*CI
                      #   self.xjjj += 1
                    
                    ######################################### lye cooler HX (shell and tube) 
                    
                    #SEC_chiller_lyecooler_at_eta75 = interp_func5(cd_eta75)

                    #Q_lyecool_eta75 = od_HX*1000*SEC_chiller_lyecooler_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 # must be in watts
                    U_lyecooler = 280 #OHTC in W/m^2K; %%%%%%%%%%%Liquid to Liquid
                    
                    # Q_lyecooler_at_eta75

                    #T_angl_rated = interp_func9(cd_eta75)
                    #T_gl_rated = interp_func10(cd_eta75)

                    Q_catlyecool_eta75 = np.maximum((T_gl_out_at_eta75-273.15-80)*Cp_KOH_lye_at_eta75*Q_gl_out_at_eta75*rho_KOH_gl_out_at_eta75,0) 
                    Q_anlyecool_eta75 = np.maximum((T_angl_out_at_eta75-273.15-80)*Cp_KOH_anlye_at_eta75*Q_angl_out_at_eta75*rho_KOH_angl_out_at_eta75,0) 

                    #T_lyecooler_in = np.mean([T_angl_out_at_eta75, T_gl_out_at_eta75])

                    delTlm_catlyecooler_at_eta75 = ((T_gl_out_at_eta75 - T_cwr) - (353.15 - T_cws)) / np.log((T_gl_out_at_eta75 - T_cwr) / (353.15 - T_cws))
                    delTlm_anlyecooler_at_eta75 = ((T_angl_out_at_eta75- T_cwr) - (353.15 - T_cws)) / np.log((T_angl_out_at_eta75 - T_cwr) / (353.15 - T_cws))


                    area_catlyecooler_at_eta75 = Q_catlyecool_eta75 / (U_lyecooler * delTlm_catlyecooler_at_eta75 * FF)

                    area_anlyecooler_at_eta75 = Q_anlyecool_eta75 / (U_lyecooler * delTlm_anlyecooler_at_eta75 * FF)

                    area_lyecooler_at_eta75 = max(area_catlyecooler_at_eta75,area_anlyecooler_at_eta75 ) # the higher of the 2 areas is taken for sizing of the lye cooler HX
                    # calc. of lye pump power and capex
                    


                    L_tube = 6 # per pass
                    flow_area = np.pi*D_tube**2/4.0  # per tube flow area

                    N_tube_lyecooler = np.ceil(od_HX*area_lyecooler_at_eta75/(np.pi*D_tube*L_tube))  #/N_lye # Total No. of tubes per lyecool HX , total number of 2 passes
                    #N_tube_anlyecooler = np.ceil(area_anlyecooler_at_eta75/(np.pi*D_tube*L_tube)) 

                    N_pass = 2 # number of passes
                    N_tube_pass_lyecooler = N_tube_lyecooler/N_pass # calc. number of parallel tubes per pass
                    #N_tube_pass_anlyecooler = N_tube_anlyecooler/N_pass # calc. number of parallel tubes per pass

                    area_lyecooler_flow_pass = N_tube_pass_lyecooler* flow_area
                    #area_anlyecooler_flow_pass = N_tube_pass_anlyecooler* flow_area

                    #W_pump75 = m_refr_at_eta75*(0.4)/eta_pump
                                      
                    vel_tubes_lyecooler = np.maximum(Q_gl_out_at_eta75,Q_angl_out_at_eta75) /area_lyecooler_flow_pass # velocity of lye in HX tubes at rated cc
                    #vel_tubes_anlyecooler = (Q_angl_out) /area_anlyecooler_flow_pass

                    vel_ratio = vel_tubes_lyecooler/V_target
                    #vel_anratio = vel_tubes_anlyecooler/V_target
                    N_tube_pass_lyecooler = N_tube_pass_lyecooler*np.maximum(vel_ratio,1)
                    area_lyecooler_flow_pass = N_tube_pass_lyecooler* flow_area
                    area_lyecooler_at_eta75 = (np.pi*D_tube*L_tube)*N_tube_pass_lyecooler*N_pass
                    N_tube_lyecooler = N_tube_pass_lyecooler*N_pass
                    N_lye = max(1, int(np.floor(area_lyecooler_at_eta75 / 1000 + 0.5)))# No. of lyecoolers per circuit, the no. of lyecoolers based on the max alloable HT area is 1000 m^2, velocity ratio is also equated to the N_lye
                    #N_anlye = max(1, int(np.floor(area_anlyecooler_at_eta75 / 1000 + 0.9)),np.floor(vel_anratio))# No. of lyecoolers per circuit, the no. of lyecoolers based on the max alloable HT area is 1000 m^2, velocity ratio is also equated to the N_lye

                    vel_tubes_lyecooler_eta75 = (np.maximum(Q_gl_out_at_eta75,Q_angl_out_at_eta75)) /area_lyecooler_flow_pass # recalculate the velocity in cathode lyecooler tubes after calc. of N_lye  , at rated operation.
                    vel_tubes_lyecooler = (np.maximum(Q_gl_out,Q_angl_out)) /area_lyecooler_flow_pass
                    
                    #vel_tubes_anlyecooler_eta75  = (Q_anlyecool_eta75/N_lye) /area_anlyecooler_flow_pass # velocity in anode lyecooler tubes at candidate cd


                    
                    


                    
                    
                    #N_lye = 2*N_lye
                    #area_lyecooler_at_eta75  = area_catlyecooler_at_eta75 + area_anlyecooler_at_eta75

                    CP_lyecooler = 2*N_lye * 10 ** (4.1884- 0.2503 * np.log10(max(2*area_lyecooler_at_eta75 /(2* N_lye), 2.0))+ 0.1974 * (np.log10(max(2*area_lyecooler_at_eta75 / (2*N_lye), 2.0))) ** 2)

                    CP_lyecooler_min = 2*N_lye * 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

                    eps = np.finfo(float).eps
                    Fp_HX = (1.0 * (Pg == 0)+ (Pg > 0)* ((Pg < 5)+ (Pg >= 5)* 10 ** (0.03881- 0.11272 * np.log10(Pg + eps)+ 0.08183 * (np.log10(Pg + eps)) ** 2)))

                    FBM_lyecool = 1.63 + 1.66 * 1.8 * Fp_HX
                    BM_lyecool = sd * FBM_lyecool * CP_lyecooler* CI
                    BM_0_lyecool = CP_lyecooler*(1.63+1.66)*CI

                    if (CP_lyecooler == CP_lyecooler_min) or (area_lyecooler_at_eta75 / (N_lye) > 3000):
                        BM_lyecool_min = CP_lyecooler_min * FBM_lyecool* CI
                        BM_lyecool = sd * BM_lyecool_min * (area_lyecooler_at_eta75/ 2.0) ** nhe
                        BM_0_lyecool = CP_lyecooler_min*(1.63+1.66)*CI
                    #    self.xiii += 1

                    # H2cooler, ads_cooler, deoxo_cooler all shell and tube

                    #SEC_chiller_h2cooler_eta75 = interp_func4(cd_eta75)

                    #Q_h2cool_eta75 = od_HX*1000*SEC_chiller_h2cooler_eta75* vap_h2_pdt_eta75 * MH2 * 3600 # must be in watts

                    #Q_cond_ads_eta75 = od_HX*interp_func6(cd_eta75) # must be in watts

                    #Q_cond_deoxo_eta75 = od_HX*interp_func7(cd_eta75) # must be in watts

                    #Calculation of T_reg at eta75
                    dry_gas_at_eta75 = vap_h2_pdt_eta75*(1-adspec_H2O) #dry gas flowrate at rated h2 production rate
                    vap_ads_at_eta75 = vap_h2_pdt_eta75 * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber

                    ads_H2O_in_at_eta75 = (dry_gas_at_eta75*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
                    ads_H2O_out_at_eta75 = (vap_ads_at_eta75*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;

                    ads_cap_rate_at_eta75 = ads_H2O_in_at_eta75-ads_H2O_out_at_eta75 #; % rate of moisture capture required
                    qcc_at_eta75 = ads_cap_rate_at_eta75*t_b/(m_ads*eta_ads) #; % anticipated ads cc density in mol/kg
                    ads_cc_at_eta75 = m_ads*qcc_at_eta75*eta_ads #; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required
                    ads_cap_rate_at_eta75 = ads_cc_at_eta75/t_b
                    q_res_at_eta75 = q_max-qcc_at_eta75 # excess residual capacity above the reqd capture rate
                    rq_at_eta75 = q_res_at_eta75 / q_max # fraction of reduction in capacity after reqd capture capacity of the adsorbent
                    des_rate_at_eta75 = ads_cc_at_eta75 / t_des #; % the reqd des_rate increases with current density as more mositure is in gas stream.

                    rp_at_eta75 = purge_ads/(des_rate_at_eta75) #; % ratio of purge flow rate to des rate (based on desorb time)

                    xH2O_reg_out_at_eta75 = (x_H2O_h2cool_out+1.0/rp_at_eta75)/(1.0+1.0/rp_at_eta75) #; % mositure moefrac at regen exit

                    p_H2O_reg_out_at_eta75 = x[1]*10**5*(x_H2O_h2cool_out+1.0/rp_at_eta75)/(1.0+1.0/rp_at_eta75)

                    p_vap_reg_eff_at_eta75 = x[1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out_at_eta75-x_H2O_h2cool_out))

                    b_reg_at_eta75 = ((rq_at_eta75**0.2472)/(1.0-rq_at_eta75**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff_at_eta75)

                    T_reg_at_eta75 = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg_at_eta75/b_ref))**-1) # T_reg is much more sensitive to the rq (residual ads cc after required adsorption as per specifications) than the purge gas circulation rate.

                    T_base_vec_eta_75 = T_glsepHXout #* np.ones(len(T_reg_at_eta75))   # replicate as in MATLAB

                    thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

                    m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)

                    heater_ads_at_eta75 = 0.001*(purge_ads * Cp_H2_80 * (T_reg_at_eta75 - T_base_vec_eta_75) + des_rate_at_eta75 * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg_at_eta75- T_base_vec_eta_75) / t_des / eta_heater) * t_des / t_b)

                    #condensate_H2O_regen_cond = (purge_ads * (1.0 - x_H2O_h2cool_out) * xH2O_reg_out / (1.0 - xH2O_reg_out)- x_H2O_h2cool_out * purge_ads)

                    #cond_ads = (condensate_H2O_regen_cond*lat_heat_water+ purge_ads * Cp_H2_80 * (T_reg - T_glsepHXout)) * t_des / t_b 

                    #m_cw_cond_ads = cond_ads / (Cp_cw * (T_cwr - T_cws))




                    U_purif= 60# %%%%%%%%%%%%%%% LIQUID TO GAS is 60, common for these HX

                    # LMTDs
                    delTlm_h2cool_at_eta75 = ((T_gl_out_at_eta75 - T_cwr) - (T_amb - T_cws)) / np.log((T_gl_out_at_eta75 - T_cwr) / (T_amb - T_cws))

                    delTlm_deoxo_at_eta75 = ((T_deoxo_out - T_cwr) - (T_amb - T_cws)) / np.log((T_deoxo_out - T_cwr) / (T_amb - T_cws))

                    delTlm_des_cool_at_eta75 = ((T_reg_at_eta75 - T_cwr) - (T_amb - T_cws)) / np.log((T_reg_at_eta75 - T_cwr) / (T_amb - T_cws))

                    # HT Areas, shell dias, cross-sectional areas etc
                    area_h2cool_at_eta75      = od_HX*Q_cond_h2cooler_at_eta75 / (U_purif * delTlm_h2cool_at_eta75 * FF)
                    area_deoxo_cool_at_eta75  = od_HX*Q_cond_deoxo_at_eta75 / (U_purif * delTlm_deoxo_at_eta75 * FF)
                    area_des_cool_at_eta75    =  od_HX*Q_cond_ads_at_eta75 / (U_purif * delTlm_des_cool_at_eta75 * FF)

                    N_tube_h2cooler = np.ceil(area_h2cool_at_eta75/(np.pi*D_tube*L_tube))
                    N_tube_deoxocool = np.ceil(area_deoxo_cool_at_eta75/(np.pi*D_tube*L_tube))  
                    N_tube_descool = np.ceil(area_des_cool_at_eta75/(np.pi*D_tube*L_tube))

                    Ds_h2cooler = ((4*N_tube_h2cooler*(3/4)**0.5*pt**2)/np.pi)**0.5 # shell dia of h2 cooler estimate for delP calculations, based on a vornoi cell area belonging and surrounding each tube
                    Ds_deoxocool = ((4*N_tube_deoxocool*(3/4)**0.5*pt**2)/np.pi)**0.5
                    Ds_descool = ((4*N_tube_descool*(3/4)**0.5*pt**2)/np.pi)**0.5
                    Ds_lyecool = ((4*N_tube_lyecooler*(3/4)**0.5*pt**2)/np.pi)**0.5
                    
                    As_h2cooler =   Ds_h2cooler*(0.4*Ds_h2cooler)*(pt-D_tube)/pt
                    As_deoxocool =   Ds_deoxocool*(0.4*Ds_deoxocool)*(pt-D_tube)/pt
                    As_descool =   Ds_descool*(0.4*Ds_descool)*(pt-D_tube)/pt
                    As_lyecool =   Ds_lyecool*(0.4*Ds_lyecool)*(pt-D_tube)/pt

                    ##### CW circ power calc.
                    m_circ_lyecool_at_eta75 = (1/(2*N_lye))*Q_lyecooler_at_eta75/(Cp_cw*(T_cwr-T_cws)) # per lye cooler unit
                    m_circ_h2cooler_at_eta75 = Q_cond_h2cooler_at_eta75/(Cp_cw*(T_cwr-T_cws))  
                    m_circ_deoxocool_at_eta75 = Q_cond_deoxo_at_eta75/(Cp_cw*(T_cwr-T_cws))
                    m_circ_descool_at_eta75 = Q_cond_ads_at_eta75/(Cp_cw*(T_cwr-T_cws))

                    m_circ_lyecool = (1/(2*N_lye))*Q_lyecooler/(Cp_cw*(T_cwr-T_cws)) # per lye cooler
                    m_circ_h2cooler = Q_cond_h2cooler/(Cp_cw*(T_cwr-T_cws))  
                    m_circ_deoxocool = Q_cond_deoxo/(Cp_cw*(T_cwr-T_cws))
                    m_circ_descool = Q_cond_ads/(Cp_cw*(T_cwr-T_cws))


                    Re_lyecool_at_eta75 = (m_circ_lyecool_at_eta75/As_lyecool )*de/mu_cw
                    Re_h2cooler_at_eta75 = (m_circ_h2cooler_at_eta75/As_h2cooler  )*de/mu_cw
                    Re_deoxocool_at_eta75 = (m_circ_deoxocool_at_eta75/As_deoxocool )*de/mu_cw
                    Re_descool_at_eta75 = (m_circ_descool_at_eta75/As_descool )*de/mu_cw
                    
                    Re_lyecool = (m_circ_lyecool/As_lyecool )*de/mu_cw
                    Re_h2cooler = (m_circ_h2cooler/As_h2cooler  )*de/mu_cw
                    Re_deoxocool = (m_circ_deoxocool/As_deoxocool )*de/mu_cw
                    Re_descool = (m_circ_descool/As_descool )*de/mu_cw

                    shell_delP_lyecool_at_eta75 = (L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(np.exp(0.576 - 0.19 * np.log(Re_lyecool_at_eta75))*(m_circ_lyecool_at_eta75/As_lyecool )**2/(2*9.81*rho_cw*de))*N_lye*2
                    shell_delP_h2cooler_at_eta75 = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(np.exp(0.576 - 0.19 * np.log(Re_h2cooler_at_eta75))*(m_circ_h2cooler_at_eta75/As_h2cooler  )**2/(2*9.81*rho_cw*de))
                    shell_delP_deoxocool_at_eta75 = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(np.exp(0.576 - 0.19 * np.log(Re_deoxocool_at_eta75))*(m_circ_deoxocool_at_eta75 /As_deoxocool )**2/(2*9.81*rho_cw*de))
                    shell_delP_descool_at_eta75 = (L_tube/(0.4*Ds_descool))*Ds_descool*(np.exp(0.576 - 0.19 * np.log(Re_descool_at_eta75))*(m_circ_descool_at_eta75 /As_descool )**2/(2*9.81*rho_cw*de))

                    shell_delP_lyecool = (L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(np.exp(0.576 - 0.19 * np.log(np.maximum(Re_lyecool,1e-06)))*(m_circ_lyecool/As_lyecool )**2/(2*9.81*rho_cw*de))*N_lye*2
                    
                    shell_delP_h2cooler = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(np.exp(0.576 - 0.19 * np.log(Re_h2cooler))*(m_circ_h2cooler/As_h2cooler  )**2/(2*9.81*rho_cw*de))
                    shell_delP_deoxocool = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(np.exp(0.576 - 0.19 * np.log(Re_deoxocool))*(m_circ_deoxocool /As_deoxocool )**2/(2*9.81*rho_cw*de))
                    shell_delP_descool = (L_tube/(0.4*Ds_descool))*Ds_descool*(np.exp(0.576 - 0.19 * np.log(Re_descool))*(m_circ_descool /As_descool )**2/(2*9.81*rho_cw*de))  



                    delP_cw_circ_at_eta75 = (shell_delP_lyecool_at_eta75+ shell_delP_h2cooler_at_eta75 + shell_delP_deoxocool_at_eta75 + shell_delP_descool_at_eta75)*1.2
                    delP_cw_circ = (shell_delP_lyecool+ shell_delP_h2cooler + shell_delP_deoxocool + shell_delP_descool)*1.2

                    #Number of H2 coolers
                    N_h2cool = max(1, int(np.floor(area_h2cool_at_eta75 / 3000.0 + 0.5)))


                    # Cost correlations
                    CP_h2cool = N_h2cool * 10 ** (4.1884- 0.2503 * np.log10(max(area_h2cool_at_eta75 / N_h2cool, 2.0))+ 0.1974 * (np.log10(max(area_h2cool_at_eta75  / N_h2cool, 2.0))) ** 2)

                    CP_h2cool_min = N_h2cool * 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

                    CP_deoxo_cool = 10 ** (4.1884- 0.2503 * np.log10(max(area_deoxo_cool_at_eta75, 2.0))+ 0.1974 * (np.log10(max(area_deoxo_cool_at_eta75, 2.0))) ** 2)

                    CP_deoxo_min_cool = 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

                    CP_des_cool = 10 ** (4.1884- 0.2503 * np.log10(max(area_des_cool_at_eta75, 2.0))+ 0.1974 * (np.log10(max(area_des_cool_at_eta75, 2.0))) ** 2)

                    CP_des_cool_min = 10 ** (4.1884- 0.2503 * np.log10(2.0)+ 0.1974 * (np.log10(2.0)) ** 2)

                    # Bounds check
                    #if (area_h2cool_at_eta75   / N_h2cool) > 1000 or area_des_cool_at_eta75 > 1000 or area_deoxo_cool_at_eta75 > 1000:
                    #    raise ValueError("HX area above correlation upper bound")

                    # Bare-module costs
                    FBM_purif = 1.63 + 1.66 * 1.8 * Fp_HX   # SS tube, CS shell

                    BM_h2cool   = sd * FBM_purif * CP_h2cool * CI
                    BM_0_h2cool = sd *(1.63 + 1.66) * CP_h2cool * CI
                    BM_deoxocool    = sd * FBM_purif * CP_deoxo_cool * CI
                    BM_0_deoxocool = sd * (1.63 + 1.66) * CP_deoxo_cool * CI
                    BM_des_cool = sd * FBM_purif * CP_des_cool * CI
                    BM_0_des_cool = sd * (1.63 + 1.66)* CP_des_cool * CI

                    #BM_purif   = BM_h2cool + BM_deoxo + BM_des_cool
                    #BM_0_purif = (BM_h2cool + BM_deoxo + BM_des_cool) * CI / FBM_purif
                  
                          ############################# PUMPS (ALL CENTRRIFUGAL due to high flowrate, calculation of energy and capex based on total HX pressure drop and fittings)

                    od_pump = 1.15 # 15 % overdeisgn factor for pumps
                    eta_pump =0.6
                    ###### lye cooler tube side delP and lye circ pump power calc.
                    Re_tube = 1260*  vel_tubes_lyecooler*D_tube/0.00093 # Re number of tubes inside the HX
                    ff = (0.316*Re_tube**-0.25) # friction factor for smooth tubes turbulent flow , Blasius equation
                    Re_tube_at_eta75 = 1260*  vel_tubes_lyecooler_eta75*D_tube/0.00093 # Re number of tubes inside the HX
                    ff_at_eta75 = (0.316*Re_tube_at_eta75**-0.25)

                    delP_catlyecooler = (1+alpha_delPHX)*ff*L_tube*1260*vel_tubes_lyecooler**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
                    delP_catlyect  = (1+alpha_delPlyect)*delP_catlyecooler # total pressure drop in individual circuits

                    delP_catlyecooler_at_eta75 = (1+alpha_delPHX)*ff_at_eta75 *L_tube*1260*vel_tubes_lyecooler_eta75**2/(2*D_tube)*N_lye
                    delP_catlyect_at_eta75 = (1+alpha_delPlyect)*delP_catlyecooler_at_eta75 
                    
                    delP_anlyecooler = (1+alpha_delPHX)*ff*L_tube*1260*vel_tubes_lyecooler**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
                    delP_anlyect  = (1+alpha_delPlyect)*delP_anlyecooler # total pressure drop in individual circuits

                    delP_anlyecooler_at_eta75 = (1+alpha_delPHX)*ff_at_eta75 *L_tube*1260*vel_tubes_lyecooler_eta75**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
                    delP_anlyect_at_eta75  = (1+alpha_delPlyect)*delP_anlyecooler_at_eta75 # total pressure drop in individual circuits  

                    W_catlyepump =  0.001*(delP_catlyect + cell_delP)*Q_gl_out /eta_pump 

                    W_anlyepump_eta75 = 0.001*(delP_anlyect_at_eta75 + ancell_delP_at_eta75)*Q_angl_out_at_eta75 /eta_pump # in kW

                    W_anlyepump =   0.001*(delP_anlyect + ancell_delP)*Q_angl_out /eta_pump 

                    W_catlyepump_eta75 = 0.001*(delP_catlyect_at_eta75 + cell_delP_at_eta75)*Q_gl_out_at_eta75 /eta_pump # in kW

                    W_lyepump75 = np.maximum(W_anlyepump_eta75 , W_catlyepump_eta75) # maximum value is selected for lye pump sizing, in kW

                    W_cwpump75 = 0.001*delP_cw_circ_at_eta75*(m_cw_circ_at_eta75/rho_cw)/eta_pump    # in kW  

                    W_cwpump = 0.001*delP_cw_circ*(m_cw_circ/rho_cw)/eta_pump

                    W_pump =  (W_anlyepump + W_catlyepump+ W_cwpump)# total pumping power in kW

                    W_pump75 = (W_anlyepump_eta75 + W_catlyepump_eta75 + W_cwpump75)
                    #W_pump75 = od_pump*SEC_pump_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 # unit is kW

                    if W_lyepump75 > 0:
                    # Pump cost
                      CP_pump_lye = 2*10 ** (3.3892
                                  + 0.0536 * np.log10(max(od_pump*W_lyepump75, 0.05))
                                  + 0.1538 * (np.log10(max(od_pump*W_lyepump75, 0.05)))**2) # factor of 2 because of 2 identical lye pumps

                      CP_pump_lye_min = 2*10 ** (3.3892
                                      + 0.0536 * np.log10(0.05)
                                      + 0.1538 * (np.log10(0.05))**2)   # 1 kW

                      FBM_pump_lye = 1.89 + 1.35 * (np.mean([2.3, 1.4]) *
                                            ((x[1] < 11) * 1 + (x[1] > 11) * 1.157))

                      BM_lyepumps = sd * CP_pump_lye * FBM_pump_lye * CI
                      BM_0_lyepumps = CP_pump_lye*(1.89+1.35)* CI
                    

                    # apply 6/10 rule if min. cc > calc. cc
                      if (CP_pump_lye == CP_pump_lye_min) or (od_pump*W_lyepump75 > 500):
                        BM_lyepumps_min = CP_pump_lye_min * FBM_pump_lye* CI
                        BM_lyepumps = sd * BM_lyepumps_min * (od_pump*W_lyepump75 / 0.05) ** 0.6
                        BM_0_lyepumps = CP_pump_lye_min*(1.89+1.35)* CI
                        #xa += 1
                    else:
                      BM_lyepumps = BM_0_lyepumps =0

                    if W_cwpump75 > 0:
                    # Pump cost
                      CP_cwpump = 10 ** (3.3892
                                  + 0.0536 * np.log10(max(od_pump*W_cwpump75, 0.05))
                                  + 0.1538 * (np.log10(max(od_pump*W_cwpump75, 0.05)))**2) # factor of 2 because of 2 identical lye pumps

                      CP_cwpump_min = 10 ** (3.3892
                                      + 0.0536 * np.log10(0.05)
                                      + 0.1538 * (np.log10(0.05))**2)   # 1 kW

                      FBM_cwpump = 1.89 + 1.35 * (np.mean([2.3, 1.4]) *
                                            ((x[1] < 11) * 1 + (x[1] > 11) * 1.157))

                      BM_cwpump = sd * CP_cwpump * FBM_cwpump * CI
                      BM_0_cwpumps = CP_cwpump*(1.89+1.35)* CI
                    

                    # apply 6/10 rule if min. cc > calc. cc
                      if (CP_cwpump == CP_cwpump_min) or (od_pump*W_cwpump75> 500):
                        BM_cwpump_min = CP_cwpump_min * FBM_cwpump* CI
                        BM_cwpump = sd * BM_cwpump_min * (od_pump*W_cwpump75 / 0.05) ** 0.6
                        BM_0_cwpumps = CP_cwpump_min*(1.89+1.35)* CI
                        #xa += 1
                    else:
                      BM_cwpump = BM_0_cwpumps = 0  
                      BM_0_pumps = BM_0_cwpumps + BM_0_lyepumps
                    ################################################################################### calc. of Wsys and SEC_sys ###################################################################################################

                    # ===== INITIALIZE W_sys75 =====
                    W_sys75 = 0.0

                          
                          # ===== COMPUTE W_sys75 HERE (inside valid condition) =====


                    #W_sys75 = SEC_sys_at_eta75 * vap_h2_pdt_eta75 * MH2 * 3600 # rated input power to system, 'rated' therefore no consdieration of degradation in the calculation
                    #W_sys_peak = SEC_sys_at_peak * vap_h2_pdt_peak * MH2 * 3600 # peak allowable input power to system, 10% above the rated curr dens.

                    W_sys75 = W_pump75 + SEC_stack_at_eta75* vap_h2_pdt_eta75 * MH2 * 3600 + (heater_ads_at_eta75 + heater_deoxo_at_eta75 + heater_regen + Q_lyeheater_at_eta75)/eta_heater + W_h2comp + W_refrcomp_at_eta75 + W_refrfan_at_eta75 + W_compfan
                    W_sys75 = W_sys75/eta_pow
                    pareto_W_sys75[i] = W_sys75
                    W_sys = W_pump + SEC_stack* vap_h2_pdt * MH2 * 3600 + (heater_ads + heater_deoxo + heater_regen + Q_lyeheater)/eta_heater + W_refrcomp_SEC + W_refrfan_SEC + W_comp_SEC + W_fan_SEC
                    W_sys = W_sys/eta_pow  
                    W_sys_exstack = W_pump + (heater_ads + heater_deoxo + heater_regen + Q_lyeheater)/eta_heater + W_refrcomp_SEC + W_refrfan_SEC + W_comp_SEC + W_fan_SEC # system power draw ex of stack, ie only BOP
                          # ========================================================  

                    Pow_lyeheater =  heater_frac_ratedpower*W_sys75*1000 # rated lye heater power is heater_frac_ratedpower% of the W_sys75 in Watt
                    P_rated_el   = eta_heater * Pow_lyeheater# actula heat delivered to lye in the heater
                    BM_heaters = 7.5 * 3*W_sys75      # 7.5 $/kW, %7.5$/kW , 3 heaters in B.O.P,Grid connected Hydrogen production via large-scale water electrolysis{Nguyen, 2019 #186}, W_sys75 is in kW, od = 1.15
                    BM_heaters_lye = Pow_lyeheater*0.001*150*(800/619) # heaters have to be sized higher when the AWE is totally reliant on renewable power compared to when a battery is available, so as  to maximize the full load hours ( only when renew power is used), and increase load factor. $150/kW
                    

                    BM_pow = 199*W_sys75            # 199 $/kW, power electronics includes buck converter, transfomer, DC to DC converter, 1.15 over design factor
                    

                    
                    
                
                    

                    #------ calc. of t_oper based from actual PV+wind hybrid generation profiles-------
                    
                    #f_min = 0.3; #10% load factor
                    min_load = cdeta_fixed*cd_eta75
              

                    #SEC_sys_min_load = interp_SEC_sys(min_load)
                    #SEC_sys_min_load1 = interp_func1(min_load1) # system SEC at min. load
                    vap_h2_pdt_minload = interp_vap_h2_pdt(min_load)

                    ################ calc. of W_sys at min. load ####################
                    T_gl_out_minload = interp_T_gl_out(min_load)  
                    T_angl_out_minload = interp_T_angl_out(min_load)
                    w_KOH_gl_out_minload = interp_w_KOH_gl_out(min_load)
                    w_KOH_angl_out_minload = interp_w_KOH_angl_out(min_load)

                    Cp_KOH_lye_minload = (4.101*10**3-3.526*10**3*w_KOH_gl_out_minload  + 9.644*10**-1*(T_gl_out_minload-273.15)+1.776*(T_gl_out_minload-273.15)*w_KOH_gl_out_minload)
                    Cp_KOH_anlye_minload = (4.101*10**3-3.526*10**3*w_KOH_angl_out_minload + 9.644*10**-1*(T_angl_out_minload-273.15)+1.776*(T_angl_out_minload-273.15)*w_KOH_angl_out_minload)
                    Q_gl_out_minload      = interp_Q_gl_out(min_load)
                    Q_angl_out_minload      = interp_Q_angl_out(min_load)
                    vel_tubes_lyecooler_minload  = (np.maximum(Q_gl_out_minload,Q_angl_out_minload)/N_lye)/area_lyecooler_flow_pass # use maximum of the flowrate
                    rho_KOH_gl_out_minload = (1001.53 - 0.08343*(T_gl_out_minload - 273.15) - 0.004*(T_gl_out_minload-273.15)**2 + 5.51232*10**-6*(T_gl_out_minload-273.15)**3 - 8.21*10**-10*(T_gl_out_minload-273.15)**4)*np.exp(0.86*w_KOH_gl_out_minload)
                    rho_KOH_angl_out_minload = (1001.53 - 0.08343*(T_angl_out_minload - 273.15) - 0.004*(T_angl_out_minload-273.15)**2 + 5.51232*10**-6*(T_angl_out_minload-273.15)**3 - 8.21*10**-10*(T_angl_out_minload-273.15)**4)*np.exp(0.86*w_KOH_angl_out_minload)

                    Re_tube_minload = np.maximum(rho_KOH_gl_out_minload,rho_KOH_angl_out_minload)*  vel_tubes_lyecooler_minload*D_tube/0.00093 # Re number of tubes inside the HX
                    ff_minload = (0.316*Re_tube_minload**-0.25) # friction factor for smooth tubes turbulent flow , Blasius equation

                    delP_catlyecooler_minload = (1+alpha_delPHX)*ff_minload*L_tube*rho_KOH_gl_out_minload*vel_tubes_lyecooler_minload**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
                    delP_catlyect_minload  = (1+alpha_delPlyect)*delP_catlyecooler_minload # total pressure drop in individual circuits
                    
                    delP_anlyecooler_minload = (1+alpha_delPHX)*ff_minload*L_tube*rho_KOH_angl_out_minload*vel_tubes_lyecooler_minload**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
                    delP_anlyect_minload  = (1+alpha_delPlyect)*delP_anlyecooler_minload # total pressure drop in individual circuits

                    cell_delP_minload = interp_cell_delP(min_load)
                    ancell_delP_minload = interp_ancell_delP(min_load)

                    W_catlyepump_minload =  0.001*(delP_catlyect_minload + cell_delP_minload)*Q_gl_out_minload /eta_pump 
                    W_anlyepump_minload =   0.001*(delP_anlyect_minload + ancell_delP_minload)*Q_angl_out_minload /eta_pump 



                    Q_lyecooler_minload = np.maximum((T_gl_out_minload-273.15-80)*Cp_KOH_lye_minload*Q_gl_out_minload*rho_KOH_gl_out_minload,0)  + np.maximum((T_angl_out_minload-273.15-80)*Cp_KOH_anlye_minload*Q_angl_out_minload*rho_KOH_angl_out_minload,0) # lye cooler duty at candidate load
                    Q_cond_h2cooler_minload = interp_Q_cond_h2cooler(min_load)
                    Q_cond_deoxo_minload = interp_Q_cond_deoxo(min_load)
                    Q_cond_ads_minload = interp_Q_cond_ads(min_load)

                    m_circ_lyecool_minload = 0.5*Q_lyecooler_minload/(Cp_cw*(T_cwr-T_cws)) # per lye cooler
                    m_circ_h2cooler_minload = Q_cond_h2cooler_minload/(Cp_cw*(T_cwr-T_cws))  
                    m_circ_deoxocool_minload = Q_cond_deoxo_minload/(Cp_cw*(T_cwr-T_cws))
                    m_circ_descool_minload = Q_cond_ads_minload/(Cp_cw*(T_cwr-T_cws))      
                    Re_lyecool_minload = (m_circ_lyecool_minload/As_lyecool )*de/mu_cw
                    Re_h2cooler_minload = (m_circ_h2cooler_minload/As_h2cooler  )*de/mu_cw
                    Re_deoxocool_minload = (m_circ_deoxocool_minload/As_deoxocool )*de/mu_cw
                    Re_descool_minload = (m_circ_descool_minload/As_descool )*de/mu_cw

                    shell_delP_lyecool_minload = (L_tube/(0.4*Ds_lyecool))*Ds_lyecool*(np.exp(0.576 - 0.19 * np.log(np.maximum(Re_lyecool_minload,1e-06)))*(m_circ_lyecool_minload/As_lyecool )**2/(2*9.81*rho_cw*de))

                    shell_delP_h2cooler_minload = (L_tube/(0.4*Ds_h2cooler))*Ds_h2cooler*(np.exp(0.576 - 0.19 * np.log(Re_h2cooler_minload))*(m_circ_h2cooler_minload/As_h2cooler  )**2/(2*9.81*rho_cw*de))

                    shell_delP_deoxocool_minload = (L_tube/(0.4*Ds_deoxocool))*Ds_deoxocool*(np.exp(0.576 - 0.19 * np.log(Re_deoxocool_minload))*(m_circ_deoxocool_minload /As_deoxocool )**2/(2*9.81*rho_cw*de))
                    shell_delP_descool_minload = (L_tube/(0.4*Ds_descool))*Ds_descool*(np.exp(0.576 - 0.19 * np.log(Re_descool_minload))*(m_circ_descool_minload /As_descool )**2/(2*9.81*rho_cw*de))  

                    delP_cw_circ_minload = (shell_delP_lyecool_minload + shell_delP_h2cooler_minload + shell_delP_deoxocool_minload + shell_delP_descool_minload)*1.2
                    #W_catlyepump_eta75 = 0.001*(delP_catlyect_at_eta75 + cell_delP_at_eta75)*Q_gl_out_at_eta75 /eta_pump # in kW


                    W_cwpump_minload = 0.001*delP_cw_circ_minload*(m_cw_circ/rho_cw)/eta_pump

                    W_pump_minload =  (W_anlyepump_minload + W_catlyepump_minload+ W_cwpump_minload)# total pumping power in kW
                    #Calculation of T_reg at eta75
                    dry_gas_at_minload = vap_h2_pdt_minload*(1-adspec_H2O) #dry gas flowrate at rated h2 production rate
                    vap_ads_at_minload = vap_h2_pdt_minload * (1 - adspec_H2O) / (1 - x_H2O_h2cool_out) # vapor rate into adsorber
                    ads_H2O_in_at_minload = (dry_gas_at_minload*x_H2O_h2cool_out)/(1-x_H2O_h2cool_out)  #% water to adsorber;
                    ads_H2O_out_at_minload = (vap_ads_at_minload*adspec_H2O)/(1-adspec_H2O) # % moisture out of adsorber;
                    ads_cap_rate_at_minload = ads_H2O_in_at_minload-ads_H2O_out_at_minload #; % rate of moisture capture required
                    qcc_at_minload = ads_cap_rate_at_minload*t_b/(m_ads*eta_ads) #; % anticipated ads cc density in mol/kg
                    ads_cc_at_minload = m_ads*qcc_at_minload*eta_ads #; % adsorption cc of adsorbent, or moisture deposited onto adsorbent to get the spec required
                    ads_cap_rate_at_minload = ads_cc_at_minload/t_b
                    q_res_at_minload = q_max-qcc_at_minload # excess residual capacity above the reqd capture rate
                    rq_at_minload = q_res_at_minload / q_max # fraction of reduction in capacity after reqd capture capacity of the adsorbent
                    des_rate_at_minload = ads_cc_at_minload / t_des #; % the reqd des_rate increases with current density as more mositure is in gas stream.
                    rp_at_minload = purge_ads/(des_rate_at_minload) #; % ratio of purge flow rate to des rate (based on desorb time)
                    xH2O_reg_out_at_minload = (x_H2O_h2cool_out+1.0/rp_at_minload)/(1.0+1.0/rp_at_minload) #; % mositure moefrac at regen exit
                    p_H2O_reg_out_at_minload = x[1]*10**5*(x_H2O_h2cool_out+1.0/rp_at_minload)/(1.0+1.0/rp_at_minload)
                    p_vap_reg_eff_at_minload = x[1]*10**5*(x_H2O_h2cool_out+alpha_purge*(xH2O_reg_out_at_minload-x_H2O_h2cool_out))
                    b_reg_at_minload = ((rq_at_minload**0.2472)/(1.0-rq_at_minload**0.2472))**(1.0/0.2472)*(1.0/p_vap_reg_eff_at_minload)
                    T_reg_at_minload = np.maximum(T_glsepHXout+75.0,(1.0/(293.15)+R_const/H_ads*np.log(b_reg_at_minload/b_ref))**-1) # T_reg is much more sensitive to the rq (residual ads cc after required adsorption as per specifications) than the purge gas circulation rate.
                    T_base_vec_minload = T_glsepHXout #* np.ones(len(T_reg_at_eta75))   # replicate as in MATLAB
                    thick_ads = max(((Pg + 1) * ads_D/ (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness
                    m_steel_ads = ((np.pi * (ads_D / 2 + thick_ads) ** 2 - np.pi * (ads_D / 2) ** 2)* L_ads* rho_steel* 1.5)
                    heater_ads_minload = 0.001*(purge_ads * Cp_H2_80 * (T_reg_at_minload - T_base_vec_minload) + des_rate_at_minload * H_ads + (m_steel_ads * Cp_steel + m_ads * Cp_ads)* ((T_reg_at_minload- T_base_vec_minload) / t_des / eta_heater) * t_des / t_b)
                    glsep_O2_minload = interp_glsep_O2(min_load)

                    deoxo_H2O_minload     = 2.0 * glsep_O2_minload
                    deoxo_H2reac_minload = 2.0 * glsep_O2_minload
                    deoxo_heat_minload   = 2.0 * glsep_O2_minload * 244.9 * 10.0**3   # 244.9 kJ/mol → J/mol

                    T_deoxo_minload = ( deoxo_heat_minload
                    + therm_cat_deoxo * (T_glsepHXout + 125.0)
                    + therm_st_deoxo * (T_glsepHXout + 125.0)
                    + vap_h2_pdt_minload * Cp_vap_h2cool * T_glsepHXout) / (vap_h2_pdt_minload * Cp_vap_h2cool + therm_st_deoxo + therm_cat_deoxo)

                    deoxo_T = 150 +273.15 # deoxo maintains temp. at 150C
                    heater_deoxo_minload = 0.001*((deoxo_T > T_deoxo_minload).astype(float)* vap_h2_pdt_minload* Cp_vap_h2cool* (deoxo_T - T_deoxo_minload)/ eta_heater)
                    Q_lyeheater_minload = 0.001*((np.maximum((273.15+80-T_gl_out_minload)*Cp_KOH_lye_minload*Q_gl_out_minload*rho_KOH_gl_out_minload,0)  + np.maximum((273.15+80-T_angl_out_minload)*Cp_KOH_anlye_minload*Q_angl_out_minload*rho_KOH_angl_out_minload,0)))

                    W_comp_minload = 0.001*((max(Ncomp , 0) * k_isen / (k_isen - 1.0))* R_const * T_suc_comp * Z_comp * (rc ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt_minload / (eta_isen_comp * eta_mech_motor * eta_mech_comp)) #+ (P_final / x[1] > 1.0)* (1.0 * k_isen / (k_isen - 1.0) * R_const * T_amb * Z_comp* (rc_max1 ** ((k_isen - 1.0) / k_isen) - 1.0)* vap_h2_pdt_minload / (eta_isen_comp * eta_mech_motor * eta_mech_comp)))  # [web:1], in kW
                    intercool_comp_minload = max(Ncomp, 0) * (T_dis_comp - T_suc_comp) * vap_h2_pdt_minload * Cp_H2_80 #+ 1.0 * (T_max_comp - T_amb) * vap_h2_pdt_minload * Cp_H2_80)  # [web:1]

                    air_fan_minload = intercool_comp_minload / (delT_air * Cp_air * rho_air)  # [web:1]

                    W_fan_minload = 0.001*air_fan_minload * delP_fan / eta_fan  # [web:1]

                    m_cw_circ_minload =   (Q_lyecooler_minload + Q_cond_ads_minload + Q_cond_deoxo_minload + Q_cond_h2cooler_minload)/(Cp_cw*(T_cwr-T_cws))

                    m_refr_minload = m_cw_circ_minload*Cp_cw*(T_cwr-T_cws)/refr_lat_evap    #; % total refireigenrant circulation rate inside the chiller
                    W_refrcomp_minload = 0.001*(delH_comp*m_refr_minload/(eta_mech_comp*eta_mech_motor))
                    W_refrfan_minload = 0.001*((m_refr_minload*((cond_duty)))/(delT_air*Cp_air*rho_air))*delP_fan/eta_fan

                    SEC_stack_minload = interp_SEC_stack(min_load)

                    P_min = 1000*(W_pump_minload + SEC_stack_minload* vap_h2_pdt_minload * MH2 * 3600 + (heater_ads_minload + heater_deoxo_minload + heater_regen + Q_lyeheater_minload)/eta_heater  + W_refrcomp_minload + W_refrfan_minload + W_comp_minload + W_fan_minload)/eta_pow # in Watts
                    #in Watts, system power at min load
                    #vap_h2_pdt_standby_load = interp_vap_h2_pdt(minIn_SEC[-1])
                    
                    

                    #P_min_standby = interp_func14(minIn_SEC[-1])*vap_h2_pdt_standby_load * MH2 * 3600*1000 +  interp_func12(minIn_SEC[-1])*vap_h2_pdt_standby_load * MH2 * 3600*1000 # min. power for hot standby
                    T_gl_out_standby = interp_T_gl_out(minIn_SEC[-1]) 
                    T_angl_out_standby = interp_T_angl_out(minIn_SEC[-1]) 
                    Q_angl_out_standby = interp_Q_angl_out(minIn_SEC[-1]) 
                    Q_gl_out_standby = interp_Q_gl_out(minIn_SEC[-1]) 
                    w_KOH_gl_out_standby = interp_w_KOH_gl_out(minIn_SEC[-1])
                    w_KOH_angl_out_standby = interp_w_KOH_angl_out(minIn_SEC[-1])
                    Cp_KOH_lye_standby =  (4.101*10**3-3.526*10**3*w_KOH_gl_out_standby +9.644*10**-1*(T_gl_out_standby-273)+1.776*(T_gl_out_standby-273)*w_KOH_gl_out_standby)
                    Cp_KOH_anlye_standby =  (4.101*10**3-3.526*10**3*w_KOH_angl_out_standby +9.644*10**-1*(T_angl_out_standby-273)+1.776*(T_angl_out_standby-273)*w_KOH_angl_out_standby)
                    rho_KOH_gl_out_standby = (1001.53 -0.08343*(T_gl_out_standby-273.15) -0.004*(T_gl_out_standby-273.15)**2 + 5.51232*10**-6*(T_gl_out_standby-273.15)**3 - 8.21*10**-10*(T_gl_out_standby-273.15)**4)*np.exp(0.86*w_KOH_gl_out_standby)
                    rho_KOH_angl_out_standby = (1001.53 -0.08343*(T_angl_out_standby-273.15) -0.004*(T_angl_out_standby-273.15)**2 + 5.51232*10**-6*(T_angl_out_standby-273.15)**3 - 8.21*10**-10*(T_angl_out_standby-273.15)**4)*np.exp(0.86*w_KOH_angl_out_standby)

                    vel_tubes_lyecooler_standby  = (np.maximum(Q_gl_out_standby  ,Q_angl_out_standby)/N_lye)/area_lyecooler_flow_pass # use maximum of the flowrate
                    Re_tube_standby = 1260*  vel_tubes_lyecooler_standby*D_tube/0.00093 # Re number of tubes inside the HX
                    ff_standby = (0.316*Re_tube_standby**-0.25) # friction factor for smooth tubes turbulent flow , Blasius equation
                    delP_catlyecooler_standby = (1+alpha_delPHX)*ff_standby*L_tube*1260*vel_tubes_lyecooler_standby**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.
                    delP_catlyect_standby = (1+alpha_delPlyect)*delP_catlyecooler_standby # total pressure drop in individual circuits
                    delP_anlyecooler_standby = (1+alpha_delPHX)*ff_minload*L_tube*1260*vel_tubes_lyecooler_standby**2/(2*D_tube)*N_lye # total pressure drop inside lye cooler.

                    delP_anlyect_standby  = (1+alpha_delPlyect)*delP_anlyecooler_standby # total pressure drop in individual circuits
                    cell_delP_standby = interp_cell_delP(minIn_SEC[-1])
                    ancell_delP_standby= interp_ancell_delP(minIn_SEC[-1])

                    W_catlyepump_standby =  0.001*(delP_catlyect_standby + cell_delP_standby)*Q_gl_out_standby /eta_pump 
                    W_anlyepump_standby =   0.001*(delP_anlyect_standby + ancell_delP_standby)*Q_angl_out_standby /eta_pump 
                  
                    Pump_lye_standby = (W_anlyepump_minload + W_catlyepump_minload) # only the lye circ pump power is needed for standby
                    P_controls = 1000 # control power input to plant in Watts

                    P_min_standby = 1000*Pump_lye_standby/eta_pow + P_controls/eta_pow  # (273.15+80-T_gl_out_standby)*Cp_KOH*x[0]*x[2]*0.008*Nc*gc  + (273.15+80-T_angl_out_standby)*Cp_KOH*x[0]*x[2]*0.008*Nc*gc #interp_func12(minIn_SEC[-1])*vap_h2_pdt_standby_load * MH2 * 3600*1000 # min. power for hot standby

                    Heater_lye_standby = (273.15+80-T_gl_out_standby)*Cp_KOH_lye_standby*Q_gl_out_standby*rho_KOH_gl_out_standby  + (273.15+80-T_angl_out_standby)*Cp_KOH_anlye_standby*Q_angl_out_standby*rho_KOH_angl_out_standby # min. heater duty to maintain temp at set point during warm standby, currently not part of min. standby power

                    #P_min1 = SEC_sys_min_load1 * vap_h2_pdt_min_load1 * MH2 * 3600*1000
                    #awebop_power = np.where(pv_powerout >= P_min, np.minimum(pv_powerout, W_sys75*1000), 0)
                  
                    #global pv_powerout, wind_powerout
                    
                    #genratio = self.genratio
                    pv_powerout_scaled = pv_powerout* genratio/2 * (W_sys75)
                    wind_powerout_scaled = wind_powerout * genratio/2 * (W_sys75)*(1/1504.5) # to get the correct mean annual energy produced including wind effects


                    hybrid_powerout = wind_powerout_scaled+pv_powerout_scaled
                    P_min_vec = np.full(len(hybrid_powerout), P_min)
                    below_min = hybrid_powerout < P_min
              #      below_min1 = hybrid_powerout < P_min1
                    n_true = np.sum(below_min)
                    
                    # Hot standby: P_min_standby <= P < P_min
                    prod_mask = hybrid_powerout >= P_min
                    standby_mask = (hybrid_powerout >= P_min_standby) & (hybrid_powerout < P_min) # standby mode active 
                    # Idle: P < P_min_standby
                    idle_mask = hybrid_powerout  < P_min_standby # idle mode active
                    # --- Time in each mode (hours) ---------------------------------------------
                    # A start occurs when we were NOT in production at t-1, but ARE in production at t



                    ## Thermal RC model for calculation of temp. decay rate in idle and AWE ramp rate to set point temperature (80C)... all rho and Cp values calculated at 80C
                    #  Calc. of glsep thickness and material mass
                    od_glsep = 1.15
                    #Q_gl_out = x[2]*0.008*x[0]*0.001*Nc*gc # in m^3/s
                    Pg = x[1]-1 # guage pressure

                    vol_liq_catglsep_standby = 120 * Q_gl_out_standby 
                    vol_liq_anglsep_standby =  120 * Q_angl_out_standby  # m^3, 2 mins residence time, liquid volume inside the separator (sized at rated liq volume resid time), at standby conditions
                    #vol_glsep     = od_glsep * 120 * Q_gl_out_at_eta75 *2                       # overdesigned volume, 50% fill level
                  # glsep_D       = ((4 * vol_glsep) / (3 * np.pi)) ** (1.0 / 3.0)   # diameter [m]
                  # glsep_L       = 3 * glsep_D                                      # length [m]

                    # Pressure vessel thickness
                    thick_glsep = max(((Pg + 1) * glsep_D / (2 * 944 * 0.9 - 1.2 * (Pg + 1)) + 0.00315), 0.0063) # 0.00315 is the corrosion thickness, and 0.0063 is min. thickness

                    vol_glsep_steel = np.pi* (glsep_D+thick_glsep)**2/4.0*glsep_L -  np.pi* (glsep_D)**2/4.0*glsep_L
                    mass_glsep = 2*vol_glsep_steel* rho_steel # rho_steel is the density of the carbon steel (MOC og glsep), mult by 2 for 2 pieces of glsep
                    rho_KOH = (1001.53 -0.08343*(80) -0.004*(80)**2 + 5.51232*10**-6*(80)**3 - 8.21*10**-10*(80)**4)*np.exp(0.86*.302) # density of KOH at nominal conditions ie 80C, and 30.2 wt%
                    Cp_KOH = (4.101*10**3-3.526*10**3*0.302 +9.644*10**-1*(80)+1.776*(80)*0.302)

                    mass_KOH  = vol_liq_catglsep_standby*rho_KOH_gl_out_standby + vol_liq_anglsep_standby*rho_KOH_angl_out_standby  + (gc*2*(0.05*0.008*x[0]*0.001*(Nc))*0.85 + gc*0.05*0.008*Nc*x[4]*0.001*0.57)*rho_KOH # rho_KOH is the KOH density at 80C, 30.2 wt%, 85 % porosity of electrode is assumed.
                    therm_KOH = vol_liq_catglsep_standby*rho_KOH_gl_out_standby*Cp_KOH_lye_standby + vol_liq_anglsep_standby*rho_KOH_angl_out_standby*Cp_KOH_anlye_standby + (gc*2*(0.05*0.008*x[0]*0.001*(Nc))*0.85 + gc*0.05*0.008*Nc*x[4]*0.001*0.57)*rho_KOH*Cp_KOH # therm cc of KOH, J/kg

                    mass_stack_SS = (1+0.2*(x[1]-1)/(15-1))*steel/4.5   # consists of SS316L
                    mass_stack_Ni =  (Ni/Ni_foam)*rho_Ni # consists of Ni 
                    heat_cc_elec = (mass_stack_Ni*Cp_Ni +  (mass_stack_SS + mass_glsep)*Cp_steel + therm_KOH)*1.2 # factor of 1.2 to account for heat cc of piping, valves, HX, pumps in the electrolyte circuit 
                    
                    Area_stack = 2*gc*(0.05*0.003*2+0.05*0.001*(Nc-1) + 2*(0.05*x[0]*0.001*(Nc))+ (0.05*x[4]*0.001*(Nc)) + 0.008*0.005 + 0.008*(2*(x[0]*0.001*(Nc))+ (x[4]*0.001*(Nc)))) # total stack exposed surface area

                    Area_glsep = 2*(np.pi*(glsep_D+thick_glsep)*glsep_L + 2*np.pi*(glsep_D+thick_glsep)**2/4.0)   # total glsep exposed area assuming cylindrical shape
                    Tot_Area = (Area_stack + Area_glsep)*1.3 # 1.3 factor to account for extra area from piping, valves HX etc.
                    htc_elec = 5 # taken as const, in W/m^2*K
                    therm_tau = heat_cc_elec/(Tot_Area*htc_elec)/3600 # thermal time constant
                    alpha = np.exp(-dt / therm_tau)   # dimensionless temp decay factor
                    
                    pareto_therm_tau[i] = therm_tau

                    T_sp = 353.15 # set point at 80C
                    T = T_sp        # assume hot at start, or set to T_amb
                    T_hist = np.zeros_like(hybrid_powerout, dtype=float)
                    hA = Tot_Area * htc_elec
                    heater_rated_limited = np.zeros_like(hybrid_powerout, dtype=bool)
                    
                    # hours where heater is limiting AND still heating up


                    for kk, P_ren in enumerate(hybrid_powerout):
                      P_from_renew = eta_pow * eta_heater * P_ren
                      P_rated_el   = eta_heater * Pow_lyeheater

                      if P_rated_el <= P_from_renew:
                      # heater is at its rating; renewables could give more
                        P_heater_el = P_rated_el
                        heater_rated_limited[kk] = True
                      else:
                      # renewables are limiting
                        P_heater_el = P_from_renew
                        heater_rated_limited[kk] = False

                      Q_heater = P_heater_el   
                      #P_heater = np.minimum(eta_heater*Pow_lyeheater,eta_pow*eta_heater*P_ren)
                      T_ss_heat = T_amb + Q_heater / hA # heater time constant is NIL, that is instantaneous response of lye heating to heater power supplied.
                      if P_ren < P_min_standby:
                      # idle cooling
                        T = T_amb + (T - T_amb) * alpha

                      elif (P_ren >= P_min_standby) and (P_ren < P_min):
                      # heating only, no production
                        T = T_ss_heat + (T - T_ss_heat) * alpha
                        if T > T_sp: T = T_sp

                      else:  # P >= P_min
                        if T < T_sp:
                          # still heating to setpoint (no production until T>=T_sp)
                          T = T_ss_heat + (T - T_ss_heat) * alpha
                          if T > T_sp: T = T_sp
                        else:
                          T = T_sp   # operate at setpoint
                          
                      T_hist[kk] = T
                      pareto_T_hist[i, :] = T_hist

                    below_Tsp        = T_hist < 353.15 
                    heater_at_rating = heater_rated_limited 
                    mask_heating_limited = heater_at_rating & below_Tsp
                    hours_heating_limited = mask_heating_limited.sum() * dt 
                    frac_heating_limited = hours_heating_limited/8760
                    pareto_frac_heating_limited[i] = frac_heating_limited         

                    startup_mask = (prod_mask) & (T_hist < T_sp)

                    prod_mask_eff   = prod_mask & ~startup_mask   # truly productive
                    standby_mask_eff = standby_mask | startup_mask

                    prod_shift = np.roll(prod_mask_eff, 1)
                    prod_shift[0] = False  # no start at first element by definition
                    starts_mask = (~prod_shift) & prod_mask_eff
                    n_starts = np.sum(starts_mask)

                    pareto_n_starts[i] = n_starts
                    hours_prod    = np.sum(prod_mask_eff)    * dt
                    hours_standby = np.sum(standby_mask_eff) * dt

                    hours_idle    = np.sum(idle_mask)    * dt

                    pareto_hours_prod[i] = hours_prod
                    pareto_hours_standby[i] = hours_standby
                    pareto_hours_idle[i]  = hours_idle    
                    startup_hours = startup_mask.sum() * dt

                    E_standby = 0.001*np.sum(standby_mask_eff * P_min_standby * dt)  # kWh per year

                    # startup time per hot start (h)
                    # 2) Dips start where we go from ON (False) to OFF (True)
                    dip_start_mask = (~below_min[:-1]) & (below_min[1:])   # ON -> OFF
                    dip_stop_mask  = (below_min[:-1]) & (~below_min[1:])   # OFF -> ON
                    
                    dip_start_idx = np.where(dip_start_mask)[0] + 1
                    dip_stop_idx  = np.where(dip_stop_mask)[0] + 1
                    L = dip_stop_idx[1:] - dip_start_idx[:-1]
                    short_cycles = np.sum(L == 1)       # 1 h dips
                    long_cycles  = np.sum(L >  1)       # >1 h dips
                    
                    #start_mask = (below_min[:-1] == True) & (below_min[1:] == False)
                      
                    # Number of starts (scalar)
                    #n_starts = np.sum(start_mask)

                    # total startup-lost hours per year

                    # effective operating hours at/above min load (if you want it explicitly)
                    hours_above_min = np.sum(~below_min)    # number of time steps with operation
                    #effective_operating_hours = hours_above_min - startup_lost_hours
                    P_clip = np.minimum(hybrid_powerout, W_sys * 1000)
                    awebop_power = np.where(P_clip >= P_min, P_clip, 1e-06)
                    awebop_power_curt = np.where(P_clip >= P_min_standby, P_clip, 1e-6) # the W_sys is used here, as the further calc. load _factor is used find the annual h2 production, so the peak power is not used
                    total_energy_supplied_curt = np.sum(awebop_power_curt)
                    heater_mask = startup_mask | standby_mask_eff
              # kWh lost in startups
                    P_lost       = awebop_power* startup_mask
                    E_lost       = P_lost.sum() * dt

                    total_energy_supplied = np.sum(awebop_power)  # If array in kW, yields kWh for the year, is the total energy supplied by the pow gen within the load range of the awe electrolyzer
              #      total_energy_supplied1 = np.sum(awebop_power1)  # If array in kW, yields kWh for the year, is the total energy supplied by the pow gen within the load range of the awe electrolyzer  
                    E_effective = total_energy_supplied - E_lost
                    total_energy_supplied = np.sum(E_effective)
              #      E_effective1 = total_energy_supplied1 - E_lost1
                    theoretical_max = W_sys75*1000 * len(pv_powerout)
                    heater_start_pow = (np.minimum(Pow_lyeheater,np.maximum(awebop_power_curt-P_min_standby,0)))    # this is the power consumed by the heater during standby, MAx. theoretical load is the rated load, and the LF is calcuated is based on the rated power of the AWE
                    pump_start_pow =1000*Pump_lye_standby + 1000 # in W, second term for control power (1kW fixed)
                    pareto_p_min_standby[i] = Pump_lye_standby + 1 # in kW
                    #print(W_sys75)
                    pareto_frac_standby_ener[i] = (np.sum(heater_start_pow*heater_mask*0.001*dt)+np.sum(pump_start_pow*standby_mask_eff*0.001*dt))/(total_energy_supplied_curt*0.001)# fraction of renew absorbed energy used up in non-production uses such as standby heating and lye circ
                    plant_availability = 0.97
                    load_factor = E_effective / theoretical_max*100*plant_availability
                    pareto_load_factor[i] = load_factor

              #      load_factor1 = E_effective1 / theoretical_max*100*plant_availability
                    #======== calculation of stack degradation, SEC_avg of platn life and replacement present value 
                    #m_degrate = self.m_degrate
                    #nj = self.nj   # or whatever you call the exponential coeff
                    #bj=0.00045 # term multiplier for current density in the expenential degradation dependence, bj=0.00004
                    degrate = base_degrate + m_degrate_sweep*(np.maximum(maxT,1)**nj_sweep )*np.exp(bj*x[-1])  # base_degrate is 1% per annum
                    alphaa = sw*2*5*10**-4 
                    degrate_eff = degrate* (1.0 + alphaa * n_starts)  # degrate = 0.125  + m_degrate * max(maxT_interp - T_thresh, 0).^3.*exp(0.00001*xFine); degradation rate is % per annum
                    pareto_degrate_eff[i] = degrate_eff 

                    #degrate = base_degrate + m_degrate * np.exp(bj*x[-1])
                    dur_stack = degrade_stack / degrate_eff * 8760 # durability of the stack in hours  
                    #print(degrate_eff)



                
                    t_oper = load_factor*8760/100 # t_oper is the number of operating hours at rated load
                    #t_oper = -0.6453*(self.cd_eta75_base_ratio*100) -13.571*(self.cd_eta75_base_ratio*100) + 3486.1 # operational plant hours per year , 40% CF

                    t_total = t_oper * t_sys

                    Nrep = int(np.floor(t_total / dur_stack))
                    pareto_Nrep[i] = Nrep 
                    dur_rem = t_total - Nrep * dur_stack

                    SEC_initial = SEC_stack                   # kWh/kg H₂ (BOL)
                    SEC_end = SEC_initial * (1 + degrade_stack / 100) 
                    SEC_avg_per_stack = 0.5 * (SEC_initial + SEC_end)
                    if dur_rem > 0:
                      frac_life = dur_rem / dur_stack  # Fraction of last stack used
                      SEC_end_partial = SEC_initial + frac_life * (SEC_end - SEC_initial)
                      SEC_avg_partial = 0.5 * (SEC_initial + SEC_end_partial)
                    else:
                      SEC_avg_partial = 0

                    SEC_avg_stack = (Nrep * dur_stack * SEC_avg_per_stack + dur_rem * SEC_avg_partial) /t_total#  avg stack SEC over plant lifetime
                    pareto_SEC_avg_stack[i] = (Nrep * dur_stack * SEC_avg_per_stack + dur_rem * SEC_avg_partial) /t_total
                  
                    SEC_avg_sys = ((SEC_avg_stack* vap_h2_pdt * MH2 * 3600 + W_sys_exstack + np.sum(heater_start_pow*heater_mask*0.001*dt)/8760 +  np.sum(pump_start_pow*standby_mask_eff*0.001*dt)/8760))/(vap_h2_pdt*MH2*3600)/eta_pow
                    #SEC_avg_sys=SEC_avg_sys; #% This value is currently not being used in the glob. optim GA code. This shud be ideally the value to minimize.

                    #replacement_pv = Tot_stack_cost / (1 + r) ** (dur_stack / t_oper)
                    repl_inter_yrs = dur_stack/t_oper
                    PV_repl=0.0
                    for k in range(1,Nrep+1):
                      PV_repl+= Tot_stack_cost/(1+r)**(k*repl_inter_yrs)

                    Tot_stack_cost_life = Tot_stack_cost+PV_repl # total stack costs over plant lifetime   
                    ######################################### Total BM_capex calculation... ###########################################

                    BM_capex = Tot_stack_cost_life+BM_pow + BM_heaters_lye + BM_heaters + BM_h2cool+ BM_ads + BM_de+BM_glsep + BM_cwpump+BM_lyepumps + BM_evap +BM_refr + BM_shaft_h2comp +BM_shaft_refrcomp+ BM_h2comp + BM_refrcomp + BM_intercool + BM_fan +BM_refrfan +BM_lyecool +BM_deoxocool + BM_des_cool
                    BM_0_capex = Tot_stack_cost/Fp_stack +BM_pow + BM_heaters_lye + BM_heaters + BM_0_h2cool + BM_0_ads + BM_0_de + BM_0_glsep + BM_0_cwpumps + BM_0_lyepumps + BM_0_evap + BM_0_refr + BM_0_shaft_h2comp+BM_0_shaft_refrcomp + BM_0_h2comp +BM_0_refrcomp + BM_0_intercool + BM_0_fan + BM_0_refrfan + BM_0_lyecool +BM_0_deoxocool + BM_0_des_cool
                    capex_aux = BM_0_capex*0.5# Auxilliaries
                    capex_cont =  0.18*BM_capex # contingnecy as a percentage of total fixed costs
                    TIC = 1.18*BM_capex + capex_aux

                    BMM_h2compression[i] = BM_intercool + BM_shaft_h2comp + BM_h2comp + BM_fan
                    BMM_refrig[i] = BM_refr+BM_evap+BM_shaft_refrcomp+BM_refrcomp+BM_refrfan
                    BMM_gasliqsep[i] = BM_glsep+BM_h2cool
                    BMM_h2purif[i] = BM_des_cool + BM_deoxocool + BM_ads + BM_de + BM_heaters #contains deoxo heater and ads heater

                    BMM_lyecool[i] =BM_lyecool
                    BMM_heaters[i] =BM_heaters_lye
                    BMM_pumps[i] = BM_cwpump+BM_lyepumps
                    BMM_pow [i] = BM_pow
                    ############################### LCOE calculation ##########################
                    base_cc = 100 # base plant cc in MW
                    m_scale =0.9
                    PV_basecapex = 900 # $/kW
                    PV_capex = (PV_basecapex*(100*1000)*((genratio/2 * (W_sys75)*0.001)/(100))**(m_scale))/(genratio/2 * (W_sys75)) # scaled capex based on installed renew capacity
                    wind_basecapex = 1500 # $/kW
                    wind_capex = (wind_basecapex*(100*1000)*((genratio/2 * (W_sys75)*0.001)/(100))**(m_scale))/(genratio/2 * (W_sys75)) # scaled capex based on installed renew capacity
                    PV_opex = 17
                    wind_opex =44
                    

                    n_renew = 25
                    CRF_renew = (r*(1+r)**n_renew)/((1+r)**n_renew-1)
                                          # hours per step
                    #% Cumulative energy produced per annum kWh/(kW installed)/yr, factor of 0.001 as the data from SAM is in watts / per unit kW insatlled capacity
                    CF_wind = np.sum(wind_powerout_scaled)*0.001/((genratio/2 * (W_sys75))*8760) 
                    CF_solar = np.sum(pv_powerout_scaled)*0.001/((genratio/2 * (W_sys75))*8760)
                    LCOE_wind = (wind_capex*CRF_renew+wind_opex)/(CF_wind*8760)
                    LCOE_solar = (PV_capex*CRF_renew+PV_opex)/(CF_solar*8760)
                    E_gen   = np.sum(hybrid_powerout) # total energy generated by the hybrid renew system per annum kWh/yr

                    elec_cost = 0.5*(LCOE_wind+LCOE_solar)*(E_gen/total_energy_supplied_curt) # the total energy supplied is actually the energy used by the electrolyzer, ie the power profiles from the renew faciltiy falling within the electrolyzer's operating range.
                    pareto_LCOE_awe[i] =0.5*(LCOE_wind+LCOE_solar)*(E_gen/total_energy_supplied_curt) # the total energy supplied is actually the energy used by the electrolyzer, ie the power profiles from the renew faciltiy falling within the electrolyzers operating range.
                    pareto_curtail_factor[i] = (E_gen/total_energy_supplied_curt) # curtailment factor, the amount of energy not utilized due to the AWE power inflexibility
                    pareto_cd_eta75[i] = cd_eta75
                    pareto_vap_h2_pdt[i] = (vap_h2_pdt* MH2* t_oper * 3600)
                    ##################################################################################################################################################################################################################################################################
                    ##############################################################################################################################################################################################################################################
                    

                    scipy.io.savemat(fname, {
                        'pareto_F': pareto_F,
                        'pareto_X': pareto_X,
                        'pareto_G': pareto_G,
                        'pareto_W_sys75': pareto_W_sys75,
                        'pareto_load_factor': pareto_load_factor,
                        'pareto_LCOE_awe': pareto_LCOE_awe,
                        'pareto_curtail_factor': pareto_curtail_factor,
                        'pareto_maxT': pareto_maxT,
                        'pareto_vap_h2_pdt' : pareto_vap_h2_pdt,
                        'pareto_SEC_sta' : pareto_SEC_sta,
                        'pareto_cd_eta75' : pareto_cd_eta75,
                        'pareto_Nrep' : pareto_Nrep,
                        'pareto_degrate_eff':pareto_degrate_eff,
                        'pareto_n_starts':pareto_n_starts,
                        'pareto_frac_heating_limited' : pareto_frac_heating_limited,
                        'pareto_hours_prod' : pareto_hours_prod,
                        'pareto_hours_standby' : pareto_hours_standby,
                        'pareto_hours_idle'  : pareto_hours_idle,
                        'pareto_therm_tau' : pareto_therm_tau,
                        'BM_pow': BMM_pow,
                        'BM_pumps':BMM_pumps,
                        'BM_heaters' : BMM_heaters,
                        'BM_lyecool' : BMM_lyecool,
                        'BM_h2purif' : BMM_h2purif,
                        'BM_gasliqsep' : BMM_gasliqsep,
                        'BM_refrig' : BMM_refrig,
                        'BM_h2compression' : BMM_h2compression, 
                        'pareto_p_min_standby' : pareto_p_min_standby,
                        'pareto_frac_standby_ener' : pareto_frac_standby_ener,     
                        'pareto_T_hist' : pareto_T_hist,
                        'pareto_SEC_avg_stack' : pareto_SEC_avg_stack
                    })  

        print(f'Saved {fname}')
        end_time = time.perf_counter()
        #print(f"cd_eta75 ratio {ratio_val}: elapsed time = {end_time - start_time:.2f} s",
        #    flush=True)


