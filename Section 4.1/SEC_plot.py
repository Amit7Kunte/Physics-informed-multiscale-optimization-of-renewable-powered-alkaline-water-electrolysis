import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import os
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
# this code is for the generation of the SEC plot for the model scale in Fig.4 of the paper
#os.chdir(r"C:\Users\KUNTEAM\Desktop\Neom_Model_Data\WP4.4")
# Load scaled input/output data
mat = scipy.io.loadmat('preparedData_SEC.mat', squeeze_me=True)
allInputs = mat['allInputs']
allOutputs = mat['allOutputs']

# Load scaling min/max data
'''scaling = scipy.io.loadmat('preparedData_scaled.mat', squeeze_me=True)
minIn = scaling['minIn'].flatten()
maxIn = scaling['maxIn'].flatten()
minIn = np.delete(minIn, 4)
maxIn = np.delete(maxIn, 4)
minOut = scaling['minOut'][0]
maxOut = scaling['maxOut'][0]'''

# Get current density (last column of input data)
curr_dens = allInputs[:,-1]

# Output variable names and column mapping
output_vars = [
    'SEC_sys', 'SEC_stack', 'eta_stack', 'eta_volt',
    'eta_curr_stack','eta_curr_sys','SEC_glsep','SEC_chiller_lyecooler','SEC_compr' , 'SEC_pumps', 'SEC_H2purif' ,'SEC_lyeheaters'
]
col_map = {name: idx for idx, name in enumerate(output_vars)}

# Extract required outputs for subplot 1
SEC_sys = allOutputs[:, col_map['SEC_sys']]
SEC_stack = allOutputs[:, col_map['SEC_stack']]
eta_volt = allOutputs[:, col_map['eta_volt']]
eta_curr_stack = allOutputs[:, col_map['eta_curr_stack']]
eta_curr_sys = allOutputs[:, col_map['eta_curr_sys']]

# Efficiency calculations
HHV_H2 = 39.68  # kWh/kg
eta_stack = (HHV_H2 / SEC_stack) * 100

# Extract required outputs for subplot 2
SEC_H2purif = allOutputs[:, col_map['SEC_H2purif']]
SEC_chiller_h2cooler = allOutputs[:, col_map['SEC_glsep']]
SEC_chiller_lyecooler = allOutputs[:, col_map['SEC_chiller_lyecooler']]
SEC_compr = allOutputs[:, col_map['SEC_compr']]
#SEC_deoxo = allOutputs[:, col_map['SEC_deoxo']]
SEC_pumps = allOutputs[:, col_map['SEC_pumps']]
SEC_lyeheaters = allOutputs[:, col_map['SEC_lyeheaters']]

# Interpolation grid for current density (spacing 500)
idx = np.argsort(curr_dens)
x = curr_dens[idx]
y_sys = SEC_sys[idx]
y_H2 = SEC_H2purif[idx]

curr_dens_interp = np.linspace(x.min(), x.max(), 1000) # not 100–200
#curr_dens_interp = np.linspace(curr_dens.min(), curr_dens.max(), 100)
def smooth_spline(x, y, x_new):
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    # Apply Savitzky-Golay filter after interpolation
    y_smooth = savgol_filter(y_smooth, window_length=51, polyorder=3)
    return y_smooth

def interpolate_curve(x, y, x_new):
    spline = make_interp_spline(x, y, k=3)
    return spline(x_new)

#SEC_sys_smooth = smooth_spline(curr_dens, SEC_sys, curr_dens_interp)
# Interpolate curves for subplot 1
pchip_sys = PchipInterpolator(x, y_sys)
pchip_H2  = PchipInterpolator(x, SEC_H2purif[idx])
SEC_sys_interp    = pchip_sys(curr_dens_interp)
SEC_H2_purif_interp = pchip_H2(curr_dens_interp)
#SEC_sys_interp = interpolate_curve(curr_dens, SEC_sys, curr_dens_interp)
#SEC_sys_interp = SEC_sys_smooth
SEC_stack_interp = interpolate_curve(curr_dens, SEC_stack, curr_dens_interp)
eta_stack_interp = interpolate_curve(curr_dens, eta_stack, curr_dens_interp)
eta_volt_interp = interpolate_curve(curr_dens, eta_volt, curr_dens_interp)
eta_curr_stack_interp = interpolate_curve(curr_dens, eta_curr_stack, curr_dens_interp)
eta_curr_sys_interp = interpolate_curve(curr_dens, eta_curr_sys, curr_dens_interp)


SEC_sys_interp = savgol_filter(SEC_sys_interp, window_length= 50, polyorder=2)
SEC_H2_purif_interp = savgol_filter(SEC_H2_purif_interp, window_length= 50, polyorder=2)
# Interpolate curves for subplot 2
#SEC_ads_interp = interpolate_curve(curr_dens, SEC_ads, curr_dens_interp)
SEC_chiller_gasliqsep_interp = interpolate_curve(curr_dens, SEC_chiller_h2cooler, curr_dens_interp)
SEC_chiller_lyecooler_interp = interpolate_curve(curr_dens, SEC_chiller_lyecooler, curr_dens_interp)
SEC_compr_interp = interpolate_curve(curr_dens, SEC_compr, curr_dens_interp)
#SEC_deoxo_interp = interpolate_curve(curr_dens, SEC_deoxo, curr_dens_interp)
SEC_pumps_interp = interpolate_curve(curr_dens, SEC_pumps, curr_dens_interp)
#SEC_H2_purif_interp = interpolate_curve(curr_dens, SEC_H2purif, curr_dens_interp)   # +interpolate_curve(curr_dens, SEC_deoxo, curr_dens_interp)
#SEC_lyeheaters_interp = np.interp(curr_dens, SEC_lyeheaters, curr_dens_interp)
# For SEC_lyeheater use np.interp instead of make_interp_spline
SEC_lyeheaters_interp = np.interp(curr_dens_interp, curr_dens, SEC_lyeheaters)

# sort data
from scipy.signal import savgol_filter
import numpy as np

idx = np.argsort(curr_dens)
x = curr_dens[idx]
y = SEC_sys[idx]

#curr_dens_interp = np.linspace(x.min(), x.max(), 50)
y_lin = np.interp(curr_dens_interp, x, y)

# very light smoothing – just kill tiny wiggles
#SEC_sys_interp = savgol_filter(y_lin, window_length=51, polyorder=3)

# find the rated curr dens and rated power values for components for the snakey diagram
rated_load = 75
idxx = np.where(np.diff(np.sign(eta_stack - rated_load)))[0]
i = idxx[-1]
cd1, cd2 = curr_dens_interp[i], curr_dens_interp[i + 1]
eta1, eta2 = eta_stack[i], eta_stack[i + 1]
cd_eta75 = cd1 + (rated_load - eta1) * (cd2 - cd1) / (eta2 - eta1)

#interp_SEC_stack      = interp1d(curr_dens_interp, SEC_stack_interp,      kind='linear', fill_value="extrapolate")


import pandas as pd

# Make sure these exist: curr_dens_interp, SEC_sys_interp, SEC_stack_interp,
# eta_stack_interp, eta_volt_interp, eta_curr_stack_interp, eta_curr_sys_interp

df_plot1 = pd.DataFrame({
    'j_Aperm2': curr_dens_interp,
    'SEC_sys_kWhperkg': SEC_sys_interp ,
    'SEC_stack_kWhperkg': SEC_stack_interp,
    'eta_stack_pct': eta_stack_interp,
    'eta_volt': eta_volt_interp,
    'eta_curr_stack': eta_curr_stack_interp,
    'eta_curr_sys': eta_curr_sys_interp
})

df_plot1.to_csv('currdens_vs_globalmetrics.csv', index=False)

# Make sure these exist: curr_dens_interp,
# SEC_chiller_gasliqsep_interp, SEC_chiller_lyecooler_interp,
# SEC_comp_interp, SEC_pumps_interp, SEC_H2_purif_interp,
# SEC_lyeheaters_interp

df_plot2 = pd.DataFrame({
    'j_Aperm2': curr_dens_interp,
    'SEC_gasliqsep_kWhperkg': SEC_chiller_gasliqsep_interp,
    'SEC_lyecooler_kWhperkg': SEC_chiller_lyecooler_interp,
    'SEC_comp_kWhperkg': SEC_compr_interp,
    'SEC_pumps_kWhperkg': SEC_pumps_interp,
    'SEC_H2_purif_kWhperkg': SEC_H2_purif_interp,
    'SEC_lyeheaters_kWhperkg': SEC_lyeheaters_interp
})

df_plot2.to_csv('currdens_vs_equipmentSEC.csv', index=False)


# Set Matplotlib parameters for publication quality
plt.rcParams.update({
    'figure.facecolor': 'w',
    'axes.labelweight': 'bold',
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.edgecolor': 'k',
    'axes.linewidth': 1.5,
    'legend.fontsize': 12,
    'xtick.major.size': 6,
    'ytick.major.size': 6
})
legend_fontsize = 10
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
label_fontsize = 10

# Subplot 1
axs[0].plot(curr_dens_interp, SEC_sys_interp, label='SEC system')
axs[0].plot(curr_dens_interp, SEC_stack_interp, label='SEC stack')
axs[0].plot(curr_dens_interp, eta_stack_interp, label='stack power efficiency')
axs[0].plot(curr_dens_interp, eta_volt_interp, label='stack voltage efficiency')
axs[0].plot(curr_dens_interp, eta_curr_stack_interp, label='stack current efficiency')
axs[0].plot(curr_dens_interp, eta_curr_sys_interp, label='system current efficiency')
axs[0].set_xlabel('Current Density (A/m$^2$)', fontweight='bold', fontsize=label_fontsize)
axs[0].set_ylabel('SEC (kWh/kg) / Efficiency', fontweight='bold', fontsize=label_fontsize)

ticks = np.arange(0, curr_dens.max() + 4000, 4000)
tick_labels = [str(int(x)) for x in ticks]
axs[0].set_xticks(ticks)
axs[0].set_xticklabels(tick_labels, fontsize=label_fontsize-2)
axs[0].set_ylim(40, 100)
axs[0].set_yticks(np.arange(40, 101, 10))  # ticks every 10 units
# Optional: if you want big readable ticks:
axs[0].set_yticklabels([str(i) for i in np.arange(40, 101, 10)], fontsize=label_fontsize-2)

#
axs[0].legend(loc='upper right', frameon=False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_xlim(left=0)


# Subplot 2
axs[1].plot(curr_dens_interp, SEC_H2_purif_interp, label='SEC H2purif')
axs[1].plot(curr_dens_interp, SEC_chiller_gasliqsep_interp, label='SEC G/L cooler')
axs[1].plot(curr_dens_interp, SEC_chiller_lyecooler_interp, label='SEC lyecooler')
axs[1].plot(curr_dens_interp, SEC_compr_interp, label='SEC compressor')
#axs[1].plot(curr_dens_interp, SEC_deoxo_interp, label='SEC de-oxidizer')
axs[1].plot(curr_dens_interp, SEC_pumps_interp, label='SEC pumps')
axs[1].plot(curr_dens_interp, SEC_lyeheaters_interp, label='SEC lyeheaters')
axs[1].set_xlabel('Current Density (A/m$^2$)', fontweight='bold', fontsize=label_fontsize)
axs[1].set_ylabel('SEC (kWh/kg)', fontweight='bold', fontsize=label_fontsize)

axs[1].set_xticks(ticks)
axs[1].set_xticklabels(tick_labels, fontsize=label_fontsize-2)
axs[1].set_yticklabels([str(int(y)) if y==int(y) else str(round(y,2)) for y in axs[1].get_yticks()], fontsize=label_fontsize-2)
axs[1].legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 1))
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].set_xlim(left=0)
axs[1].set_ylim(0, 10)
axs[1].set_yticks(np.arange(0, 6, 1))  # ticks every 10 units
# Optional: if you want big readable ticks:
axs[1].set_yticklabels([str(i) for i in np.arange(0, 6, 1)], fontsize=label_fontsize-2)

plt.tight_layout()
for ax in axs:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
for ax in axs:
    ax.set_frame_on(True)                # ensures axis spines form a visible rectangle
    for spine in ax.spines.values():
        spine.set_edgecolor('black')     # set frame color
        spine.set_linewidth(1.0)         # set frame thickness

axs[0].legend(loc='upper right', frameon=False, 
              fontsize=legend_fontsize, 
              prop={'weight':'bold'})
axs[1].legend(loc='upper right', frameon=False, bbox_to_anchor=(1,1), 
              fontsize=legend_fontsize, 
              prop={'weight':'bold'})
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size(legend_fontsize-2)
fontP.set_weight('bold')

axs[0].legend(loc='upper right', frameon=True,
              prop=fontP)
axs[1].legend(loc='upper center', frameon=True,
              prop=fontP)

plt.show()
plt.pause(0.10)
