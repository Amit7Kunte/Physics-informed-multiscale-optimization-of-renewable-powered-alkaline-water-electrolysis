from matplotlib.lines import Line2D
import numpy as np
import h5py
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import re
import statsmodels.api as sm

#os.chdir(r"D:\Comsol_Tut\EquationBasedModelling\HTO_paper_models\latest_model_tcd\Neom_report_models\Publication Models\modified_model\Optimization_study\saved data")
# --- User config ---
model_file = 'trainedANN_seed7084_boxcox_3.keras'
scaling_file = 'preparedData_BoxCox_scaled.mat'
matfiles = glob.glob('iModel_Wgde_1.2_P_5_vin_0.025_dpore_500_dpored_0_Wsep_*_Ls_2000_datan_ANN_valid_inputs.mat')
lam = -0.5

def inv_boxcox(y, lam, c=0.0):
    """Inverse of x_bc = ((x+c)**lam - 1)/lam  (lam != 0)
       or x_bc = log(x+c) for lam == 0.
    """
    if lam == 0:
        return np.exp(y) - c
    else:
        return np.power(lam * y + 1.0, 1.0 / lam) - c

output_vars = ['maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O', 
    'w_KOH_angl_out','w_KOH_gl_out','Q_gl_out','Q_angl_out','glsep_O2' , 'H2_mixedToHTO', 'Q_cond_h2cooler' ,'Q_cond_ads', 'Q_cond_deoxo','T_gl_out','T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]


output_names = [
    'maxT',
    #r'$\mathbf{\left\{\Delta T\right\}_{max}\ \left(^{\circ}\mathrm{C}\right)}$',
    'SEC_stack',
    'vap_h2_pdt', 'H_T_O', 
    'w_KOH_angl_out','w_KOH_gl_out','Q_gl_out','Q_angl_out','glsep_O2' , 'H2_mixedToHTO', 'Q_cond_h2cooler' ,'Q_cond_ads', 'Q_cond_deoxo','T_gl_out','T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]


scaling = scipy.io.loadmat(scaling_file)
minIn = scaling['minIn'][0]
maxIn = scaling['maxIn'][0]
minIn     = np.delete(minIn, 4)
maxIn     = np.delete(maxIn, 4)
minOut = scaling['minOut'][0]
maxOut = scaling['maxOut'][0]
model = tf.keras.models.load_model(model_file)

fig, axes = plt.subplots(8, 3, figsize=(17, 25), dpi=200)
axes = axes.flatten()

colors = plt.cm.tab10.colors
markers = ['o', 's', '^', 'v', 'D', 'P', 'X', 'h', '*', '<', '>']

rmse_per_file_per_output = [[] for _ in range(len(output_names))]  # store RMSE values for each output per file

# For custom legend handles
legend_handles, legend_labels = [], []

for file_idx, matfile in enumerate(matfiles):
    with h5py.File(matfile, 'r') as f:
        j_max = maxIn[-1] 
        currentDensity_full = np.array(f['currentDensity']).squeeze()
        mask = currentDensity_full <= j_max

        currentDensity = currentDensity_full[mask] 
        tokens = matfile.split('_')

        def extract_param(tokens, key):
            try:
                return float(tokens[tokens.index(key) + 1])
            except Exception:
                return np.nan
        input_scalar = [
            extract_param(tokens, 'Wgde'),
            extract_param(tokens, 'P'),
            extract_param(tokens, 'vin'),
            extract_param(tokens, 'dpore'),
            #extract_param(tokens, 'dpored'),
            extract_param(tokens, 'Wsep')
        ]
        X_ann = np.array([input_scalar + [cd] for cd in currentDensity])
         
        X_ann_scaled = 2 * (X_ann - minIn) / (maxIn - minIn) - 1
        y_pred_scaled = model.predict(X_ann_scaled)
        y_pred = 0.5 * (y_pred_scaled + 1) * (maxOut - minOut) + minOut



        y_true_list = []
        for v in output_vars:
            data_full = np.array(f[v]).squeeze() if v in f else np.full_like(currentDensity_full, np.nan)
            data = data_full[mask]
            y_true_list.append(data)
        y_true = np.column_stack(y_true_list)

        cols = [6, 7, 8, 15, 16]

        # undo Box–Cox only on selected columns
        y_pred[:, cols] = inv_boxcox(y_pred[:, cols], lam, 0.0)

        cols = [6, 7, 15, 16]

        j = currentDensity
        j = j.reshape(-1)           # current density, length N
        frac = 1                  # fraction of data used in each local fit (tune this)

        #y_smooth = y_pred.copy()    # y_pred is after inv_boxcox

        for c in cols:
            y = y_pred[:, c]
            # LOWESS smoothing: local linear regression
            y_loc = sm.nonparametric.lowess(
                endog=y, exog=j,
                frac=frac,         # larger -> smoother, smaller -> more local detail
                it=1,              # robustifying iterations; increase if needed
                return_sorted=False
            )
            y_pred[:, c] = y_loc
        '''cols = [6, 7]
        j = currentDensity
        j = j.reshape(-1)          # shape (N,)
        X = np.column_stack([j**2, j, np.ones_like(j)])  # [j^2, j, 1]

        #y_smooth = y_pred.copy()

        for c in cols:
            y = y_pred[:, c]
            # least‑squares quadratic fit: y ≈ a*j^2 + b*j + c
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            #coeffs = np.polyfit(j, y, deg=3)   # [a, b, c, d]
            #p = np.poly1d(coeffs)              # polynomial object
            #y_pred[:, c] = p(j)
            y_pred[:, c] = X @ coef
        #plt.plot(currentDensity,y_true[:,15])
        #plt.plot(currentDensity,y_pred[:,15])    '''


        # Calculate and save RMSE per output
        for idx in range(len(output_names)):
            rmse_val = np.sqrt(np.nanmean((y_true[:, idx] - y_pred[:, idx]) ** 2))
            rmse_per_file_per_output[idx].append(rmse_val)

        # Plotting curves
        for idx, name in enumerate(output_names):
            ax = axes[idx]
            cidx = file_idx % len(colors)
            midx = file_idx % len(markers)
            h1, = ax.plot(currentDensity, y_true[:, idx], marker=markers[midx], linestyle='-', color=colors[cidx],
                          linewidth=1.1, markersize=3, alpha=0.7)
            h2, = ax.plot(currentDensity, y_pred[:, idx], marker=markers[midx], linestyle='--', color=colors[cidx],
                          linewidth=1.1, markersize=3, alpha=0.7)
            # For legend in first subplot only (avoids multiple legends)
            if idx == 0:
                # Collect handles for one per combination for legend
                legend_handles.append(Line2D([0], [0], color=colors[cidx], marker=markers[midx], linestyle='-', linewidth=1.1, markersize=6))
                legend_labels.append(f'CFD-{file_idx+1}')
                legend_handles.append(Line2D([0], [0], color=colors[cidx], marker=markers[midx], linestyle='--', linewidth=1.1, markersize=6))
                legend_labels.append(f'ANN-{file_idx+1}')

for idx, name in enumerate(output_names):
    ax = axes[idx]
    ax.set_xlabel('Current Density (A/m$^2$)', fontweight='bold', fontsize=6)
    ax.set_ylabel(name, fontweight='bold', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=5)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
    ax.grid(False)
    # RMSE annotations for all files for each output, slightly lower for axes[0] to fit legend
    y_start = 0.88 if idx == 0 else 0.98
    dy = 0.1
    for file_idx, rmse_val in enumerate(rmse_per_file_per_output[idx]):
        color = colors[file_idx % len(colors)]
        ax.text(0.99, y_start - file_idx*dy, f"RMSE: {rmse_val:.2e}",
                color=color, fontsize=6, fontweight='bold',
                ha='right', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
# Create legend handles with small marker size
from matplotlib.lines import Line2D

# Create dummy handles (one per file) same style as plotted curves
legend_handles = []
legend_labels = []


legend_handles = []
legend_labels = []

for file_idx, matfile in enumerate(matfiles):
    tokens = matfile.split('_')

    def extract_param(tokens, key):
        try:
            return tokens[tokens.index(key) + 1]
        except Exception:
            return 'NA'

    # Extract all necessary parameters
    Wgde_val = extract_param(tokens, 'Wgde')
    P_val = extract_param(tokens, 'P')
    vin_val = extract_param(tokens, 'vin')
    dpore_val = extract_param(tokens, 'dpore')
    dpored_val = extract_param(tokens, 'dpored')

    desc = f"Wgde={Wgde_val}, P={P_val}, vin={vin_val}, dpore={dpore_val}, dpored={dpored_val}"

    color = colors[file_idx % len(colors)]
    marker = markers[file_idx % len(markers)]

    legend_handles.append(Line2D([0], [0], color=color, marker=marker, linestyle='-', linewidth=1.1, markersize=3))
    legend_labels.append(f"CFD ({desc})")
    legend_handles.append(Line2D([0], [0], color=color, marker=marker, linestyle='--', linewidth=1.1, markersize=3))
    legend_labels.append(f"ANN ({desc})")




print(f"Number of handles: {len(legend_handles)}")
print(f"Number of labels: {len(legend_labels)}")
# Place legend outside the plot area
axes[0].legend(legend_handles, legend_labels, loc='upper center', fontsize=3, frameon=False)

plt.subplots_adjust(hspace=0.1, wspace=0.3)  # widen vertical and horizontal spacing
plt.show()
plt.pause(0.01)

#ee = (((y_pred[:,7])*lambd))**1/(lambd-1)
#plt.plot(currentDensity,y_true[:,7])
#plt.plot(currentDensity,ee) 

import pandas as pd
import os, re

def make_safe(name):
    name = name.replace(' ', '')
    name = name.replace('$','').replace('{','').replace('}','')
    return re.sub(r'[^A-Za-z0-9_]', '_', name)

# collect per-output, per-case series
per_output_case_data = [ {} for _ in output_names ]
all_j_values = set()

for file_idx, matfile in enumerate(matfiles):
    with h5py.File(matfile, 'r') as f:
        currentDensity = np.array(f['currentDensity']).squeeze()

        # rebuild y_true, y_pred as in your plotting code
        tokens = matfile.split('_')

        def extract_param(tokens, key):
            try:
                return float(tokens[tokens.index(key) + 1])
            except Exception:
                return np.nan

        input_scalar = [
            extract_param(tokens, 'Wgde'),
            extract_param(tokens, 'P'),
            extract_param(tokens, 'vin'),
            extract_param(tokens, 'dpore'),
            extract_param(tokens, 'Wsep')
        ]

        X_ann = np.array([input_scalar + [cd] for cd in currentDensity])
        X_ann_scaled = 2 * (X_ann - minIn) / (maxIn - minIn) - 1
        y_pred_scaled = model.predict(X_ann_scaled)
        y_pred = 0.5 * (y_pred_scaled + 1) * (maxOut - minOut) + minOut

        y_true_list = []
        for v in output_vars:
            data = np.array(f[v]).squeeze() if v in f else np.full_like(currentDensity, np.nan)
            y_true_list.append(data)
        y_true = np.column_stack(y_true_list)

    base = os.path.splitext(os.path.basename(matfile))[0]
    case_label = make_safe(base)

    # store j and update union
    all_j_values.update(currentDensity.tolist())

    for idx, name in enumerate(output_names):
        # map j -> values for this case
        series_dict = {}
        for j_val, y_t_val, y_p_val in zip(currentDensity, y_true[:, idx], y_pred[:, idx]):
            series_dict.setdefault(j_val, (y_t_val, y_p_val))

        per_output_case_data[idx][case_label] = series_dict

# build wide CSVs with union of all j
all_j_sorted = np.array(sorted(all_j_values))

for idx, name in enumerate(output_names):
    safe_name = make_safe(name)
    df = pd.DataFrame({'j_Am2': all_j_sorted})

    for case_label, series_dict in per_output_case_data[idx].items():
        cfd_col = []
        ann_col = []
        for j_val in all_j_sorted:
            if j_val in series_dict:
                cfd_val, ann_val = series_dict[j_val]
            else:
                cfd_val, ann_val = (np.nan, np.nan)
            cfd_col.append(cfd_val)
            ann_col.append(ann_val)
        df[f'CFD_{case_label}'] = cfd_col
        df[f'ANN_{case_label}'] = ann_col

    df.to_csv(f'DNNval_{safe_name}.csv', index=False)
    print(f"Saved {f'DNNval_{safe_name}.csv'}")



