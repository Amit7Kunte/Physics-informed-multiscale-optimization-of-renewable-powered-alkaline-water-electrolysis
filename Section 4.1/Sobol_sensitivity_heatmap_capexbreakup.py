import os
import numpy as np
import scipy.io
import tensorflow as tf
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#os.chdir(r"D:\Comsol_Tut\EquationBasedModelling\HTO_paper_models\latest_model_tcd\Neom_report_models\Publication Models\modified_model\Optimization_study\saved data\For Joule_pubPlots")

# Load scaling parameters and data
scaling = scipy.io.loadmat('preparedData_capex_rated75_heatmap.mat')
minIn = scaling['minIncapex'].flatten()
maxIn = scaling['maxIncapex'].flatten()
minOut = scaling['minOutcapexHeat'].flatten()
maxOut = scaling['maxOutcapexHeat'].flatten()

problem = {
    'num_vars': 5,
    'names': ['Wgde', 'P', 'vin', 'dpore', 'Wsep'],
    'bounds': [(minIn[i], maxIn[i]) for i in range(5)]
}


# Load trained ANN model (8 output)
model = tf.keras.models.load_model('trainedANN_capex_breakup.keras')


def scale_inputs(X):
    X_scaled = np.empty_like(X)
    for i in range(5):
        X_scaled[:, i] = 2 * (X[:, i] - minIn[i]) / (maxIn[i] - minIn[i]) - 1
    return X_scaled


def unscale_outputs(Y_scaled):
    return 0.5 * (Y_scaled + 1) * (maxOut - minOut) + minOut


# CAPEX breakup output labels (8 total)
main_labels = [
    r'$\mathbf{Total~CAPEX}$',
    r'$\mathbf{Total~CAPEX/kW}$',
    r'$\mathbf{H_2,compression}$',
    r'$\mathbf{Refrigeration}$',
    r'$\mathbf{Gas{-}liq.~separation}$',
    r'$\mathbf{H_2,purification}$',
    r'$\mathbf{Lye,coolers}$',
    r'$\mathbf{Pumps}$',
    r'$\mathbf{Stack}$',
]
unit = r'$\mathbf{(\$/\mathbf{kW})}$'
output_labels = [lbl + ' ' + unit for lbl in main_labels]



input_labels = [
    r'$\mathbf{W_{GDE}}$',
    r'$\mathbf{P}$',
    r'$\mathbf{V_{in}}$',
    r'$\mathbf{d_p}$',
    r'$\mathbf{W_{SEP}}$'
]

# Generate Saltelli samples
n_samples = 1024*4
param_vals = saltelli.sample(problem, n_samples, calc_second_order=False)
param_vals_scaled = scale_inputs(param_vals)
y_pred_scaled = model.predict(param_vals_scaled, verbose=0)
y_pred = unscale_outputs(y_pred_scaled)  # unscaled predictions

print(param_vals.shape[0], y_pred.shape[0])
expected = n_samples * (problem['num_vars'] + 2)  # 1024 * (5+2) = 7168

print("Expected:", expected)
assert param_vals.shape[0] == expected, "Saltelli sample mismatch"
assert y_pred.shape[0] == expected, "Prediction sample mismatch"

S1_matrix = np.zeros((y_pred.shape[1], problem['num_vars']))
ST_matrix = np.zeros_like(S1_matrix)

for i in range(y_pred.shape[1]):
    Si = sobol.analyze(problem, y_pred[:, i], calc_second_order=False, print_to_console=False)
    S1_matrix[i, :] = Si['S1']
    ST_matrix[i, :] = Si['ST']

# Pearson correlation matrix for directionality
pearson_matrix = np.zeros_like(S1_matrix)
for i in range(y_pred.shape[1]):
    for j in range(problem['num_vars']):
        pearson_matrix[i, j] = np.corrcoef(param_vals[:, j], y_pred[:, i])[0, 1]

S1_colored = S1_matrix * np.sign(pearson_matrix)
ST_colored = ST_matrix * np.sign(pearson_matrix)

'''fig, axes = plt.subplots(1, 2, figsize=(20, 14))

sns.heatmap(
    S1_colored, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
    xticklabels=input_labels, yticklabels=output_labels, ax=axes[0],
    linecolor='black', linewidths=0.5, vmin=-1, vmax=1
)
cbar_ax0 = axes[0].figure.axes[-1]
for label in cbar_ax0.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(11)

axes[0].set_title('First Order Sobol Indices (S1) with Pearson Direction', fontsize=16)
axes[0].set_xlabel('Inputs', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Outputs', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)

sns.heatmap(
    ST_colored, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
    xticklabels=input_labels, yticklabels=output_labels, ax=axes[1],
    linecolor='black', linewidths=0.5, vmin=-1, vmax=1
)
cbar_ax1 = axes[1].figure.axes[-1]
for label in cbar_ax1.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(11)

axes[1].set_title('Total Order Sobol Indices (ST) with Pearson Direction', fontsize=16)
axes[1].set_xlabel('Inputs', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Outputs', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)
# ... (after your sns.heatmap call)
axes[0].set_yticklabels(output_labels, fontsize=10) # change 10 to your preferred size
axes[1].set_yticklabels(output_labels, fontsize=10)'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def custom_fmt(x):
    return "0.00" if abs(x) < 0.01 else f"{x:.2f}"  # use .2f for visual match to your plot

# Build the annotation array
annot_arr = np.empty_like(ST_colored, dtype=object)
for i in range(ST_colored.shape[0]):
    for j in range(ST_colored.shape[1]):
        annot_arr[i, j] = custom_fmt(ST_colored[i, j])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    ST_colored,
    annot=annot_arr,  # <-- use this, not True!
    fmt="",           # <-- string passthrough
    cmap="RdYlGn", center=0,
    xticklabels=input_labels, yticklabels=output_labels, ax=ax,
    linecolor='black', linewidths=0.7, vmin=-1, vmax=1,
    annot_kws={"size": 10, "weight": "bold", "color": "black"},
)

ax.set_yticklabels(output_labels, fontsize=11)
ax.set_xticklabels(input_labels, fontsize=14, fontweight='bold', rotation=45)
cbar_ax = ax.figure.axes[-1]
for label in cbar_ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(14)
#ax.set_title('Total Order Sobol Indices (ST) with Pearson Direction', fontsize=18)
#ax.set_xlabel('Inputs', fontsize=16, fontweight='bold')
#ax.set_ylabel('Outputs', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()
plt.pause(0.01)
df_ST = pd.DataFrame(
    ST_colored,
    columns=[lbl.strip('$') for lbl in input_labels],
    index=[lbl.strip('$') for lbl in output_labels]
)

# Transpose so inputs become columns (X, bottom) in Origin
df_ST_T = df_ST.T
df_ST_T_revcols = df_ST_T.iloc[:, ::-1]
df_ST_T_revcols.to_csv("sobol_ST_colored_T.csv")


'''# Similarly for S1_colored if needed
df_S1 = pd.DataFrame(
    S1_colored,
    columns=[lbl.strip('$') for lbl in input_labels],
    index=[lbl.strip('$') for lbl in output_labels]
)
df_S1.to_csv("sobol_S1_colored.csv")'''