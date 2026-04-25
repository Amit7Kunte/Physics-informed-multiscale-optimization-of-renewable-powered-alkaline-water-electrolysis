import random
import numpy as np
import tensorflow as tf
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import os

os.chdir(r"D:\Comsol_Tut\EquationBasedModelling\HTO_paper_models\latest_model_tcd\Neom_report_models\Publication Models\modified_model\Optimization_study\saved data\For Joule_pubPlots")

# fixed single seed
seed = 7084

mat = scipy.io.loadmat('preparedData_BoxCox_scaled.mat')
allInputsScaled = mat['allInputsScaled']
allOutputsScaled = mat['allOutputsScaled']

#weights = tf.constant(
#    [1., 1., 1., 1., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#    dtype = tf.float32
#)
'''outputVars = [
    'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O',
    'w_KOH_angl_out', 'w_KOH_gl_out', 'Q_gl_out', 'Q_angl_out','glsep_O2',
    'H2_mixedToHTO', 'Q_cond_h2cooler', 'Q_cond_ads',
    'Q_cond_deoxo', 'T_gl_out', 'T_angl_out', 'cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]'''

outputVars = [
    'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O',
    'w_KOH_angl_out', 'w_KOH_gl_out', 'Q_gl_out', 'Q_angl_out','glsep_O2',
     'Q_cond_h2cooler', 'Q_cond_ads',
    'Q_cond_deoxo', 'T_gl_out', 'T_angl_out', 'cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]


#def weighted_mse(y_true, y_pred):
#    sq = tf.square(y_true-y_pred)
#    return tf.reduce_mean(sq*weights)
#logfile = open('seed_loss_log.csv', 'w')
#logfile.write('seed,final_val_loss\n')


class LogLossRMSECallback(Callback):
    def __init__(self, seed, plotEpochFreq=20):
        super().__init__()
        self.plotEpochFreq = plotEpochFreq
        self.seed = seed
        self.train_loss = []
        self.val_loss = []
        self.fig = None
        self.ax = None

    def on_train_begin(self, logs=None):
        # create figure once
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (MSE)')
        self.ax.set_yscale('log')
        self.ax.grid(True)
        self.line_train, = self.ax.plot([], [], label='Train loss')
        self.line_val,   = self.ax.plot([], [], label='Val loss')
        self.ax.legend()
        plt.ion()
        plt.show(block=False)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        if (epoch + 1) % self.plotEpochFreq == 0 or (epoch + 1) == self.params['epochs']:
            epochs = range(1, len(self.train_loss) + 1)
            self.line_train.set_data(epochs, self.train_loss)
            self.line_val.set_data(epochs, self.val_loss)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()



# ---------------- SINGLE-SEED TRAINING ----------------

os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

totalSamples = allInputsScaled.shape[0]
idx = np.random.permutation(totalSamples)
numTrain = int(0.8 * totalSamples)
numVal = int(0.1 * totalSamples)

trainIdx = idx[:numTrain]
valIdx = idx[numTrain:numTrain + numVal]
testIdx = idx[numTrain + numVal:]

XTrain = allInputsScaled[trainIdx, :]
YTrain = allOutputsScaled[trainIdx, :]
XVal = allInputsScaled[valIdx, :]
YVal = allOutputsScaled[valIdx, :]
XTest = allInputsScaled[testIdx, :]
YTest = allOutputsScaled[testIdx, :]

inputSize = XTrain.shape[1]
outputSize = YTrain.shape[1]
hiddenSize = 256

start_time = time.time()

model = Sequential([
    InputLayer(input_shape=(inputSize,)),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    Dense(hiddenSize, activation='elu'),
    #Dense(hiddenSize, activation='elu'),
    #Dense(hiddenSize, activation='elu'),
    Dense(outputSize, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.002),
    loss='mean_squared_error'   #'mean_squared_error'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.75, patience=20, min_lr=1e-9, verbose=1
)

history = model.fit(
    XTrain, YTrain,
    epochs=500,
    batch_size=16,
    validation_data=(XVal, YVal),
    callbacks=[LogLossRMSECallback(seed, plotEpochFreq=20), reduce_lr],
    verbose=1
)

elapsed_time = time.time() - start_time
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
safe_seed = abs(int(seed)) % 100000
#model.save_weights('ANN_weights_seed6359.h5')
#logfile.write(f"{safe_seed},{final_val_loss}\n")
#logfile.flush()

print(f"Elapsed time for seed {seed}: {elapsed_time:.2f} seconds")
YPred = model.predict(XTest)
rmse_test = np.sqrt(np.mean((YPred - YTest) ** 2))
print(f'Test RMSE (seed {seed}): {rmse_test:.6f}')


from sklearn.metrics import r2_score
import pandas as pd

minOut = mat['minOut'].flatten()
maxOut = mat['maxOut'].flatten()
unscale = lambda ys: 0.5 * (ys + 1) * (maxOut - minOut) + minOut

#XTest_unscaled = unscale(XTest)
YTest_unscaled = unscale(YTest)
YPred_unscaled = unscale(YPred)

from scipy.special import inv_boxcox
# or: from scipy.stats import boxcox, boxcox_normmax, boxcox_llf

# indices that were Box–Cox–transformed during preprocessing
cols_bc = [6, 7, 8, 14, 15]

# lam must be exactly the same vector you saved when doing the Box–Cox
lam = -0.5   # or whatever you named it in MATLAB

# lam is a scalar (float), same for all Box–Cox outputs
for c in cols_bc:
    YTest_unscaled[:, c] = inv_boxcox(YTest_unscaled[:, c], lam)
    YPred_unscaled[:, c] = inv_boxcox(YPred_unscaled[:, c], lam)



import statsmodels.api as sm
currentDensity_index = 4          # 5th column, 0‑based index
j = XTest[:, currentDensity_index].reshape(-1)
cols_smooth = [6, 7, 14, 15]  # the columns you want smoothed


frac = 1.0  # tune if you want more/less smoothing

'''for c in cols_smooth:
    y = YPred_unscaled[:, c]
    y_loc = sm.nonparametric.lowess(
        endog=y,
        exog=j,
        frac=frac,
        it=1,
        return_sorted=False
    )
    YPred_unscaled[:, c] = y_loc
'''

cols_scale = [2, 6, 7 ,8 ]
factors = np.array([22.414*3600, 1000*3600, 1000*3600.0, 22.414*3600])  # one factor per column, all rates converted to LPH

YTest_unscaled[:, cols_scale] *= factors
YPred_unscaled[:, cols_scale] *= factors


r2_unscaled = []
rmse_unscaled = []
for j in range(outputSize):
    y_true = YTest_unscaled[:, j]
    y_pred = YPred_unscaled[:, j]
    r2_unscaled.append(r2_score(y_true, y_pred))
    rmse_unscaled.append(np.sqrt(np.mean((y_true - y_pred) ** 2)))

print("Unscaled test R²:",   [f"{r:.4f}" for r in r2_unscaled])
print("Unscaled test RMSE:", [f"{e:.3e}" for e in rmse_unscaled])
if final_val_loss < 1e-5:
    outdir = "ANN_saved"
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f'trainedANN_seed{safe_seed}_boxcox_3.keras')
    model.save(filename)
    print(f"Finished training for seed {safe_seed}, model saved as trainedANN_seed{safe_seed}.keras")


    # metrics CSV for this seed (unscaled)
    metrics_df = pd.DataFrame({
        'outputVar': outputVars,
        'R2_test_unscaled':   r2_unscaled,
        'RMSE_test_unscaled': rmse_unscaled
    })
    metrics_df.to_csv(f"ANN_test_metrics_unscaled_seed{safe_seed}.csv", index=False)

    
    
   # print(f"Finished training for seed {safe_seed}, files saved as:")
   # print(" ",json_path)
   # print(" ",weights_path)

else:
    print(f"Validation loss {final_val_loss:.2e} for seed {safe_seed} not better than threshold, model not saved.")
    
from sklearn.metrics import r2_score

# Per‑output R² and RMSE on TEST set
n_outputs = YTest.shape[1]
r2_list   = []
rmse_list = []

for j in range(n_outputs):
    y_true = YTest_unscaled[:, j]
    y_pred = YPred_unscaled[:, j]
    r2     = r2_score(y_true, y_pred)
    rmse   = np.sqrt(np.mean((y_true - y_pred)**2))
    r2_list.append(r2)
    rmse_list.append(rmse)

# del r2_list
# Save for later plotting
#np.savez(f"test_metrics_seed{safe_seed}.npz",
#        r2=np.array(r2_list),
#         rmse=np.array(rmse_list),
#        outputVars=np.array(outputVars, dtype=object))


import pandas as pd

# choose which outputs to plot (by index)
out_indices = [1, 2, 3, 4, 10]  # SEC_sys, maxT, SEC_stack, vap_h2_pdt, H_T_O, H2_mixedToHTO
output_labels = [
    "ΔT$_{stack}$",
    "SEC$_{stack}$",
    "H$_2$$_{sys}$",
    "HTO",
    "H$_2$ mix/HTO"
]

n_outputs = len(out_indices)
ncols = int(np.ceil(np.sqrt(n_outputs)))
nrows = int(np.ceil(n_outputs / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows))
axes = axes.flatten()

for i, (ax, idx) in enumerate(zip(axes, out_indices)):
    y_true = YTest_unscaled[:, idx].flatten()
    y_pred = YPred_unscaled[:, idx].flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2   = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, s=40, alpha=0.6, edgecolor='k')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2)

    ax.set_title(output_labels[i], fontsize=14, pad=8)
    ax.set_xlabel('True (scaled)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted (scaled)', fontsize=11, fontweight='bold')

    ax.text(
        0.05, 0.88,
        f"R\u00b2 = {r2:.4f}\nRMSE = {rmse:.3e}",
        fontsize=12,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        transform=ax.transAxes
    )

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.set_aspect('equal', adjustable='datalim')

# turn off unused axes if any
for j in range(n_outputs, len(axes)):
    axes[j].axis('off')

#lt.tight_layout()
#plt.show()
#plt.pause(0.01)
#from sklearn.metrics import r2_score
#import pandas as pd

# --- Unscale TEST data (back to physical units) ---
#minOut = mat['minOut'].flatten()
#maxOut = mat['maxOut'].flatten()
#unscale = lambda ys: 0.5 * (ys + 1) * (maxOut - minOut) + minOut

#YTest_unscaled = unscale(YTest)
#YPred_unscaled = unscale(YPred)

# --- Per‑output R² and RMSE on UNscaled TEST set ---
'''r2_unscaled = []
rmse_unscaled = []

for j in range(outputSize):
    y_true = YTest_unscaled[:, j]
    y_pred = YPred_unscaled[:, j]
    r2_unscaled.append(r2_score(y_true, y_pred))
    rmse_unscaled.append(np.sqrt(np.mean((y_true - y_pred) ** 2)))

print("Unscaled test R²:", [f"{r:.4f}" for r in r2_unscaled])
print("Unscaled test RMSE:", [f"{e:.3e}" for e in rmse_unscaled])'''

# --- (Optional) save for Origin ---
metrics_df = pd.DataFrame({
    'outputVar': outputVars,
    'R2_test_unscaled': r2_unscaled,
    'RMSE_test_unscaled': rmse_unscaled
})
metrics_df.to_csv(f"ANN_test_metrics_unscaled_seed{safe_seed}.csv", index=False)

out_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, 15, 16,17]
output_labels_phys  = [
    'maxT', 'SEC_stack', 'vap_h2_pdt', 'H_T_O',
    'w_KOH_angl_out', 'w_KOH_gl_out', 'Q_gl_out', 'Q_angl_out','glsep_O2',
    'Q_cond_h2cooler', 'Q_cond_ads',
    'Q_cond_deoxo', 'T_gl_out', 'T_angl_out','cell_delP','ancell_delP','eta_volt','eta_curr_stack'
]

n_outputs = len(out_indices)
ncols = int(np.ceil(np.sqrt(n_outputs)))
nrows = int(np.ceil(n_outputs / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows))
axes = axes.flatten()

for i, (ax, idx) in enumerate(zip(axes, out_indices)):
    y_true = YTest_unscaled[:, idx].flatten()
    y_pred = YPred_unscaled[:, idx].flatten()
    r2   = r2_unscaled[idx]
    rmse = rmse_unscaled[idx]

    ax.scatter(y_true, y_pred, s=40, alpha=0.6, edgecolor='k')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2)

    ax.set_title(output_labels_phys[i], fontsize=14, pad=8)
    ax.set_xlabel('True (phys. units)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted (phys. units)', fontsize=11, fontweight='bold')

    ax.text(
        0.05, 0.88,
        f"R\u00b2 = {r2:.4f}\nRMSE = {rmse:.3e}",
        fontsize=12,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        transform=ax.transAxes
    )

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_aspect('equal', adjustable='datalim')

'''for j in range(n_outputs, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
plt.pause(0.01)'''
import pandas as pd

for i, idx in enumerate(out_indices):
    df = pd.DataFrame({
        'True_unscaled': YTest_unscaled[:, idx],
        'Pred_unscaled': YPred_unscaled[:, idx],
    })
    df.to_csv(f'fitness_unscaled_output{idx}_seed{safe_seed}.csv', index=False)


#logfile.close()
