
import random
import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
os.chdir(r"D:\Comsol_Tut\EquationBasedModelling\HTO_paper_models\latest_model_tcd\Neom_report_models\Publication Models\modified_model\Optimization_study\saved data")
plt.ion()  # Turn on interactive plotting mode

# Load multi-output training data
mat = scipy.io.loadmat('preparedData_capex_rated75_heatmap.mat')
X = mat['allInputscapexScaled']
y = mat['allOutputcapexHeatmapScaled']
minOut = mat['minOutcapexHeat'].flatten()
maxOut = mat['maxOutcapexHeat'].flatten()
if len(y.shape) == 1:
    y = y.reshape(-1, 1)

# Set fixed seed for full reproducibility
seed = 17738
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Data split (80% train, 10% val, 10% test)
totalSamples = X.shape[0]
idx = np.random.permutation(totalSamples)
numTrain = int(0.8 * totalSamples)
numVal = int(0.1 * totalSamples)
trainIdx = idx[:numTrain]
valIdx = idx[numTrain:numTrain+numVal]
testIdx = idx[numTrain+numVal:]

XTrain = X[trainIdx, :]
yTrain = y[trainIdx, :]
XVal = X[valIdx, :]
yVal = y[valIdx, :]
XTest = X[testIdx, :]
yTest = y[testIdx, :]

inputSize = X.shape[1]
outputSize = y.shape[1]        # should now be 9
hiddenSize = 256

class LogLossRMSECallback(tf.keras.callbacks.Callback):
    def __init__(self, plotEpochFreq=20):
        super().__init__()
        self.plotEpochFreq = plotEpochFreq
        self.train_loss = []
        self.val_loss = []
        self.train_rmse = []
        self.val_rmse = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        self.train_rmse.append(np.sqrt(logs.get('loss', 0)))
        self.val_rmse.append(np.sqrt(logs.get('val_loss', 0)))
        if (epoch + 1) % self.plotEpochFreq == 0:
            plt.figure(201)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.semilogy(range(1, len(self.train_loss) + 1), self.train_loss, '-', label='Train Loss')
            plt.semilogy(range(1, len(self.val_loss) + 1), self.val_loss, 'r-', label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Log-scale Training & Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.semilogy(range(1, len(self.train_rmse) + 1), self.train_rmse, '-', label='Train RMSE')
            plt.semilogy(range(1, len(self.val_rmse) + 1), self.val_rmse, 'r-', label='Val RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title('Log-scale Training & Validation RMSE')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.pause(0.01)

# Model definition: output layer size = 9 for multi-output regression
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(inputSize,)),
    tf.keras.layers.Dense(hiddenSize, activation='elu'),
    tf.keras.layers.Dense(hiddenSize, activation='elu'),
    tf.keras.layers.Dense(hiddenSize, activation='elu'),
    tf.keras.layers.Dense(hiddenSize, activation='elu'),
    tf.keras.layers.Dense(hiddenSize, activation='elu'),
    tf.keras.layers.Dense(hiddenSize, activation='elu'),
   # tf.keras.layers.Dense(hiddenSize, activation='elu'),
   # tf.keras.layers.Dense(hiddenSize, activation='elu'),
    tf.keras.layers.Dense(outputSize, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.008),
    loss='mean_squared_error'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=20, min_lr=1e-9, verbose=1
)

history = model.fit(
    XTrain, yTrain,
    epochs=500,
    batch_size=8,
    validation_data=(XVal, yVal),
    callbacks=[LogLossRMSECallback(plotEpochFreq=20), reduce_lr],
    verbose=1
)

plt.ioff()
plt.figure(201)
plt.show()

# ---- Evaluate on test set (for each output separately) ----
yTestPred = model.predict(XTest)
r2_test_all = [r2_score(yTest[:,i], yTestPred[:,i]) for i in range(outputSize)]
r2_test_avg = np.mean(r2_test_all)

# Validation predictions
yValPred = model.predict(XVal)

# Merged val+test
#y_merge = np.concatenate([yVal, yTest], axis=0)
#y_pred_merge = np.concatenate([yValPred, yTestPred], axis=0)
y_merge = np.concatenate([yTest], axis=0)
y_pred_merge = np.concatenate([yTestPred], axis=0)
r2_merge_all = [r2_score(y_merge[:,i], y_pred_merge[:,i]) for i in range(outputSize)]
r2_merge_avg = np.mean(r2_merge_all)

final_val_loss = history.history['val_loss'][-1]

print(f"Test R² (each output): {[f'{r:.4f}' for r in r2_test_all]}")
print(f"Test R² (average): {r2_test_avg:.4f}")
print(f"Merged Val+Test R² (each): {[f'{r:.4f}' for r in r2_merge_all]}")
print(f"Merged Val+Test R² (average): {r2_merge_avg:.4f}")
print(f"Final validation loss: {final_val_loss:.2e}")

# --------------- Scatter plots for all outputs ---------------
plt.figure(figsize=(20, 20))
for out_idx in range(outputSize):
    plt.subplot(3,3,out_idx+1)
    plt.scatter(yTest[:,out_idx], yTestPred[:,out_idx], s=30, alpha=0.7, edgecolors='k', label=f'Test data')
    b, a = np.polyfit(yTest[:,out_idx], yTestPred[:,out_idx], deg=1)
    yseq = np.linspace(min(yTest[:,out_idx]), max(yTest[:,out_idx]), 100)
    plt.plot(yseq, a + b * yseq, color='red', lw=2, label='Fit')
    plt.plot(yseq, yseq, 'k--', lw=1.2, label='Ideal')
    plt.xlabel(f'Test Value (Target {out_idx+1})')
    plt.ylabel(f'Pred. (Output {out_idx+1})')
    plt.title(f'Output {out_idx+1}: $R^2$={r2_test_all[out_idx]:.3f}')
    plt.legend(fontsize=9)
    plt.grid(True)
plt.suptitle('Multi-Output CAPEX ANN Regression Results (Test Set)', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# Unscale both prediction and targets
# Unscale predictions and targets to $/kW
unscale = lambda ys: 0.5 * (ys + 1) * (maxOut - minOut) + minOut
y_pred_unscaled = unscale(yTestPred)
y_true_unscaled = unscale(yTest)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
for i in range(outputSize):
    # Compute RMSE for current component/output
    rmse_val = np.sqrt(np.mean((y_pred_unscaled[:, i] - y_true_unscaled[:, i])**2))
    
    plt.subplot(3, 3, i+1)
    plt.scatter(y_true_unscaled[:, i], y_pred_unscaled[:, i], s=30, alpha=0.7, edgecolors='k', label='Test data')
    b, a = np.polyfit(y_true_unscaled[:, i], y_pred_unscaled[:, i], deg=1)
    yseq = np.linspace(min(y_true_unscaled[:, i]), max(y_true_unscaled[:, i]), 100)
    plt.plot(yseq, a + b * yseq, color='red', lw=2, label='Fit')
    plt.plot(yseq, yseq, 'k--', lw=1.2, label='Ideal')
    plt.xlabel(f'Test Target (Output {i+1}) [$\\$/kW$]')
    plt.ylabel(f'Prediction (Output {i+1}) [$\\$/kW$]')
    plt.title(f'Output {i+1}: RMSE={rmse_val:.2f} $/kW')
    plt.legend(fontsize=9)
    plt.grid(True)
plt.suptitle('Multi-Output CAPEX ANN Regression Results (Test Set) - Unscaled RMSE', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
plt.pause(0.01)
# --------------- Save model ---------------
model.save('trainedANN_capex_breakup.keras')
