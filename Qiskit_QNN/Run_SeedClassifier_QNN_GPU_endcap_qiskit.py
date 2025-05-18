"""
QNN Model for Endcap Dataset using Qiskit and PyTorch.

This script implements a binary classification QNN similarly to our Qiskit-based
implementation for the barrel dataset, but here we use the endcap dataset.
The differences are mostly in the data loading (i.e. using the Endcap pickle and applying
the endcap selection for tsos_eta) and in the naming of output files.
The quantum circuit is built using Qiskit (Aer simulator) and then wrapped using TorchConnector
so that it can be optimized using PyTorch.
"""

########################################################################################################
# Data IO and Preprocessing
########################################################################################################
import sys
import os
import pickle
import pandas as pd

# Add parent directory to sys.path if needed
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from BDT_model.HLTIO import preprocess

# Path to the Endcap pickle file
pkl_path = "../DYToLL_PU200_Spring23_NThltIter2FromL1/DYToLL_PU200_Spring23_NThltIter2FromL1_Endcap.pkl"

# Load the pickle file
with open(pkl_path, "rb") as file:
    data = pickle.load(file)

# Assume the first element of the tuple is the DataFrame
df = data[0]

# pT cut
df = df[df['gen_pt'] < 1e9]
df = df[df['gen_pt'] > 0]

# Apply setClassLabel to compute the class labels
df = preprocess.setClassLabel(df)
# Compute the distance features (drop extra variables as per original flow)
df = preprocess.addDistHitL1Tk(df, addAbsDist=False)
# Endcap selection: select events with |tsos_eta| > 1.2
df = df[((df['tsos_eta'] > 1.2) | (df['tsos_eta'] < -1.2))].copy()

# Define the list of required columns
required_columns = [
    "expd2hitl1tk1",
    "expd2hitl1tk2",
    "expd2hitl1tk3",
    "dR_L1TkMuSeedP",
    "dPhi_L1TkMuSeedP",
    "tsos_qbp",
    "tsos_dydz",
    "tsos_dxdz",
    "tsos_err0",
    "tsos_err2",
    "tsos_err5",
    "y_label"  # class label from setClassLabel
]

# Check if all required columns are present.
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("Warning: The following required columns are missing:", missing)

# Create a new DataFrame with the required columns and fill NaNs with -1
df_final = df[required_columns].copy()
df_final = df_final.fillna(-1.)

print(df_final.head())
print(df_final[df_final["y_label"] == 0])

########################################################################################################
# Data Slicing and Standardization
########################################################################################################
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Convert DataFrame to numpy arrays
X_np = df_final.drop(columns=["y_label"]).values.astype(np.float32)
y_np = df_final["y_label"].values.astype(np.float32).reshape(-1, 1)

# Split the data into train and test sets (80% train, 20% test) with stratification
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

# Compute scaling parameters using training data
scaler = StandardScaler()
scaler.fit(x_train_np)
stdTransPar = [scaler.mean_, scaler.scale_]

# Save the scaling parameters for future inference
scalefiles_dir = "scalefiles"
if not os.path.exists(scalefiles_dir):
    os.makedirs(scalefiles_dir)
scale_filepath = os.path.join(scalefiles_dir, "endcap_qnn_scale.txt")
with open(scale_filepath, "w") as f_scale:
    f_scale.write("%s\n" % str(scaler.mean_.tolist()))
    f_scale.write("%s\n" % str(scaler.scale_.tolist()))

# Standardize training and test data using the fixed transformation from BDT_model
x_train_np, x_test_np = preprocess.stdTransformFixed(x_train_np, x_test_np, stdTransPar)

# Convert standardized numpy arrays to torch tensors.
x_train = torch.tensor(x_train_np, dtype=torch.float32)
x_test  = torch.tensor(x_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test  = torch.tensor(y_test_np, dtype=torch.float32)

# Set GPU device (using GPU 1 here)
gpu_id = 1
torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
x_train = x_train.to(device)
x_test  = x_test.to(device)
y_train = y_train.to(device)
y_test  = y_test.to(device)

# Print train and test class distributions
print("Training set class distribution:")
print(pd.Series(y_train.cpu().numpy().flatten()).value_counts())
print("Test set class distribution:")
print(pd.Series(y_test.cpu().numpy().flatten()).value_counts())

########################################################################################################
# QNN Model Definition using Qiskit and PyTorch
########################################################################################################
import torch.nn as nn
import torch.optim as optim

# Qiskit and Qiskit Machine Learning imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as EstimatorQNNV1
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator, Estimator as EstimatorV1

# Set the number of qubits and circuit depth (same as your Pennylane version)
n_qubits = 11  
q_depth = 5   # Depth as defined originally

# Create a quantum circuit with n_qubits
qc = QuantumCircuit(n_qubits)

# Define input parameters (one per qubit)
x_params = [Parameter(f"x{i}") for i in range(n_qubits)]

# ---------------------------
# Embedding Block (mimicking QAOAEmbedding)
# ---------------------------
# For each depth layer 'd' and for every qubit 'i', apply a Ry and Rz gate
# with gate angles given by the product of the input and a trainable parameter.
embedding_params = []  # Will hold parameters for the embedding layer
for d in range(q_depth):
    for i in range(n_qubits):
        p_e1 = Parameter(f"w_e_{d}_{i}_0")
        p_e2 = Parameter(f"w_e_{d}_{i}_1")
        embedding_params.extend([p_e1, p_e2])
        # Multiply the input by the corresponding embedding parameter.
        qc.ry(x_params[i] * p_e1, i)
        qc.rz(x_params[i] * p_e2, i)
    qc.barrier()  # Optional barrier for clarity

# ---------------------------
# Entangling Block (mimicking StronglyEntanglingLayers)
# ---------------------------
# For each depth layer 'd' and for every qubit 'i', apply a U3 gate with three trainable parameters.
# Then apply a chain of CX (CNOT) gates to entangle neighboring qubits.
entangling_params = []  # Will hold parameters for the entangling layers
for d in range(q_depth):
    for i in range(n_qubits):
        p2_a = Parameter(f"w2_{d}_{i}_0")
        p2_b = Parameter(f"w2_{d}_{i}_1")
        p2_c = Parameter(f"w2_{d}_{i}_2")
        entangling_params.extend([p2_a, p2_b, p2_c])
        qc.u(p2_a, p2_b, p2_c, i)
    # Apply a chain of CNOT gates across adjacent qubits
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
    qc.barrier()  # Optional barrier for clarity

# Combine all weight parameters: first the embedding parameters then the entangling ones.
weight_params = embedding_params + entangling_params

# --- Simulator and Estimator Setup ---
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
# Import the V2 primitives and QNN classes
from qiskit.primitives import StatevectorEstimator, Estimator as EstimatorV1
from qiskit_machine_learning.neural_networks import EstimatorQNN as EstimatorQNNV1
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Define the observable: Measure the expectation value of Z on qubit 0.
observable = SparsePauliOp("Z" + "I" * (n_qubits - 1))

# For newer Qiskit versions, use StatevectorEstimator (V2)
estimator = StatevectorEstimator()

# Use the appropriate QNN class - V2 compatible version
# If direct V2 EstimatorQNN is available
try:
    # Try using V2 version directly
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=x_params,
        weight_params=weight_params,
        observables=observable,
        estimator=estimator
    )
except TypeError:
    # Fallback to a different initialization if V2 API is different
    # Import the backwards compatibility module if needed
    from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    
    # Create QNN with different parameter approach for V2
    qnn = EstimatorQNN(
        circuit=qc,
        input_parameters=x_params,
        weight_parameters=weight_params,
        observables=observable,
        estimator=estimator
    )

# Wrap the QNN with TorchConnector so it can be used as a PyTorch module.
qiskit_qnn = TorchConnector(qnn)

# Define the PyTorch model by wrapping the qiskit_qnn.
class PureQNN(nn.Module):
    def __init__(self):
        super(PureQNN, self).__init__()
        self.qnn = qiskit_qnn

    def forward(self, x):
        # The raw output of the QNN is assumed to be in [-1, 1].
        # Map it to [0, 1] as a probability.
        return (self.qnn(x) + 1) / 2

# Instantiate the model and move it to the GPU device.
model = PureQNN().to(device)

# Define optimizer, learning rate scheduler, and loss function.
optimizer = optim.AdamW(model.parameters(), lr=0.01)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
criterion = nn.BCELoss()  # Binary cross-entropy loss for classification

########################################################################################################
# Training Loop
########################################################################################################
from tqdm import tqdm
epochs = 100
batch_size = 1000
train_losses = []
test_losses = []

for epoch in tqdm(range(epochs), desc="Epochs", position=0, leave=True):
    model.train()
    running_loss = 0.0
    perm = torch.randperm(len(x_train))
    batch_steps = list(range(0, len(x_train), batch_size))
    
    for i in tqdm(batch_steps, desc=f"Epoch {epoch+1} Batches", leave=True, position=1):
        indices = perm[i : i + batch_size]
        x_batch, y_batch = x_train[indices], y_train[indices]
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / (len(x_train) / batch_size)
    train_losses.append(avg_train_loss)
    
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = criterion(y_test_pred, y_test).item()
        test_losses.append(test_loss)
    
    scheduler.step(test_loss)
    
    y_test_prob = y_test_pred.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()
    y_pred_labels = (y_test_prob >= 0.5).astype(int)
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
    roc_auc = roc_auc_score(y_test_np, y_test_prob)
    accuracy = accuracy_score(y_test_np, y_pred_labels)
    
    tqdm.write(f"Epoch {epoch+1}/{epochs}")
    tqdm.write(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
    tqdm.write(f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
    tqdm.write("Confusion Matrix:")
    tqdm.write(f"{confusion_matrix(y_test_np, y_pred_labels)}")
    
    if epoch == 0:
        best_test_loss = test_loss
        patience_counter = 0
    else:
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 100:
                tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                break

########################################################################################################
# Save the Model and Training Information
########################################################################################################
os.makedirs('./qiskit_endcap_result', exist_ok=True)
torch.save(model.state_dict(), './qiskit_endcap_result/qnn_seed_classifier_weights_endcap_qiskit.pt')

training_info = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_test_loss': best_test_loss,
    'epochs_completed': epoch + 1,
    'hyperparameters': {
        'n_qubits': n_qubits,
        'q_depth': q_depth,
        'batch_size': batch_size,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
}
torch.save(training_info, './qiskit_endcap_result/qnn_training_info_endcap_qiskit.pt')

########################################################################################################
# Plot Training and Evaluation Metrics
########################################################################################################
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Test Loss")
plt.savefig("./qiskit_endcap_result/Training_Test_Loss_endcap_qiskit.png", dpi=300, bbox_inches='tight')

from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools

with torch.no_grad():
    y_test_pred = model(x_test)
    y_test_prob = y_test_pred.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()
    y_pred_labels = (y_test_prob >= 0.5).astype(int)
    final_conf_matrix = confusion_matrix(y_test_np, y_pred_labels)

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test_np, y_test_prob)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_val:.4f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("./qiskit_endcap_result/ROC_Curve_endcap_qiskit.png", dpi=300, bbox_inches='tight')

# Confusion Matrix Plot
plt.figure(figsize=(5, 5))
plt.imshow(final_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['0', '1'])
plt.yticks(tick_marks, ['0', '1'])
thresh = final_conf_matrix.max() / 2.
for i, j in itertools.product(range(final_conf_matrix.shape[0]), range(final_conf_matrix.shape[1])):
    plt.text(j, i, format(final_conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if final_conf_matrix[i, j] > thresh else "black")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("./qiskit_endcap_result/Confusion_Matrix_endcap_qiskit.png", dpi=300, bbox_inches='tight')

# Output Score Distribution Plot
mask_signal = (y_test_np == 1)
mask_background = (y_test_np == 0)
plt.figure(figsize=(6, 4))
plt.hist(y_test_prob[mask_signal], bins=20, alpha=0.4, label="Signal (1)", color="blue", density=True)
plt.hist(y_test_prob[mask_background], bins=20, alpha=0.4, label="Background (0)", color="red", density=True)
plt.xlabel("Predicted Score")
plt.ylabel("Density")
plt.title("Output Score Distribution")
plt.legend()
plt.savefig("./qiskit_endcap_result/Output_Score_Distribution_endcap_qiskit.png", dpi=300, bbox_inches='tight')

# Normalized Confusion Matrix Plot
cm = confusion_matrix(y_test_np, y_pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(5, 5))
plt.imshow(cm_normalized, interpolation='nearest', cmap='viridis')
plt.title("Normalized Confusion Matrix")
plt.colorbar(shrink=0.8)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['0', '1'])
plt.yticks(tick_marks, ['0', '1'])
thresh = cm_normalized.max() / 2.
for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
    plt.text(j, i, f"{cm_normalized[i, j]:.3f}",
             horizontalalignment="center",
             color="black" if cm_normalized[i, j] > thresh else "white")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("./qiskit_endcap_result/Normalized_Confusion_Matrix_endcap_qiskit.png", dpi=300, bbox_inches='tight')

# Test Score Distribution Plot (with log scale on y-axis)
plt.figure(figsize=(6, 4))
plt.hist(y_test_prob[mask_signal], bins=20, alpha=0.4, label="Signal (1)", color="blue", density=True)
plt.hist(y_test_prob[mask_background], bins=20, alpha=0.4, label="Background (0)", color="red", density=True)
plt.xlabel("Predicted Score")
plt.ylabel("a.u.")
plt.title("Test Score Distribution")
plt.grid(True, ls="--", alpha=0.7)
plt.yscale("log")
plt.legend()
plt.savefig("./qiskit_endcap_result/Test_Score_Distribution_endcap_qiskit.png", dpi=300, bbox_inches='tight')
