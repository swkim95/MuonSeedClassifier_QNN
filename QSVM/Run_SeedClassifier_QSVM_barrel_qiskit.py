"""
This script performs binary classification using a Quantum Support Vector Machine (QSVM)
implemented with Qiskit. Unlike traditional Neural Networks, QSVM uses quantum kernels
to find optimal decision boundaries in a quantum feature space.

Key concepts for beginners:
- QSVM: Uses quantum computing to enhance classical Support Vector Machines
- Quantum Kernel: Maps classical data to quantum states and measures similarity
- Feature Map: Circuit that encodes classical data into quantum states
- No training loop needed: QSVM trains in one step (unlike iterative neural networks)

The QSVM workflow:
1. Load and preprocess data (same as neural networks)
2. Create a quantum feature map (circuit that encodes data)
3. Create a quantum kernel (measures similarity between quantum states)
4. Train QSVM classifier (one-step process)
5. Evaluate and visualize results
"""

########################################################################################################
# Data IO part - Loading and preprocessing the muon seed classification data
########################################################################################################
import sys
import os
import pickle
import pandas as pd
import numpy as np

# Add parent directory to sys.path for importing custom modules
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom preprocessing functions from BDT_model
from BDT_model.HLTIO import preprocess

print("="*80)
print("QUANTUM SUPPORT VECTOR MACHINE (QSVM) FOR MUON SEED CLASSIFICATION")
print("="*80)
print("Loading and preprocessing data...")

# Path to the original pickle file containing muon data
pkl_path = "./DYToLL_PU200_Spring23_NThltIter2FromL1/DYToLL_PU200_Spring23_NThltIter2FromL1_Barrel.pkl"

# Load the pickle file
with open(pkl_path, "rb") as file:
    data = pickle.load(file)

# Extract the DataFrame from the loaded data
df = data[0]

# Apply data quality cuts
print("Applying data quality cuts...")
# Remove unphysical pT values (pT should be positive and reasonable)
df = df[df['gen_pt'] < 1e9]  # Remove extremely high pT values
df = df[df['gen_pt'] > 0]    # Remove zero or negative pT values

# Apply setClassLabel to compute binary class labels (0 = background, 1 = signal)
df = preprocess.setClassLabel(df)

# Compute distance features between hits and L1 tracks
df = preprocess.addDistHitL1Tk(df, addAbsDist=False)

# Apply eta cuts to focus on barrel region (|eta| < 1.2)
df = df[((df['tsos_eta'] < 1.2) & (df['tsos_eta'] > -1.2))].copy()

# Define the input features for our QSVM
# These features describe the muon seed properties and track-hit distances
required_columns = [
    "expd2hitl1tk1",     # Expected distance to hit from L1 track 1
    "expd2hitl1tk2",     # Expected distance to hit from L1 track 2  
    "expd2hitl1tk3",     # Expected distance to hit from L1 track 3
    "dR_L1TkMuSeedP",    # Delta R between L1 track and muon seed
    "dPhi_L1TkMuSeedP",  # Delta phi between L1 track and muon seed
    "tsos_qbp",          # Track state parameter: q/p (charge/momentum)
    "tsos_dydz",         # Track state parameter: dy/dz slope
    "tsos_dxdz",         # Track state parameter: dx/dz slope
    "tsos_err0",         # Track state error parameter 0
    "tsos_err2",         # Track state error parameter 2
    "tsos_err5",         # Track state error parameter 5
    "y_label"            # Binary class label (0=background, 1=signal)
]

# Check if all required columns exist in the dataset
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("Warning: The following required columns are missing:", missing)

# Create final dataset with only required columns
df_final = df[required_columns].copy()
df_final = df_final.fillna(-1.)  # Fill missing values with -1

print(f"Dataset shape: {df_final.shape}")
print("First few rows of processed data:")
print(df_final.head())

# Check class distribution
print("\nClass distribution in full dataset:")
print(df_final["y_label"].value_counts())

########################################################################################################
# Data sampling and preprocessing for QSVM
########################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n" + "="*60)
print("DATA SAMPLING AND PREPROCESSING")
print("="*60)

# For QSVM tutorial, we'll use a smaller sample of data
# QSVM can be computationally intensive, so we start with manageable size
sample_size = 100
print(f"Randomly sampling {sample_size} data points for QSVM training...")

# Randomly select indices for sampling
np.random.seed(42)  # Set seed for reproducibility
random_indices = np.random.choice(df_final.index, size=sample_size, replace=False)
df_sampled = df_final.loc[random_indices]

# Check class balance in sampled data
print("Class distribution in sampled data:")
print(df_sampled["y_label"].value_counts())

# Separate features (X) and labels (y)
X = df_sampled.drop(columns=["y_label"]).values.astype(np.float32)
y = df_sampled["y_label"].values.astype(np.int32)  # QSVM expects integer labels

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# Split data into training and test sets (80% train, 20% test)
# Stratify ensures both sets have similar class distributions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class balance in both splits
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Standardize features (important for quantum algorithms)
# This scales all features to have mean=0 and std=1
print("\nStandardizing features...")
scaler = StandardScaler()
scaler.fit(X_train)  # Compute scaling parameters from training data only

# Save scaling parameters for future use
scalefiles_dir = "scalefiles"
if not os.path.exists(scalefiles_dir):
    os.makedirs(scalefiles_dir)
    
scale_filepath = os.path.join(scalefiles_dir, "barrel_qsvm_scale.txt")
with open(scale_filepath, "w") as f_scale:
    f_scale.write("%s\n" % str(scaler.mean_.tolist()))
    f_scale.write("%s\n" % str(scaler.scale_.tolist()))
print(f"Scaling parameters saved to: {scale_filepath}")

# Apply standardization to both training and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify class distributions after splitting and scaling
print("\nFinal class distributions:")
print(f"Training set: {np.bincount(y_train)}")
print(f"Test set: {np.bincount(y_test)}")

########################################################################################################
# QSVM Model Definition using Qiskit
########################################################################################################
print("\n" + "="*60)
print("QUANTUM SVM MODEL SETUP")
print("="*60)

# Import required Qiskit and Qiskit Machine Learning modules
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

print("Setting up quantum components for QSVM...")

# STEP 1: Create Quantum Feature Map
# The feature map encodes classical data into quantum states
# ZZFeatureMap applies rotations and entangling gates based on input features
n_features = X_train_scaled.shape[1]  # Number of input features (11 in our case)
feature_map_reps = 2  # Number of repetitions in the feature map circuit

print(f"Creating ZZFeatureMap with {n_features} qubits and {feature_map_reps} repetitions...")

# ZZFeatureMap creates a quantum circuit that:
# 1. Applies H gates to create superposition
# 2. Applies RZ rotations based on input data
# 3. Applies ZZ interactions between qubits for entanglement
feature_map = ZZFeatureMap(
    feature_dimension=n_features,
    reps=feature_map_reps,
    entanglement="linear"  # Connect qubits in a linear chain
)

print("Feature map circuit created successfully!")
print(f"Circuit depth: {feature_map.depth()}")
print(f"Number of parameters: {feature_map.num_parameters}")

# STEP 2: Create Quantum Kernel
# The quantum kernel measures similarity between quantum states
print("\nSetting up quantum kernel...")

# Create a sampler for quantum state measurement
sampler = Sampler()

# ComputeUncompute fidelity calculates overlap between quantum states
# This measures how "similar" two data points are in quantum feature space
fidelity = ComputeUncompute(sampler=sampler)

# Create the quantum kernel using our feature map and fidelity measure
quantum_kernel = FidelityQuantumKernel(
    fidelity=fidelity,
    feature_map=feature_map
)

print("Quantum kernel created successfully!")

# STEP 3: Create QSVM Classifier
# QSVM uses the quantum kernel instead of classical kernels (like RBF)
print("\nCreating QSVM classifier...")

# QSVC is Qiskit's quantum support vector classifier
# It works like sklearn's SVC but uses quantum kernels
qsvm = QSVC(quantum_kernel=quantum_kernel)

print("QSVM classifier created successfully!")

########################################################################################################
# Training the QSVM (One-step process)
########################################################################################################
print("\n" + "="*60)
print("TRAINING QSVM MODEL")
print("="*60)

print("Training QSVM classifier...")
print("Note: Unlike neural networks, QSVM training is a one-step optimization process")
print("This may take several minutes depending on data size and quantum circuit complexity...")

# Train the QSVM classifier
# This process:
# 1. Computes quantum kernel matrix for all training pairs
# 2. Solves the quadratic optimization problem to find support vectors
# 3. Determines optimal decision boundary in quantum feature space
qsvm.fit(X_train_scaled, y_train)

print("QSVM training completed successfully!")

# The trained model now contains:
# - Support vectors (critical training points that define the decision boundary)
# - Dual coefficients (weights for each support vector)
# - Bias term (threshold for classification decisions)

########################################################################################################
# Model Evaluation and Performance Metrics
########################################################################################################
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, auc
)

print("Evaluating QSVM performance on test set...")

# Make predictions on test set
# predict() returns class labels (0 or 1)
y_test_pred = qsvm.predict(X_test_scaled)

# For probabilistic predictions, we can use decision_function
# This returns the distance from the decision boundary
decision_scores = qsvm.decision_function(X_test_scaled)

# Convert decision scores to probabilities using sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_test_prob = sigmoid(decision_scores)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_prob)

print(f"\n{'='*40}")
print(f"QSVM PERFORMANCE RESULTS")
print(f"{'='*40}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Background', 'Signal']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"               0       1")
print(f"Actual   0   {conf_matrix[0,0]:4d}    {conf_matrix[0,1]:4d}")
print(f"         1   {conf_matrix[1,0]:4d}    {conf_matrix[1,1]:4d}")

########################################################################################################
# Visualization and Results Saving
########################################################################################################
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

import matplotlib.pyplot as plt
import itertools

# Create results directory
results_dir = './qsvm_barrel_result'
os.makedirs(results_dir, exist_ok=True)
print(f"Saving results to: {results_dir}")

# 1. ROC Curve
print("Creating ROC curve...")
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_prob)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('QSVM ROC Curve - Muon Seed Classification', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_dir}/ROC_Curve_QSVM_barrel.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrix (Raw counts)
print("Creating confusion matrix visualization...")
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('QSVM Confusion Matrix - Raw Counts', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Background (0)', 'Signal (1)'])
plt.yticks(tick_marks, ['Background (0)', 'Signal (1)'])

# Add text annotations
thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center", fontsize=16,
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f"{results_dir}/Confusion_Matrix_QSVM_barrel.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Normalized Confusion Matrix
print("Creating normalized confusion matrix...")
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Greens)
plt.title('QSVM Normalized Confusion Matrix', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Background (0)', 'Signal (1)'])
plt.yticks(tick_marks, ['Background (0)', 'Signal (1)'])

# Add text annotations for normalized values
thresh = conf_matrix_norm.max() / 2.
for i, j in itertools.product(range(conf_matrix_norm.shape[0]), range(conf_matrix_norm.shape[1])):
    plt.text(j, i, f"{conf_matrix_norm[i, j]:.3f}",
             horizontalalignment="center", fontsize=16,
             color="white" if conf_matrix_norm[i, j] > thresh else "black")

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f"{results_dir}/Normalized_Confusion_Matrix_QSVM_barrel.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Output Score Distribution
print("Creating output score distribution...")
mask_signal = (y_test == 1)
mask_background = (y_test == 0)

plt.figure(figsize=(10, 6))
plt.hist(y_test_prob[mask_signal], bins=30, alpha=0.7, 
         label=f"Signal (1) - {np.sum(mask_signal)} samples", 
         color="blue", density=True)
plt.hist(y_test_prob[mask_background], bins=30, alpha=0.7, 
         label=f"Background (0) - {np.sum(mask_background)} samples", 
         color="red", density=True)
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, 
           label='Decision threshold (0.5)')
plt.xlabel('QSVM Output Probability', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('QSVM Output Score Distribution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_dir}/Output_Score_Distribution_QSVM_barrel.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Decision Boundary Score Distribution
print("Creating decision boundary score distribution...")
plt.figure(figsize=(10, 6))
plt.hist(decision_scores[mask_signal], bins=30, alpha=0.7, 
         label=f"Signal (1)", color="blue", density=True)
plt.hist(decision_scores[mask_background], bins=30, alpha=0.7, 
         label=f"Background (0)", color="red", density=True)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.8, 
           label='Decision boundary (score=0)')
plt.xlabel('QSVM Decision Function Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('QSVM Decision Function Score Distribution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_dir}/Decision_Scores_Distribution_QSVM_barrel.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Precision-Recall Curve
print("Creating precision-recall curve...")
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('QSVM Precision-Recall Curve', fontsize=14)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{results_dir}/Precision_Recall_Curve_QSVM_barrel.png", dpi=300, bbox_inches='tight')
plt.close()

########################################################################################################
# Save Model and Results
########################################################################################################
print("\n" + "="*60)
print("SAVING MODEL AND RESULTS")
print("="*60)

# Save the trained QSVM model using pickle
import pickle
model_filepath = f"{results_dir}/qsvm_model_barrel.pkl"
with open(model_filepath, 'wb') as f:
    pickle.dump(qsvm, f)
print(f"QSVM model saved to: {model_filepath}")

# Save evaluation results
results_summary = {
    'model_type': 'QSVM',
    'dataset_info': {
        'total_samples': len(df_sampled),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': n_features,
        'class_distribution_train': np.bincount(y_train).tolist(),
        'class_distribution_test': np.bincount(y_test).tolist()
    },
    'quantum_circuit_info': {
        'feature_map_type': 'ZZFeatureMap',
        'n_qubits': n_features,
        'feature_map_reps': feature_map_reps,
        'circuit_depth': feature_map.depth(),
        'entanglement_pattern': 'linear'
    },
    'performance_metrics': {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'precision_recall_auc': float(pr_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'confusion_matrix_normalized': conf_matrix_norm.tolist()
    },
    'predictions': {
        'y_test_true': y_test.tolist(),
        'y_test_pred': y_test_pred.tolist(),
        'y_test_prob': y_test_prob.tolist(),
        'decision_scores': decision_scores.tolist()
    }
}

results_filepath = f"{results_dir}/qsvm_results_summary.pkl"
with open(results_filepath, 'wb') as f:
    pickle.dump(results_summary, f)
print(f"Results summary saved to: {results_filepath}")

# Save a human-readable summary
summary_filepath = f"{results_dir}/qsvm_performance_summary.txt"
with open(summary_filepath, 'w') as f:
    f.write("QUANTUM SUPPORT VECTOR MACHINE (QSVM) PERFORMANCE SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write("DATASET INFORMATION:\n")
    f.write(f"- Total samples used: {len(df_sampled)}\n")
    f.write(f"- Training samples: {len(X_train)}\n")
    f.write(f"- Test samples: {len(X_test)}\n")
    f.write(f"- Number of features: {n_features}\n")
    f.write(f"- Training class distribution: {dict(zip(['Background', 'Signal'], np.bincount(y_train)))}\n")
    f.write(f"- Test class distribution: {dict(zip(['Background', 'Signal'], np.bincount(y_test)))}\n\n")
    
    f.write("QUANTUM CIRCUIT INFORMATION:\n")
    f.write(f"- Feature map: ZZFeatureMap\n")
    f.write(f"- Number of qubits: {n_features}\n")
    f.write(f"- Feature map repetitions: {feature_map_reps}\n")
    f.write(f"- Circuit depth: {feature_map.depth()}\n")
    f.write(f"- Entanglement pattern: linear\n\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"- Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"- ROC AUC Score: {roc_auc:.4f}\n")
    f.write(f"- Precision-Recall AUC: {pr_auc:.4f}\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(f"                 Predicted\n")
    f.write(f"               Background  Signal\n")
    f.write(f"Actual Background  {conf_matrix[0,0]:6d}    {conf_matrix[0,1]:6d}\n")
    f.write(f"       Signal      {conf_matrix[1,0]:6d}    {conf_matrix[1,1]:6d}\n")

print(f"Performance summary saved to: {summary_filepath}")

########################################################################################################
# Final Summary and Next Steps
########################################################################################################
print("\n" + "="*80)
print("QSVM TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n📊 FINAL RESULTS SUMMARY:")
print(f"   • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • ROC AUC Score: {roc_auc:.4f}")
print(f"   • Precision-Recall AUC: {pr_auc:.4f}")

print(f"\n📁 All results saved to: {results_dir}/")
print(f"   • Model file: qsvm_model_barrel.pkl")
print(f"   • Results summary: qsvm_results_summary.pkl") 
print(f"   • Performance summary: qsvm_performance_summary.txt")
print(f"   • Scaling parameters: {scale_filepath}")

print(f"\n🎨 Visualizations created:")
print(f"   • ROC_Curve_QSVM_barrel.png")
print(f"   • Confusion_Matrix_QSVM_barrel.png")
print(f"   • Normalized_Confusion_Matrix_QSVM_barrel.png")
print(f"   • Output_Score_Distribution_QSVM_barrel.png")
print(f"   • Decision_Scores_Distribution_QSVM_barrel.png")
print(f"   • Precision_Recall_Curve_QSVM_barrel.png")

print(f"\n🚀 NEXT STEPS FOR FURTHER EXPLORATION:")
print(f"   1. Try different feature maps (e.g., PauliFeatureMap, Custom circuits)")
print(f"   2. Experiment with different entanglement patterns ('full', 'circular')")
print(f"   3. Adjust feature map repetitions for better performance")
print(f"   4. Compare with classical SVM using RBF or polynomial kernels")
print(f"   5. Scale up to larger datasets if computational resources allow")
print(f"   6. Analyze which features contribute most to classification")

print(f"\n💡 UNDERSTANDING QSVM RESULTS:")
print(f"   • Higher accuracy indicates better overall classification performance")
print(f"   • ROC AUC close to 1.0 means excellent separation between classes")
print(f"   • Confusion matrix shows detailed breakdown of correct/incorrect predictions")
print(f"   • Decision scores indicate confidence: further from 0 = more confident")

print("\nQSVM analysis complete! 🎉")
