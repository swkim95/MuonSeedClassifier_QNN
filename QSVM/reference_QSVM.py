# Before introducing any example, we set up the global seed to ensure reproducibility:
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 12345


# For this example, we will use the ad hoc dataset
# We can define the dataset dimension and get our train and test subsets:

from qiskit_machine_learning.datasets import ad_hoc_data

adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)


# This dataset is two-dimensional, the two features are represented by the x and y coordinates, and it has two class labels: A and B.
# We can plot it and see what the distribution looks like. We define utility functions to plot the dataset.

import matplotlib.pyplot as plt
import numpy as np


def plot_features(ax, features, labels, class_label, marker, face, edge, label):
    # A train plot
    ax.scatter(
        # x coordinate of labels where class is class_label
        features[np.where(labels[:] == class_label), 0],
        # y coordinate of labels where class is class_label
        features[np.where(labels[:] == class_label), 1],
        marker=marker,
        facecolors=face,
        edgecolors=edge,
        label=label,
    )


def plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total):

    plt.figure(figsize=(5, 5))
    plt.ylim(0, 2 * np.pi)
    plt.xlim(0, 2 * np.pi)
    plt.imshow(
        np.asmatrix(adhoc_total).T,
        interpolation="nearest",
        origin="lower",
        cmap="RdBu",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
    )

    # A train plot
    plot_features(plt, train_features, train_labels, 0, "s", "w", "b", "A train")

    # B train plot
    plot_features(plt, train_features, train_labels, 1, "o", "w", "r", "B train")

    # A test plot
    plot_features(plt, test_features, test_labels, 0, "s", "b", "w", "A test")

    # B test plot
    plot_features(plt, test_features, test_labels, 1, "o", "r", "w", "B test")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title("Ad hoc dataset")

    plt.show()

plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total)


# The next step is to create a quantum kernel instance that will help classify this data.
# We use the FidelityQuantumKernel class, and pass two input arguments to its constructor:
# 1. feature_map: in this case, a two-qubit ZZFeatureMap.
# 2. fidelity: in this case, the ComputeUncompute fidelity subroutine that leverages the Sampler primitive.

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

# The quantum kernel can now be plugged into classical kernel methods, such as the SVC algorithm from scikit-learn.
# This algorithm allows us to define a custom kernel in two ways:
# 1. by providing the kernel as a callable function
# 2. by precomputing the kernel matrix

# Kernel as a callable function
# We define a SVC model and directly pass the evaluate function of the quantum kernel as a callable.
# Once the model is created, we train it by calling the fit method on the training dataset and evaluate the model for accuracy with score.

from sklearn.svm import SVC

adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)

adhoc_svc.fit(train_features, train_labels)

adhoc_score_callable_function = adhoc_svc.score(test_features, test_labels)

print(f"Callable kernel classification test score: {adhoc_score_callable_function}")

# Precomputed kernel matrix
# Instead of passing a function of the quantum kernel as a callable, 
# we can also precompute training and testing kernel matrices before passing them to the scikit-learn SVC algorithm.
# To extract the train and test matrices, 
# we can call evaluate on the previously defined kernel and visualize them graphically as follows:

adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features, y_vec=train_features)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(
    np.asmatrix(adhoc_matrix_train), interpolation="nearest", origin="upper", cmap="Blues"
)
axs[0].set_title("Ad hoc training kernel matrix")

axs[1].imshow(np.asmatrix(adhoc_matrix_test), interpolation="nearest", origin="upper", cmap="Reds")
axs[1].set_title("Ad hoc testing kernel matrix")

plt.show()

# To use these matrices, we set the kernel parameter of a new SVC instance to "precomputed".
# We train the classifier by calling fit with the training matrix and training dataset.
# Once the model is trained, we evaluate it using the test matrix on the test dataset.

adhoc_svc = SVC(kernel="precomputed")

adhoc_svc.fit(adhoc_matrix_train, train_labels)

adhoc_score_precomputed_kernel = adhoc_svc.score(adhoc_matrix_test, test_labels)

print(f"Precomputed kernel classification test score: {adhoc_score_precomputed_kernel}")

# Classification with QSVC
# QSVC is an alternative training algorithm provided by qiskit-machine-learning for convenience.
# It is an extension of SVC that takes in a quantum kernel instead of the kernel.evaluate method shown before.

from qiskit_machine_learning.algorithms import QSVC

qsvc = QSVC(quantum_kernel=adhoc_kernel)

qsvc.fit(train_features, train_labels)

qsvc_score = qsvc.score(test_features, test_labels)

print(f"QSVC classification test score: {qsvc_score}")

# Evaluation of models used for classification

print(f"Classification Model                    | Accuracy Score")
print(f"---------------------------------------------------------")
print(f"SVC using kernel as a callable function | {adhoc_score_callable_function:10.2f}")
print(f"SVC using precomputed kernel matrix     | {adhoc_score_precomputed_kernel:10.2f}")
print(f"QSVC                                    | {qsvc_score:10.2f}")