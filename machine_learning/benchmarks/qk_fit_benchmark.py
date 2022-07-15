# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Variational Quantum Classifier benchmarks."""

from itertools import product
from timeit import timeit
from typing import Optional, Union

import numpy as np
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC


from .base_classifier_benchmark import DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION

from .qk_base_benchmark import QKernelBaseClassifierBenchmark


class QKernelFitBenchmarks(QKernelBaseClassifierBenchmark):
    """QuantumKernel and QuantumKernelTraining fit benchmarks."""

    version = 2
    timeout = 1200.0
    params = [
        [DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION],
        ["qasm_simulator", "statevector_simulator"],
        ["QuantumKernel", "QuantumKernelTraining"],
        ["cobyla", "nelder-mead", "l-bfgs-b"],
        ["cross_entropy", "squared_error"],
    ]
    param_names = ["dataset", "backend name", "technique", "optimizer", "loss function"]

    def __init__(self) -> None:
        super().__init__()
        self.optimizers = {
            "cobyla": COBYLA(maxiter=100),
            "nelder-mead": NELDER_MEAD(maxiter=50),
            "l-bfgs-b": L_BFGS_B(maxiter=20),
        }
        self.train_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.model: Optional[Union[QuantumKernel, QuantumKernelTrainer]] = None

    def setup(
        self,
        dataset: str,
        tech: str,
        quantum_instance_name: str,
        optimizer: str,
        loss_function: str,
    ) -> None:
        """Set up the benchmark."""
        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]
        # new
        n_qubits = self.train_features.shape[1]
        if tech == "QuantumKernel":
            self.model = self._construct_quantumkernel_classical_classifier(
                quantum_instance_name=quantum_instance_name,
                num_qubits=n_qubits,
            )
        elif tech == "QuantumKernelTraining":
            self.model = self._construct_quantumkerneltrainer(
                quantum_instance_name=quantum_instance_name,
                optimizer=optimizer,
                loss_function=loss_function,
                num_qubits=n_qubits,
            )
        else:
            raise ValueError(f"Unsupported technique: {tech}")

    # pylint: disable=invalid-name
    def time_fit_vqc(self, tech, _, __, ___, ____):
        """Time fitting VQC to data."""
        if tech == "QuantumKernel":
            self.model = QSVC(kernel=self.model.evaluate)
        # fit
        self.model.fit(self.train_features, self.train_labels)


if __name__ == "__main__":
    for dataset_name, backend_name, technique, optimizer_name, loss_function_name in product(
        *QKernelFitBenchmarks.params
    ):
        bench = QKernelFitBenchmarks()
        try:
            bench.setup(dataset_name, technique, backend_name, optimizer_name, loss_function_name)
        except NotImplementedError:
            continue
        for method in ["time_fit_vqc"]:
            elapsed = timeit(
                f'bench.{method}("{technique}", "{dataset_name}", "{backend_name}", '
                f'"{optimizer_name}", "{loss_function_name}")',
                number=10,
                globals=globals(),
            )
            print(f"{method}:\t{elapsed}")
