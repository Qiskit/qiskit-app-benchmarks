# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for QuantumKernel and QuantumKernelTraining based classifier benchmarks."""
from abc import ABC
from typing import Optional

from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

import numpy as np

from .base_classifier_benchmark import BaseClassifierBenchmark


class QKernelBaseClassifierBenchmark(BaseClassifierBenchmark, ABC):
    """Base class for quantum kernel benchmarks."""

    def __init__(self) -> None:
        reshaper = FunctionTransformer(lambda x: x.reshape(-1, 1))
        encoder = OneHotEncoder(sparse=False)
        super().__init__(
            synthetic_label_encoder=Pipeline([("reshape", reshaper), ("one hot", encoder)]),
            iris_num_classes=2,
            iris_label_encoder=Pipeline([("reshape", reshaper), ("one hot", encoder)]),
        )

    def _construct_quantumkernel_classical_classifier(
        self,
        quantum_instance_name: str,
        method="quantumclassical",
        num_qubits=1,
    ) -> QuantumKernel:
        """This method just create the matrix from the quantum kernel"""
        kernelmatrix = self._construct_quantumkernel(num_qubits, quantum_instance_name, method)
        return kernelmatrix

    def _construct_quantumkerneltrainer(
        self,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        loss_function: str = None,
        method="quantum",
        num_qubits=1,
    ) -> QuantumKernelTrainer:
        """This method returns the quantumkerneltrainer"""
        kernel = self._construct_quantumkernel(num_qubits, quantum_instance_name, method)
        qkt = QuantumKernelTrainer(
            quantum_kernel=kernel,
            loss=loss_function,
            optimizer=optimizer,
            initial_point=[np.pi / 2],
        )
        return qkt

    def _construct_quantumkernel(
        self,
        num_inputs: int,
        quantum_instance_name: str,
        method: str,
    ) -> QuantumKernel:
        """Construct a QuantumKernel"""
        if method == "quantumclassical":
            feature_map = ZZFeatureMap(num_inputs, reps=2, entanglement="linear")
            qkernel = QuantumKernel(
                feature_map=feature_map, quantum_instance=self.backends[quantum_instance_name]
            )
            return qkernel
        elif method == "quantum":
            user_params = ParameterVector("θ", 1)
            fm0 = QuantumCircuit(num_inputs)
            for i in range(num_inputs):
                fm0.ry(user_params[0], i)
            fm1 = ZZFeatureMap(num_inputs, reps=2, entanglement="linear")
            feature_map = fm0.compose(fm1)
            qkernel = QuantumKernel(
                feature_map=feature_map,
                user_parameters=user_params,
                quantum_instance=self.backends[quantum_instance_name],
            )
            return qkernel
        else:
            return ValueError(f"Unsupported method: {method}")
