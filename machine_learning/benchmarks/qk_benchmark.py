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
"""QuantumKernel and QuantumKernelTraining benchmarks."""
import pickle
from itertools import product
from timeit import timeit
from typing import Optional, Union
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sympy import evaluate
import numpy as np
from .base_classifier_benchmark import DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION
from .qk_base_benchmark import QKernelBaseClassifierBenchmark


class QKernelBenchmarks(QKernelBaseClassifierBenchmark):
    """Quantum Kernel Classifier (q&svm or qkt) benchmarks."""

    version = 1
    timeout = 1200.0
    params = [
        [DATASET_SYNTHETIC_CLASSIFICATION, DATASET_IRIS_CLASSIFICATION],
        ["qasm_simulator", "statevector_simulator"],
        ["QuantumKernel", "QuantumKernelTraining"],
    ]
    param_names = ["dataset", "backend", "technique"]

    def __init__(self) -> None:
        super().__init__()
        self.train_features: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_features: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.model: Optional[Union[QuantumKernel, QuantumKernelTrainer]] = None

    def setup(self, dataset: str, technique: str, quantum_instance_name: str) -> None:
        """Set up the benchmark."""
        self.train_features = self.datasets[dataset]["train_features"]
        self.train_labels = self.datasets[dataset]["train_labels"]
        self.test_features = self.datasets[dataset]["test_features"]
        self.test_labels = self.datasets[dataset]["test_labels"]
        n_qubits = self.train_features.shape[1]
        if technique == "QuantumKernel":
            self.model = QSVC(
                self._construct_quantumkernel_classical_classifier(
                    quantum_instance_name=quantum_instance_name, num_qubits=n_qubits
                )
            )
        elif technique == "QuantumKernelTraining":
            self.model = QSVC(
                self._construct_quantumkerneltrainer(
                    quantum_instance_name=quantum_instance_name, num_qubits=n_qubits
                )
            )
        else:
            raise ValueError(f"Unsupported technique: {technique}")
        file_name = f"qk_{technique}_{dataset}_{quantum_instance_name}.pickle"
        with open(file_name, "rb") as file:
            self.result = pickle.load(file)

    def setup_cache(self) -> None:
        """Cache qk&svm or qkt fitted model."""
        for dataset, backend, technique in product(*self.params):
            train_features = self.datasets[dataset]["train_features"]
            train_labels = self.datasets[dataset]["train_labels"]
            n_qubits = train_features.shape[1]
            if dataset != DATASET_SYNTHETIC_CLASSIFICATION & dataset != DATASET_IRIS_CLASSIFICATION:
                raise ValueError(f"Unsupported dataset: {dataset}")
            if technique == "QuantumKernel":
                _kernel = QSVC(
                    self._construct_quantumkernel_classical_classifier(
                        quantum_instance_name=backend, num_qubits=n_qubits
                    )
                )
                model = _kernel
            elif technique == "QuantumKernelTraining":
                model = QSVC(
                    self._construct_quantumkerneltrainer(
                        quantum_instance_name=backend,
                        optimizer=COBYLA(maxiter=200),
                        num_qubits=n_qubits,
                    )
                )
            else:
                ValueError(f"Unsupported technique: {technique}")
            result = model.fit(train_features, train_labels)
            file_name = f"qk_{technique}_{dataset}_{backend}.pickle"
            with open(file_name, "wb") as file:
                pickle.dump(result, file)

    # pylint: disable=invalid-name
    def time_score_vqc_classifier(self, _, __):
        """Time scoring VQC on data."""
        self.model.score(self.train_features, self.train_labels)

    def time_predict_vqc_classifier(self, _, __):
        """Time predicting with VQC."""
        self.model.predict(self.train_features)

    def track_accuracy_score_vqc_classifier(self, _, __):
        """Tracks the overall accuracy of the classification results."""
        return self.model.score(self.test_features, self.test_labels)

    def track_precision_score_vqc_classifier(self, _, __):
        """Tracks the precision score."""
        predicts = self.model.predict(self.test_features)
        return precision_score(y_true=self.test_labels, y_pred=predicts, average="micro")

    def track_recall_score_vqc_classifier(self, _, __):
        """Tracks the recall score for each class of the classification results."""
        predicts = self.model.predict(self.test_features)
        return recall_score(y_true=self.test_labels, y_pred=predicts, average="micro")

    def track_f1_score_vqc_classifier(self, _, __):
        """Tracks the f1 score for each class of the classification results."""
        predicts = self.model.predict(self.test_features)
        return f1_score(y_true=self.test_labels, y_pred=predicts, average="micro")


if __name__ == "__main__":
    bench = QKernelBenchmarks()
    bench.setup_cache()
    for dataset_name, backend_name, technique_name in product(*QKernelBenchmarks.params):
        try:
            bench.setup(dataset_name, technique_name, backend_name)
        except NotImplementedError:
            continue
        for method in (
            "time_score_vqc_classifier",
            "time_predict_vqc_classifier",
            "track_accuracy_score_vqc_classifier",
            "track_precision_score_vqc_classifier",
            "track_recall_score_vqc_classifier",
            "track_f1_score_vqc_classifier",
        ):
            elapsed = timeit(
                f'bench.{method}("{dataset_name}", "{backend_name}")', number=10, globals=globals()
            )
