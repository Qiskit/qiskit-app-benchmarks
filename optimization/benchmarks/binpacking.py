# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""bin packing benchmarks"""

import numpy as np

from qiskit import Aer
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals, QuantumInstance

from qiskit_optimization.algorithms import MinimumEigenOptimizer, GroverOptimizer
from qiskit_optimization.applications import BinPacking
from qiskit_optimization.converters import QuadraticProgramToQubo

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init


class BinPackingBenchmarks:
    """Bin Packing benchmarks"""

    version = 1
    params = ([2, 3], [2, 4, 6])
    param_names = ["number of items", "max_weight"]

    def setup(self, num_items, max_weight):
        """setup"""
        seed = 1
        algorithm_globals.random_seed = seed
        qasm_sim = Aer.get_backend("aer_simulator")
        self._qins = QuantumInstance(
            backend=qasm_sim, shots=1, seed_simulator=seed, seed_transpiler=seed
        )
        np.random.seed(seed)
        values = [np.random.randint(1, max_weight) for i in range(num_items)]
        max_number_of_bins = len(values) - 1
        self._bin_packing = BinPacking(values, max_weight, max_number_of_bins)
        self._qp = self._bin_packing.to_quadratic_program()

    @staticmethod
    def _generate_qubo(bin_packing: BinPacking):
        q_p = bin_packing.to_quadratic_program()
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(q_p)
        return qubo

    def time_generate_qubo(self, _, __):
        """generate time qubo"""
        self._generate_qubo(self._bin_packing)

    def time_qaoa(self, _, __):
        """time qaoa"""
        meo = MinimumEigenOptimizer(
            min_eigen_solver=QAOA(optimizer=COBYLA(maxiter=1), quantum_instance=self._qins)
        )
        meo.solve(self._qp)

    def time_vqe(self, _, __):
        """time vqe"""
        meo = MinimumEigenOptimizer(
            min_eigen_solver=VQE(
                optimizer=COBYLA(maxiter=1), ansatz=EfficientSU2(), quantum_instance=self._qins
            )
        )
        meo.solve(self._qp)

    def time_grover(self, _, __):
        """time grover"""
        meo = GroverOptimizer(
            num_value_qubits=self._qp.get_num_vars(),
            num_iterations=1,
            quantum_instance=self._qins,
        )
        meo.solve(self._qp)
