import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
import os
os.environ['QXToken'] = "Paste_Your_Token_Here"
import numpy as np
from typing import List, Callable
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from qiskit.primitives import StatevectorSampler, PrimitiveJob
from qiskit.circuit.library import TwoLocal
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_aer import AerSimulator, Aer

from qc_grader.challenges.iqc_2024 import grade_lab1_ex6



#Question-6
from qiskit.circuit.library import TwoLocal
num_qubits=3
rotation_blocks=['ry','rz']
entanglement_blocks='cz'
entanglement='full'
ansatz=TwoLocal(num_qubits=num_qubits,rotation_blocks=rotation_blocks,entanglement_blocks=entanglement_blocks,entanglement=entanglement,reps=1,insert_barriers=True)
ansatz.decompose().draw('mpl')

from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.circuit.random import random_circuit
from qiskit.circuit import QuantumRegister
num_qubits=3
rotation_blocks=['ry','rz']
entanglement_blocks='cz'
entanglement='full'
ansatz=TwoLocal(num_qubits=num_qubits, rotation_blocks=rotation_blocks, entanglement_blocks=entanglement_blocks, entanglement=entanglement, reps=1, insert_barriers=True)
from qiskit.transpiler import preset_passmanagers
backend_answer = FakeSherbrooke()
optimization_level_answer=3
isa_circuit=transpile(ansatz, backend=backend_answer,optimization_level=optimization_level_answer)
isa_circuit.draw('mpl')
pm = preset_passmanagers.generate_preset_pass_manager(backend=backend_answer,optimization_level=optimization_level_answer)
isa_circuit=pm.run(ansatz)

#the real shit begins here

pauli_op = SparsePauliOp(['ZII', 'IZI', 'IIZ'])
hamiltonian_isa = pauli_op.apply_layout(layout=isa_circuit.layout)


def cost_func(params, ansatz, hamiltonian, estimator, callback_dict):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    callback_dict["iters"] += 1
    callback_dict["prev_vector"] = params
    callback_dict["cost_history"].append(energy)

    print(energy) #0.6875
    return energy, result


grade_lab1_ex6(cost_func)
