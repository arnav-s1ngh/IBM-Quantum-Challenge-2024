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

from qc_grader.challenges.iqc_2024 import grade_lab1_ex4
#Question-4
from qiskit.circuit.library import TwoLocal
num_qubits=3
rotation_blocks=['ry','rz']
entanglement_blocks='cz'
entanglement='full'
ansatz=TwoLocal(num_qubits=num_qubits,rotation_blocks=rotation_blocks,entanglement_blocks=entanglement_blocks,entanglement=entanglement,reps=1,insert_barriers=True)
ansatz.decompose().draw('mpl')
grade_lab1_ex4(num_qubits,rotation_blocks,entanglement_blocks,entanglement)
