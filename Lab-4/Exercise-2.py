import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
import os
os.environ['QXToken'] = "aaea3f1f6d5e039325611b0141a47f6618f363ec3fea4eb04419baae7aa12480565e0c3adc51c57cb02eb94e44af3e94dd6f5349838de27ea47401c3b2975d4e"
os.environ['QISKIT_IBM_TOKEN'] = "aaea3f1f6d5e039325611b0141a47f6618f363ec3fea4eb04419baae7aa12480565e0c3adc51c57cb02eb94e44af3e94dd6f5349838de27ea47401c3b2975d4e"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import InstructionProperties
from qiskit.visualization import plot_distribution
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import StatevectorEstimator

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    EstimatorOptions
)

# qc-grader should be 0.18.12 (or higher)
import qc_grader

qc_grader.__version__
from qc_grader.challenges.iqc_2024 import (
    grade_lab4_ex1,
    grade_lab4_ex2,
    grade_lab4_ex3,
    grade_lab4_ex4,
    grade_lab4_ex5,
    grade_lab4_ex6,
    grade_lab4_ex7
)

import pandas as pd
import numpy as np


#Question-2
from qiskit.circuit.library import RealAmplitudes
num_qubits=5
reps=1
entanglement='full'
ansatz=RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
grade_lab4_ex2(num_qubits, reps, entanglement)
