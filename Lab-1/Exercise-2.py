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

from qc_grader.challenges.iqc_2024 import grade_lab1_ex2

#Question-2
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.x(1)
qc.z(1)
qc.measure_all()
qc.draw('mpl')
statevector_sampler=StatevectorSampler()
job_sampler=statevector_sampler.run([qc])
result_sampler=job_sampler.result()
counts_sampler=result_sampler[0].data.meas.get_counts()
print(counts_sampler)
grade_lab1_ex2(job_sampler)
