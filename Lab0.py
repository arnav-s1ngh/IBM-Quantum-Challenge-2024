import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
qc = QuantumCircuit(1)
qc.x(0)
qc.draw("mpl")
plt.show()
import os
os.environ['QXToken'] = "Paste_Your_Token_Here"
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qc_grader.challenges.iqc_2024 import grade_lab0_ex1
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
import matplotlib.pyplot as plt
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.draw("mpl")
ZZ = SparsePauliOp('ZZ')
ZI = SparsePauliOp('ZI')
IX = SparsePauliOp('IX')
IZ = SparsePauliOp('IZ')
XX = SparsePauliOp('XX')
XI = SparsePauliOp('XI')
observables = [IZ, IX, ZI, XI, ZZ, XX]
grade_lab0_ex1(observables)
