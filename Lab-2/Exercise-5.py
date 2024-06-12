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

from qc_grader.challenges.iqc_2024 import (
    grade_lab2_ex1,
    grade_lab2_ex2,
    grade_lab2_ex3,
    grade_lab2_ex4,
    grade_lab2_ex5
)
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import XGate, YGate
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeOsaka
from qiskit.transpiler import InstructionProperties, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.scheduling import ASAPScheduleAnalysis,PadDynamicalDecoupling
from qiskit.visualization.timeline import draw, IQXStandard
from qiskit.transpiler import StagedPassManager
from qiskit.visualization import plot_circuit_layout
import matplotlib.pyplot as plt
import numpy as np

#Question-5

def scoring(qc, backend):
    from util import transpile_scoring
    transpiled_qc=transpile(qc,backend=backend)
    layout=transpiled_qc._layout.final_layout
    fidelity=transpile_scoring(transpiled_qc,layout,backend)
    score=1-fidelity
    return score


### Create a random circuit

## DO NOT CHANGE THE SEED NUMBER
seed = 10000

## Create circuit

num_qubits = 6
depth = 4
qc = random_circuit(num_qubits,depth,measure=False, seed=seed)

qc.draw('mpl')

backend = FakeTorino()
circuit_depths = {
    'opt_lv_0': None,
    'opt_lv_1': None,
    'opt_lv_2': None,
    'opt_lv_3': None,
}
gate_counts = {
    'opt_lv_0': None,
    'opt_lv_1': None,
    'opt_lv_2': None,
    'opt_lv_3': None,
}

scores = {
    'opt_lv_0': None,
    'opt_lv_1': None,
    'opt_lv_2': None,
    'opt_lv_3': None,
}

pm_lv0 = generate_preset_pass_manager(backend=backend, optimization_level=0, seed_transpiler=seed)
tr_lv0 = pm_lv0.run(qc)

#level 0
circuit_depths['opt_lv_0'] = tr_lv0.depth()
gate_counts['opt_lv_0'] = tr_lv0.num_nonlocal_gates()
scores['opt_lv_0'] = scoring(tr_lv0, backend)

#level 1
pm_lv1 = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed)
tr_lv1 = pm_lv1.run(qc)
circuit_depths['opt_lv_1'] = tr_lv1.depth()
gate_counts['opt_lv_1'] = tr_lv1.num_nonlocal_gates()
scores['opt_lv_1'] = scoring(tr_lv1, backend)

#level 2
pm_lv2 = generate_preset_pass_manager(backend=backend, optimization_level=2, seed_transpiler=seed)
tr_lv2 = pm_lv2.run(qc)
circuit_depths['opt_lv_2'] = tr_lv2.depth()
gate_counts['opt_lv_2'] = tr_lv2.num_nonlocal_gates()
scores['opt_lv_2'] = scoring(tr_lv2, backend)

#level 3
pm_lv3 = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed)
tr_lv3 = pm_lv3.run(qc)
circuit_depths['opt_lv_3'] = tr_lv3.depth()
gate_counts['opt_lv_3'] = tr_lv3.num_nonlocal_gates()
scores['opt_lv_3'] = scoring(tr_lv3, backend)
ans = [pm_lv0, pm_lv1, pm_lv2, pm_lv3]

pm_ex4 = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3,
    layout_method="sabre",
    routing_method="sabre",
    translation_method="synthesis",
)

from qiskit.transpiler.passes import ASAPScheduleAnalysis, PadDynamicalDecoupling

X = XGate()
Y = YGate()

dd_sequence = [X, Y, X, Y]

backend=FakeTorino()
target = backend.target

y_gate_properties = {}
for qubit in range(target.num_qubits):
    y_gate_properties.update(
        {
            (qubit,): InstructionProperties(
                duration=target["x"][(qubit,)].duration,
                error=target["x"][(qubit,)].error,
            )
        }
    )

target.add_instruction(YGate(), y_gate_properties)


dd_pm = PassManager(
    [
        ASAPScheduleAnalysis(target=target),
        PadDynamicalDecoupling(target=target,dd_sequence=dd_sequence)
    ]
)

from qiskit.transpiler import StagedPassManager

staged_pm_dd = StagedPassManager(
    stages=["scheduling"],
    scheduling=dd_pm
)
backend_timing = backend.target.timing_constraints()
timing_constraints = TimingConstraints(
    granularity=backend_timing.granularity,
    min_length=backend_timing.min_length,
    pulse_alignment=backend_timing.pulse_alignment,
    acquire_alignment=backend_timing.acquire_alignment )
pm_asap = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    timing_constraints=timing_constraints,
    scheduling_method="asap",
    seed_transpiler=seed,
)
qc_tr = pm_asap.run(qc)
qc_tr_with_dd = staged_pm_dd.run(qc_tr)
grade_lab2_ex5(staged_pm_dd)
