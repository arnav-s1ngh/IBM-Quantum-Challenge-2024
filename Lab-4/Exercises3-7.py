import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
import os

from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

os.environ['QXToken'] = "Paste_Your_Token_Here"
os.environ['QISKIT_IBM_TOKEN'] = "Paste_Your_Token_Here"

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

import pandas as pd
import numpy as np

# Define num_qubits
num_qubits = 5

# Load the dataset
birds_dataset = pd.read_csv('birds_dataset.csv')
# Convert coefficients to complex numbers
for i in range(2**num_qubits):
    key = 'c%.0f' % i
    birds_dataset[key] = birds_dataset[key].astype(np.complex128)

# Print the dataset to verify (optional)
print(birds_dataset)

# Retrieve the coefficients of each quantum state
list_coefficients = birds_dataset[['c%.0f' % i for i in range(2**num_qubits)]].values.tolist()
list_labels = []
for name in birds_dataset['names']:
    if name=="Eagle" or name=="Heron" or name=="Osprey" or name=="Condor" or name=="Falcon" or name=="Hummingbird" or name=="Canary" or name=="Egret":
        list_labels.append(1)
    else:
        list_labels.append(0)

from qiskit.circuit.library import RealAmplitudes
num_qubits=5
reps=1
entanglement='full'
ansatz=RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)

# Define the observable
obs = SparsePauliOp("ZZZZZ")
# Define the estimator and pass manager
estimator = StatevectorEstimator()  # To train we use StatevectorEstimator to get the exact simulation
pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=3, seed_transpiler=0)


# Define the cost function
def cost_func(params, list_coefficients, list_labels, ansatz, obs, estimator, pm, callback_dict):
    """Return cost function for optimization

    Parameters:
        params (ndarray): Array of ansatz parameters
        list_coefficients (list): List of arrays of complex coefficients
        list_labels (list): List of labels
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        obs (SparsePauliOp): Observable
        estimator (EstimatorV2): Statevector estimator primitive instance
        pm (PassManager): Pass manager
        callback_dict (dict): Dictionary to store callback information

    Returns:
        float: Cost function estimate
    """

    cost = 0
    for amplitudes, label in zip(list_coefficients, list_labels):
        qc = QuantumCircuit(num_qubits)
        # Amplitude embedding
        qc.initialize(amplitudes)
        # Compose initial state + ansatz
        classifier = qc.compose(ansatz)
        # Transpile classifier
        transpiled_classifier = pm.run(classifier)
        # Transpile observable
        transpiled_obs = obs.apply_layout(layout=transpiled_classifier.layout)
        # Run estimator
        pub = (transpiled_classifier, transpiled_obs, params)
        job = estimator.run([pub])
        # Get result
        result = job.result()[0].data.evs
        # Compute cost function (cumulative)
        cost += np.abs(result - label)

    callback_dict["iters"] += 1
    callback_dict["prev_vector"] = params
    callback_dict["cost_history"].append(cost)

    # Print the iterations to screen on a single line
    print(
        "Iters. done: {} [Current cost: {}]".format(callback_dict["iters"], cost),
        end="\r",
        flush=True,
    )

    return cost


# Intialize the lists to store the results from different runs
cost_history_list = []
res_list = []

# Retrieve the initial parameters
params_0_list = np.load("params_0_list.npy")

for it, params_0 in enumerate(params_0_list):
    print('Iteration number: ', it)

    # Initialize a callback dictionary
    callback_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    # Minimize the cost function using scipy
    res = minimize(
        cost_func,
        params_0,
        args=(list_coefficients, list_labels, ansatz, obs, estimator, pm, callback_dict),
        method="cobyla",  # Classical optimizer
        options={'maxiter': 200})  # Maximum number of iterations

    # Print the results after convergence
    print(res)

    # Save the results from different runs
    res_list.append(res)
    cost_history_list.append(callback_dict["cost_history"])

def test_VQC(list_coefficients, list_labels, ansatz, obs, opt_params, estimator, pm):
    results_test = []
    for amplitudes in list_coefficients:
        qc = QuantumCircuit(num_qubits)
        qc.initialize(amplitudes)
        classifier = qc.compose(ansatz)
        transpiled_classifier = pm.run(classifier)
        transpiled_obs = obs.apply_layout(layout=transpiled_classifier.layout)
        pub = (transpiled_classifier, transpiled_obs, opt_params)
        job = estimator.run([pub])
        result = job.result()[0].data.evs
        results_test.append(result)
    return results_test

def compute_performance(result_list, list_labels):
    predictions = [1 if r >= 0.5 else 0 for r in result_list]
    accuracy = sum(pred == label for pred, label in zip(predictions, list_labels)) / len(list_labels)
    return accuracy

best_result_index = np.argmax([compute_performance(test_VQC(list_coefficients, list_labels, ansatz, obs, res.x, estimator, pm), list_labels) for res in res_list])

fake_backend = GenericBackendV2(
    num_qubits=5,
    basis_gates=["id", "rz", "sx", "x", "cx"]
  )


def update_error_rate(backend, error_rates):
    """Updates the error rates of the backend

    Parameters:
        backend (BackendV2): Backend to update
        error_rates (dict): Dictionary of error rates

    Returns:
        None
    """

    default_duration = 1e-8
    if "default_duration" in error_rates:
        default_duration = error_rates["default_duration"]

    # Update the 1-qubit gate properties
    for i in range(backend.num_qubits):
        qarg = (i,)
        if "rz_error" in error_rates:
            backend.target.update_instruction_properties('rz', qarg,
                                                         InstructionProperties(error=error_rates["rz_error"],
                                                                               duration=default_duration))
        if "x_error" in error_rates:
            backend.target.update_instruction_properties('x', qarg, InstructionProperties(error=error_rates["x_error"],
                                                                                          duration=default_duration))
        if "sx_error" in error_rates:
            backend.target.update_instruction_properties('sx', qarg,
                                                         InstructionProperties(error=error_rates["sx_error"],
                                                                               duration=default_duration))
        if "measure_error" in error_rates:
            backend.target.update_instruction_properties('measure', qarg,
                                                         InstructionProperties(error=error_rates["measure_error"],
                                                                               duration=default_duration))

            # Update the 2-qubit gate properties (CX gate) for all edges in the chosen coupling map
    if "cx_error" in error_rates:
        for edge in backend.coupling_map:
            backend.target.update_instruction_properties('cx', tuple(edge),
                                                         InstructionProperties(error=error_rates["cx_error"],
                                                                               duration=default_duration))

error_rates = {
  "default_duration": 1e-8,
  "rz_error": 1e-8,
  "x_error": 1e-8,
  "sx_error": 1e-8,
  "measure_error": 1e-8,
  "cx_error": 1e-8
}

update_error_rate(fake_backend, error_rates)

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler import PassManager

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_title('Cost on test data')
ax.set_ylabel('Cost')
ax.set_xlabel('State index')
ax.plot(list_labels, 'k-', linewidth=3, alpha=0.6, label='Labels')

error_rate_list = [1e-1, 1e-2, 1e-3, 1e-4]

fake_backend = GenericBackendV2(
    num_qubits=5,
    basis_gates=["id", "rz", "sx", "x", "cx"]
)

for error_rate_value in error_rate_list:
    error_rates['rz_error'] = error_rate_value
    error_rates['cx_error'] = error_rate_value
    update_error_rate(fake_backend, error_rates)
    estimator=Estimator(backend=fake_backend)
    pm=generate_preset_pass_manager(backend=fake_backend,optimization_level=3,seed_transpiler=0)
    opt_params=res_list[best_result_index].x
    results_test=test_VQC(list_coefficients,list_labels,ansatz,obs,opt_params,estimator,pm)

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def amplitude_embedding(num_qubits, bird_index):
    """Create amplitude embedding circuit

    Parameters:
        num_qubits (int): Number of qubits for the ansatz
        bird_index (int): Data index of the bird

    Returns:
        qc (QuantumCircuit): Quantum circuit with amplitude embedding of the bird
    """

    def generate_GHZ(qc):
        qc.h(0)
        for i, j in zip(range(num_qubits-1), range(1,num_qubits)):
            qc.cx(i, j)


    qc = QuantumCircuit(num_qubits)
    if bird_index < 5:
        generate_GHZ(qc)
    bit_str=format(bird_index,'0{0}b'.format(num_qubits))
    for i,bit in enumerate(bit_str):
        if bit=='1':
            qc.x(num_qubits-i-1)
    return qc


index_bird = 0 # You can check different birds by changing the index

# Build the amplitude embedding
qc = amplitude_embedding(num_qubits, index_bird)
qc.measure_all()

# Define the backend and the pass manager
aer_sim = AerSimulator()
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=3)
isa_circuit = pm.run(qc)

# Define the sampler with the number of shots
sampler = Sampler(backend=aer_sim)
result = sampler.run([isa_circuit]).result()
samp_dist = result[0].data.meas.get_counts()
plot_distribution(samp_dist, figsize=(15, 5))

index_bird = 0 #You can check different birds by changing the index
qc = amplitude_embedding(num_qubits, index_bird)
pm = generate_preset_pass_manager(optimization_level=3, backend=fake_backend)
transpiled_qc = pm.run(qc)

print('Depth of two-qubit gates: ', transpiled_qc.depth(lambda x: len(x.qubits) == 2))
transpiled_qc.draw(output="mpl", fold=False, idle_wires=False)
# Submit your answer using following code

old_ansatz = RealAmplitudes(num_qubits, reps=1, entanglement='full', insert_barriers=True)
pm = generate_preset_pass_manager(optimization_level=3, backend=fake_backend)
transpiled_ansatz = pm.run(old_ansatz)

print('Depth of two-qubit gates: ', transpiled_ansatz.depth(lambda x: len(x.qubits) == 2))

ansatz = RealAmplitudes(num_qubits, reps=1, entanglement='pairwise', insert_barriers=True)
pm = generate_preset_pass_manager(optimization_level=3, backend=fake_backend)
transpiled_ansatz = pm.run(ansatz)

print('Depth of two-qubit gates: ', transpiled_ansatz.depth(lambda x: len(x.qubits) == 2))


old_mapping = QuantumCircuit(num_qubits)
old_mapping.initialize(list_coefficients[index_bird])
old_classifier = old_mapping.compose(old_ansatz)

new_mapping = amplitude_embedding(num_qubits, index_bird)
new_classifier = new_mapping.compose(ansatz)

pm = generate_preset_pass_manager(optimization_level=3, backend=fake_backend)
old_transpiled_classifier = pm.run(old_classifier)
new_transpiled_classifier = pm.run(new_classifier)

print('Old depth of two-qubit gates: ', old_transpiled_classifier.depth(lambda x: len(x.qubits) == 2))
print('Current depth of two-qubit gates: ', new_transpiled_classifier.depth(lambda x: len(x.qubits) == 2))

def test_shallow_VQC(list_labels, ansatz, obs, opt_params, estimator, pm):
    results_test = []
    cost = 0
    for index_bird in range(len(list_labels)):
        qc = amplitude_embedding(5, index_bird)
        classifier = qc.compose(ansatz)
        transpiled_classifier = pm.run(classifier)
        transpiled_obs = obs.apply_layout(layout=transpiled_classifier.layout)
        pub = (transpiled_classifier, transpiled_obs, opt_params)
        job = estimator.run([pub])
        result = job.result()[0].data.evs
        print(result)
        results_test.append(abs(result))
    return results_test

estimator = Estimator(backend=fake_backend)
estimator.options.default_shots = 5000
pm = generate_preset_pass_manager(optimization_level=3, backend=fake_backend)

opt_params = np.load('opt_params_shallow_VQC.npy') # Load optimal parameters
results_test = test_shallow_VQC(list_labels, ansatz, obs, opt_params, estimator, pm)

print(f"Performance: {compute_performance(results_test, list_labels)}")

service = QiskitRuntimeService(
   channel='ibm_quantum',
   instance= 'ibm-q/open/main',
   token="Paste_Your_Token_Here"
)
backend = service.backend("ibm_sherbrooke")



def amplitude_embedding(num_qubits, bird_index):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if (bird_index >> i) & 1:
            qc.x(i)
    return qc

def test_shallow_VQC_QPU(list_labels, ansatz, obs, opt_params, options, backend):
    """Return the performance of the classifier

    Parameters:
        list_labels (list): List of labels
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        obs (SparsePauliOp): Observable
        opt_params (ndarray): Array of optimized parameters
        options (EstimatorOptions): Estimator options
        backend (service.backend): Backend to run the job

    Returns:
        job_id (str): Job ID
    """

    estimator = Estimator(backend=backend, options=options)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    pubs = []
    for bird_index, label in enumerate(list_labels):
        qc = amplitude_embedding(ansatz.num_qubits, bird_index)
        classifier = qc.compose(ansatz)
        transpiled_classifier = pm.run(classifier)
        transpiled_classifier = transpiled_classifier.bind_parameters(opt_params)
        transpiled_obs = obs.apply_layout(layout=transpiled_classifier.layout)

        pub = (transpiled_classifier, transpiled_obs, opt_params)
        pubs.append(pub)

    job = estimator.run(pubs)
    job_id = job.job_id()
    print(f"Job ID: {job_id}")
    print(f"Status: {job.status()}")

    return job_id

options_0 = EstimatorOptions(
    default_shots = 5000,
    optimization_level=0,
    resilience_level = 0,
    twirling = {'enable_measure':False},
    dynamical_decoupling ={
        'enable': False,
        'sequence_type':"XpXm"
            }

) 

options_1 = EstimatorOptions(
    default_shots = 5000,
    optimization_level=0,
    resilience_level = 1,
    twirling = {'enable_measure':True},
    dynamical_decoupling ={
        'enable': True,
        'sequence_type':"XpXm"
            }
)
grade_lab4_ex7(options_0, options_1)
