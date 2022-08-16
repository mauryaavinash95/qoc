import os

# Every computation that qoc performs has to be differentiable.
# qoc uses autograd [2] to do automatic differentiation [3].
# All operations that you perform on your operands should use autograd's
# numpy wrapper. autograd.numpy wraps the entire numpy namespace, but not all functions
# have derivatives implemented for them. You may view which functionality is supported
# on autograd's github [2].
import jax
import jaxlib

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import time
import qutip as qt
import numpy as onp
import sys, getopt
import jax.profiler

data_path = '../'
# All of the core functionality that qoc offers can be imported from the top level.
# Core functionality not shown here includes `evolve_schroedinger_discrete`
# and `evolve_lindblad_discrete`.
from qoc import (grape_schroedinger_discrete_unitary, grape_lindblad_discrete)

# qoc.standard is a module that has optimization cost functions, optimizers,
# convenience functions, and other goodies. All of the operations
# (e.g. `conjugate_transpose`) that you import
# from qoc.standard use autograd.numpy, so they are OK to use in your operations.
from qoc.standard import (Adam,
    conjugate_transpose, generate_save_file_path,
    get_creation_operator, get_annihilation_operator,
    krons, LBFGSB, matmuls, plot_controls, plot_density_population,
    plot_state_population, plot_error, TargetUnitaryInfidelity, 
    matrix_to_column_vector_list,
    corner_terms, plaquette_terms, single_qubit_terms, single_qubit_terms_custom,
    corner_terms_2, plaquette_terms_2, single_qubit_terms_2, single_qubit_terms_custom_2, 
    corner_terms_3, plaquette_terms_3, single_qubit_terms_3, single_qubit_terms_custom_3)

import jax
import matplotlib


# Define the size of the system.
argv = sys.argv[1:]
QUBIT_COUNT = 6    #4
CONTROL_EVAL_COUNT = int(3000)
USE_CUSTOM_INNER = -1
CHECKPOINT_INTERVAL = 10
try:
    opts, args = getopt.getopt(argv,"hq:s:i:c:",["qubit="])
except getopt.GetoptError:
    print('test.py -q <qubit count> -s <step_count> -i <custom_inner=0/1/2>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py -q <qubit count> -s <steps>')
        sys.exit()
    elif opt in ("-q", "--qubit"):
        QUBIT_COUNT = int(arg)
    elif opt in ("-s", "--steps"):
        CONTROL_EVAL_COUNT = int(arg)
    elif opt in ("-i", "--custominner"):
        USE_CUSTOM_INNER = int(arg)
    elif opt in ("-c", "--checkpointinverval"):
        CHECKPOINT_INTERVAL = int(arg)

if QUBIT_COUNT<1:
    print("Must specify a QUBIT_COUNT > 0")
    sys.exit()

HILBERT_SIZE = 2**QUBIT_COUNT

LENGTH = 2
WIDTH = 3
# Additionally, we need to specify information to qoc about...
# how long our system will evolve for
EVOLUTION_TIME = 500 #15 #ns
# how many controls we have
CONTROL_COUNT = 6    #4
# what domain our controls are in
COMPLEX_CONTROLS = False
# where our controls are positioned in time
#CONTROL_EVAL_COUNT = int(1e3)
# and where our system is evaluated in time
#SYSTEM_EVAL_COUNT = iunitary_countnt(1e2)
SYSTEM_EVAL_COUNT = CONTROL_EVAL_COUNT
# Note that `CONTROL_COUNT` is the length of the `controls` array that is passed
# to our `hamiltonian` function.
# `CONTROL_EVAL_COUNT` is used to determine how many points in time the `controls` are
# evaluated. It is likely this value should be consistent with a physical apparatus,
# such as the sampling rate of an AWG. The points in time where controls are evaluated
# is given by control_eval_times = np.linspace(0, evolution_time, control_eval_count).
# Note that qoc uses an interpolation method to interpolate the control parameters
# between these time points. You can change this behavior using the
# `interpolation_policy` argument.
# SYSTEM_EVAL_COUNT is used to determine the update step of the evolution.
# Similarly, system_eval_times = np.linspace(0, evolution_time, system_eval_count).
# Two important things happen at each system_eval step.
# First, cost functions that are computed multiple times throughout
# the evolution (as opposed to those only computed at the end of evolution)
# are evaluated at system_eval steps. You can change this behavior using the
# `cost_eval_step` argument. Second, qoc uses an exponential series method
# to integrate the schroedinger equation. `system_eval_times` specifies the
# time steps used in this integration. Therefore, increasing the `system_eval_count`
# will likely increase the accuracy of the evolution. The accuracy of the evolution
# can also be increased with the `magnus_policy` argument. Increasing the accuracy
# using both of these methods will increase the computational cost.
# Note that qoc does not use an exponential series method to integrate the lindblad
# equation. Therefore, increasing the `system_eval_count` for lindblad methods
# will not increase the accuracy of their evolution.

# Next, we define the system hamiltonian.
# Control parameters are values that you will use to vary time-dependent control fields
# that act on your system. Note that qoc supports both complex and real control parameters.
#matvals=qt.rand_herm(HILBERT_SIZE).data

DEVICE_HAMILTONIAN = ( 0.5 * single_qubit_terms_2(LENGTH, WIDTH, sigma = np.array(qt.operators.sigmaz().data.toarray()) )  
                   + 0.25 * corner_terms_2(LENGTH, WIDTH, sigma = np.array(qt.operators.sigmax().data.toarray()) )
                   + 0.25 * corner_terms_2(LENGTH, WIDTH, sigma = np.array(qt.operators.sigmay().data.toarray()) )  )

CONTROL = 0.5 * single_qubit_terms_custom_2(LENGTH, WIDTH, CONTROL_COUNT, [0, 1, 2, 3, 4, 5], sigma = np.array(qt.operators.sigmax().data.toarray()) )

def hamiltonian(controls, time):
    return (DEVICE_HAMILTONIAN + np.multiply( controls[:, np.newaxis, np.newaxis], CONTROL ).sum(0) )

# Now, we are ready to give qoc a problem.
# Let's try to put a photon in the cavity.
# That is, we desire the fock state transition |0, g> -> |1, g>.
# Notice that when we specify states (or probability density matrices!)
# to qoc, we always give qoc an array of states that we would like it to track,
# even if we only give qoc a single state. The `,` in np.stack((INITIAL_STATE_0`,`))
# makes a difference.

initial0 = jax.numpy.identity(HILBERT_SIZE,dtype=np.complex128)
#INITIAL_STATES = np.zeros((HILBERT_SIZE, 1), dtype=np.float64)
#INITIAL_STATES = np.array([INITIAL_STATES], dtype=np.float64)
#INITIAL_STATES = INITIAL_STATES.at[0, 0, 0].set(1.0)
#density0 = np.matmul(INITIAL_STATES, conjugate_transpose(INITIAL_STATES))
#INITIAL_DENSITIES = np.stack((density0,), axis=0)[0]
unitary0 = np.matmul(initial0, conjugate_transpose(initial0))
INITIAL_UNITARIES = np.stack((unitary0,), axis=0)    #[0]

#print("INITIAL_STATES",INITIAL_STATES)
#print("INITIAL_DENSITIES",INITIAL_DENSITIES)
print("INITIAL_UNITARIES",INITIAL_UNITARIES)

coupling_J = 1.0 
coupling_V = 1.0

MODEL_HAMILTONIAN = - coupling_J * plaquette_terms_2(LENGTH, WIDTH ) + coupling_V * corner_terms_2(LENGTH, WIDTH, sigma = np.array(qt.operators.sigmaz().data.toarray()) )
Time_system = 0.01 #ns
TARGET_UNITARIES = np.array([jax.scipy.linalg.expm(-1j * Time_system * MODEL_HAMILTONIAN)], dtype=np.complex128)

# Costs are functions that we want qoc to minimize the output of.
# In this example, we want to minimize the infidelity (maximize the fidelity) of
# the initial state and the target state.
# Note that `COSTS` is a list of cost function objects.
COSTS = [TargetUnitaryInfidelity(TARGET_UNITARIES), ]

# We want to tell qoc how often to store information about the optimization
# and how often to log output. Both `log_iteration_step` and `save_iteration_step`
# are specified in units of optimization iterations.
LOG_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES = False
SAVE_ITERATION_STEP = 1  #0
# set SAVE_ITERATION_STEP to 1 to save results

# For this problem, the LBFGSB optimizer reaches a reasonable
# answer very quickly.
#OPTIMIZER = LBFGSB()
OPTIMIZER = Adam()
ITERATION_COUNT = 1000
# In practice, we find that using a second-order optimizer, such as LBFGSB,
# gives a good initial answer. Then, this answer may be used with a first-
# order optimizer, such as Adam, to achieve the desired error.
# You can seed optimizations with a set of controls using the
# `initial_controls` argument.

#Decide whether to use multilevel loopnest instead of single loop
USE_MULTILEVEL=False   #True

#Decide whether to use custom derivatives for single step
#when USE_MULTILEVEL is False
USE_CUSTOM_STEP=False

# Decide whether to use custom derivatives for the inner loop
# Ignored if USE_MULTILEVEL is False
# No Custom Derivative                     => USE_CUSTOM_INNER = 0
# Custom Derivatives through Storage       => USE_CUSTOM_INNER = 1
# Custom Derivatives through Invertibility => USE_CUSTOM_INNER = 2
# Setting USE_CUSTOM_INNER to 1 or 2 implies that USE_CUSTOM_STEP is True
if USE_CUSTOM_INNER==-1:
	USE_CUSTOM_INNER = 2

# Before we move on, it is a good idea to check that everything looks how you would expect it to.

print("HILBERT_SIZE:\n{}"
      "".format(HILBERT_SIZE))
print("DEVICE_HAMILTONIAN:\n{}"
      "".format(DEVICE_HAMILTONIAN))

print("CONTROL",CONTROL)
'''
print("CONTROL_EVAL_TIMES:\n{}"
      "".format(np.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT)))
print("SYSTEM_EVAL_TIMES:\n{}"
      "".format(np.linspace(0, EVOLUTION_TIME, SYSTEM_EVAL_COUNT)))
'''
print("TARGET_UNITARIES",TARGET_UNITARIES)
#print("TARGET_STATES",TARGET_STATES)
print("COSTS",COSTS)

# qoc saves data in h5 format. You can parse h5 files using the `h5py` package [5].
EXPERIMENT_NAME = "LGT_lattice2_cornerZ_and_plaqS_2by3_ctrlall6_3000pts_500ns_1000iter"
SAVE_PATH = "./out"
H_SIMULATION_FILE_PATH = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)

# Next, we use the GRAPE algorithm to find a set of time-dependent
# controls that accomplishes the state transfer that we desire.
rep_count = 1
#if QUBIT_COUNT < 9:
#  rep_count = 10
tic = time.perf_counter()
for i in range(rep_count):
    result = grape_schroedinger_discrete_unitary(CONTROL_COUNT,
                                     CONTROL_EVAL_COUNT,
                                     COSTS, EVOLUTION_TIME,
                                     hamiltonian,
                                     HILBERT_SIZE,
                                     DEVICE_HAMILTONIAN,
                                     CONTROL,
                                     #INITIAL_STATES,
                                     #INITIAL_DENSITIES,
                                     INITIAL_UNITARIES,
                                     SYSTEM_EVAL_COUNT,
                                     complex_controls=COMPLEX_CONTROLS,
                                     iteration_count=ITERATION_COUNT,
                                     log_iteration_step=LOG_ITERATION_STEP,
                                     optimizer=OPTIMIZER,
                                     save_file_path=H_SIMULATION_FILE_PATH,
                                     save_intermediate_states=SAVE_INTERMEDIATE_STATES,
                                     save_iteration_step=SAVE_ITERATION_STEP,
                                     use_multilevel=USE_MULTILEVEL,
                                     use_custom_inner=USE_CUSTOM_INNER,
                                     use_custom_step=USE_CUSTOM_STEP,
                                     checkpoint_interval=CHECKPOINT_INTERVAL)
toc = time.perf_counter()
tot_time= (toc-tic)/rep_count
#print(f"Time to run code: {toc - tic:0.4f} seconds")
print(f"Time to run code: {tot_time:0.4f} seconds")
# Next, we want to do some analysis of our results.
CONTROLS_PLOT_FILE = "{}_controls_FFT.png".format(EXPERIMENT_NAME)
CONTROLS_PLOT_FILE_PATH = os.path.join(SAVE_PATH, CONTROLS_PLOT_FILE)
#POPULATION_PLOT_FILE = "{}_population.png".format(EXPERIMENT_NAME)
#POPULATION_PLOT_FILE_PATH = os.path.join(SAVE_PATH, POPULATION_PLOT_FILE)
ERROR_PLOT_FILE = "{}_infidelity.png".format(EXPERIMENT_NAME)
ERROR_PLOT_FILE_PATH = os.path.join(SAVE_PATH, ERROR_PLOT_FILE)
SHOW = True
jax.profiler.save_device_memory_profile("memory_"+str(QUBIT_COUNT)+"_"+str(CONTROL_EVAL_COUNT)+"_"+str(CHECKPOINT_INTERVAL)+"_"+str(USE_CUSTOM_INNER)+".prof")
with open('/proc/self/status', 'r') as f:
    print(f.read())

# This function will plot the controls, and their fourier transform.
plot_controls(H_SIMULATION_FILE_PATH,
              save_file_path=CONTROLS_PLOT_FILE_PATH,
              show=SHOW,)
# This function will plot the values of the diagonal elements of the
# density matrix that is formed by taking the outer product of the state
# with itself.
plot_error(H_SIMULATION_FILE_PATH,
              save_file_path=ERROR_PLOT_FILE_PATH,
              show=SHOW,)
'''
plot_state_population(H_SIMULATION_FILE_PATH,
                      save_file_path=POPULATION_PLOT_FILE_PATH,
                      show=SHOW,)
# Both of the above functions plot the iteration that achieved the lowest error
# by default. However, you check out their documentation in the source or
# on qoc's documentation [0] to plot arbitrary iterations.
# QOC locks [6] every h5 file before it writes to it or reads from it.
# Therefore, you can call these plotting functions from a seperate process
# while your optimization is running to see the current controls
# or the current population diagrams.
'''


