import os

# Every computation that qoc performs has to be differentiable.
# qoc uses autograd [2] to do automatic differentiation [3].
# All operations that you perform on your operands should use autograd's
# numpy wrapper. autograd.numpy wraps the entire numpy namespace, but not all functions
# have derivatives implemented for them. You may view which functionality is supported
# on autograd's github [2].
import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import time
import qutip as qt
import numpy as onp


data_path = '../'
# All of the core functionality that qoc offers can be imported from the top level.
# Core functionality not shown here includes `evolve_schroedinger_discrete`
# and `evolve_lindblad_discrete`.
from qoc import (grape_schroedinger_discrete, grape_lindblad_discrete)

# qoc.standard is a module that has optimization cost functions, optimizers,
# convenience functions, and other goodies. All of the operations
# (e.g. `conjugate_transpose`) that you import
# from qoc.standard use autograd.numpy, so they are OK to use in your operations.
from qoc.standard import (Adam,
    conjugate_transpose, generate_save_file_path,
    get_creation_operator, get_annihilation_operator,
    krons, LBFGSB, matmuls, plot_controls, plot_density_population,
    plot_state_population, TargetDensityInfidelity,
    TargetStateInfidelity,matrix_to_column_vector_list,)

import jax
import matplotlib

# Define the size of the system.
QUBIT_COUNT=1
HILBERT_SIZE = 2**QUBIT_COUNT

# Next, we define the system hamiltonian.
# Control parameters are values that you will use to vary time-dependent control fields
# that act on your system. Note that qoc supports both complex and real control parameters.
matvals=qt.rand_herm(HILBERT_SIZE).data
SYSTEM_HAMILTONIAN=np.array(qt.rand_herm(HILBERT_SIZE).data.toarray())
CONTROL_0 = np.array((qt.rand_herm(HILBERT_SIZE).data).toarray())
CONTROL_0_DAGGER = np.array((qt.rand_herm(HILBERT_SIZE).data).toarray())
CONTROL_1 = np.array((qt.rand_herm(HILBERT_SIZE).data).toarray())
CONTROL_1_DAGGER = np.array((qt.rand_herm(HILBERT_SIZE).data).toarray())

def hamiltonian(controls, time):
    return (SYSTEM_HAMILTONIAN
            + controls[0] * CONTROL_0
            + np.conjugate(controls[0]) * CONTROL_0_DAGGER
            + controls[1] * CONTROL_1
            + np.conjugate(controls[1]) * CONTROL_1_DAGGER)

# Additionally, we need to specify information to qoc about...
# how long our system will evolve for
EVOLUTION_TIME = 15 #ns
# how many controls we have
CONTROL_COUNT = 2
# what domain our controls are in
COMPLEX_CONTROLS = True
# where our controls are positioned in time
CONTROL_EVAL_COUNT = int(1e2)
# and where our system is evaluated in time
SYSTEM_EVAL_COUNT = int(1e2)
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

# Now, we are ready to give qoc a problem.
# Let's try to put a photon in the cavity.
# That is, we desire the fock state transition |0, g> -> |1, g>.
# Notice that when we specify states (or probability density matrices!)
# to qoc, we always give qoc an array of states that we would like it to track,
# even if we only give qoc a single state. The `,` in np.stack((INITIAL_STATE_0`,`))
# makes a difference.

INITIAL_STATES = matrix_to_column_vector_list(jax.numpy.identity(HILBERT_SIZE))
assert(INITIAL_STATES.ndim == 3)
TARGET_STATES=matrix_to_column_vector_list(qt.rand_unitary(HILBERT_SIZE))

# Costs are functions that we want qoc to minimize the output of.
# In this example, we want to minimize the infidelity (maximize the fidelity) of
# the initial state and the target state.
# Note that `COSTS` is a list of cost function objects.
COSTS = [TargetStateInfidelity(TARGET_STATES)]

# We want to tell qoc how often to store information about the optimization
# and how often to log output. Both `log_iteration_step` and `save_iteration_step`
# are specified in units of optimization iterations.
LOG_ITERATION_STEP = 1
SAVE_INTERMEDIATE_STATES = True
SAVE_ITERATION_STEP = 1

# For this problem, the LBFGSB optimizer reaches a reasonable
# answer very quickly.
#OPTIMIZER = LBFGSB()
OPTIMIZER = Adam()
ITERATION_COUNT = 20
# In practice, we find that using a second-order optimizer, such as LBFGSB,
# gives a good initial answer. Then, this answer may be used with a first-
# order optimizer, such as Adam, to achieve the desired error.
# You can seed optimizations with a set of controls using the
# `initial_controls` argument.

#Decide whether to use multilevel loopnest instead of single loop
USE_MULTILEVEL=True

# Before we move on, it is a good idea to check that everything looks how you would expect it to.
print("HILBERT_SIZE:\n{}"
      "".format(HILBERT_SIZE))
print("SYSTEM_HAMILTONIAN:\n{}"
      "".format(SYSTEM_HAMILTONIAN))
print("CONTROL_0",CONTROL_0)
print("CONTROL_0_DAGGER",CONTROL_0_DAGGER)
print("CONTROL_1",CONTROL_1)
print("CONTROL_1_DAGGER",CONTROL_1_DAGGER)
print("CONTROL_EVAL_TIMES:\n{}"
      "".format(np.linspace(0, EVOLUTION_TIME, CONTROL_EVAL_COUNT)))
print("SYSTEM_EVAL_TIMES:\n{}"
      "".format(np.linspace(0, EVOLUTION_TIME, SYSTEM_EVAL_COUNT)))
print("INITIAL_STATES",INITIAL_STATES)
print("TARGET_STATES",TARGET_STATES)
print("COSTS",COSTS)
# qoc saves data in h5 format. You can parse h5 files using the `h5py` package [5].
EXPERIMENT_NAME = "tutorial_schroed_cavity01"
SAVE_PATH = "./out"
SCHROED_FILE_PATH = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)

# Next, we use the GRAPE algorithm to find a set of time-dependent
# controls that accomplishes the state transfer that we desire.
tic = time.perf_counter()
result = grape_schroedinger_discrete(CONTROL_COUNT,
                                     CONTROL_EVAL_COUNT,
                                     COSTS, EVOLUTION_TIME,
                                     hamiltonian,
                                     HILBERT_SIZE,
                                     SYSTEM_HAMILTONIAN,
                                     CONTROL_0,CONTROL_0_DAGGER,
                                     CONTROL_1,CONTROL_1_DAGGER,
                                     INITIAL_STATES,
                                     SYSTEM_EVAL_COUNT,
                                     complex_controls=COMPLEX_CONTROLS,
                                     iteration_count=ITERATION_COUNT,
                                     log_iteration_step=LOG_ITERATION_STEP,
                                     optimizer=OPTIMIZER,
                                     save_file_path=SCHROED_FILE_PATH,
                                     save_intermediate_states=SAVE_INTERMEDIATE_STATES,
                                     save_iteration_step=SAVE_ITERATION_STEP,
                                     use_multilevel=USE_MULTILEVEL)
toc = time.perf_counter()
print(f"Time to run code: {toc - tic:0.4f} seconds")
# Next, we want to do some analysis of our results.
CONTROLS_PLOT_FILE = "{}_controls.png".format(EXPERIMENT_NAME)
CONTROLS_PLOT_FILE_PATH = os.path.join(SAVE_PATH, CONTROLS_PLOT_FILE)
POPULATION_PLOT_FILE = "{}_population.png".format(EXPERIMENT_NAME)
POPULATION_PLOT_FILE_PATH = os.path.join(SAVE_PATH, POPULATION_PLOT_FILE)
SHOW = True
# This function will plot the controls, and their fourier transform.
plot_controls(SCHROED_FILE_PATH,
              save_file_path=CONTROLS_PLOT_FILE_PATH,
              show=SHOW,)
# This function will plot the values of the diagonal elements of the
# density matrix that is formed by taking the outer product of the state
# with itself.
plot_state_population(SCHROED_FILE_PATH,
                      save_file_path=POPULATION_PLOT_FILE_PATH,
                      show=SHOW,)
# Both of the above functions plot the iteration that achieved the lowest error
# by default. However, you check out their documentation in the source or
# on qoc's documentation [0] to plot arbitrary iterations.
# QOC locks [6] every h5 file before it writes to it or reads from it.
# Therefore, you can call these plotting functions from a seperate process
# while your optimization is running to see the current controls
# or the current population diagrams.



