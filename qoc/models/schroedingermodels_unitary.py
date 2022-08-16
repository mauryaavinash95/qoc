"""
schroedingermodels.py - This module defines classes that encapsulate 
data fiels used by Schroedinger equation programs.
"""

import os


import jax
import jax.numpy as jnp
import numpy as onp
from filelock import FileLock, Timeout
import h5py

from qoc.models.programtype import ProgramType
from qoc.models.programstate import (ProgramState, GrapeState,)

class EvolveSchroedingerDiscreteStateUnitary(ProgramState):
    """
    This class encapsulates data fields that are used by the
    qoc.core.schroedingerdiscrete.evolve_schroedinger_discrete
    program.
    
    Fields:
    control_eval_count
    control_eval_times
    cost_eval_step
    costs
    dt
    evolution_time
    final_system_eval_step
    hamiltonian
    initial_states
    interpolation_policy
    magnus_policy
    method
    program_type
    save_file_lock_path
    save_file_path
    save_intermediate_states_
    step_cost_indices
    step_costs
    system_eval_count
    """
    method = "evolve_schroedinger_discrete"
    
    def __init__(self,control_eval_count,
                 cost_eval_step, costs,
                 evolution_time, hamiltonian, initial_states,
                 interpolation_policy,
                 magnus_policy,
                 save_file_path,
                 save_intermediate_states_,
                 system_eval_count,):
        """
        See class definition for arguments not listed here.
        """
        super().__init__(control_eval_count,
                         cost_eval_step, costs,
                         evolution_time, hamiltonian, interpolation_policy,
                         ProgramType.EVOLVE,
                         save_file_path, system_eval_count,)
        self.initial_states = initial_states
        self.magnus_policy = magnus_policy
        self.save_intermediate_states_ = (save_file_path is not None
                                          and save_intermediate_states_)


    def save_initial(self, controls):
        """
        Perform the initial save.
        """
        if self.save_file_path is not None:
            print("QOC is saving this evolution to {}."
                  "".format(self.save_file_path))
            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "w") as save_file:
                        save_file["controls"] = controls
                        save_file["cost_eval_step"] = self.cost_eval_step
                        save_file["costs"] = jnp.array(["{}".format(cost)
                                                       for cost in self.costs])
                        save_file["evolution_time"] = self.evolution_time
                        save_file["initial_states"] = self.initial_states
                        save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                        if self.save_intermediate_states_:
                            save_file["intermediate_states"] = jnp.zeros((self.system_eval_count,
                                                                         *self.initial_states.shape),
                                                                        dtype=jnp.complex128)
                        save_file["magnus_policy"] = "{}".format(self.magnus_policy)
                        save_file["method"] = self.method
                        save_file["program_type"] = self.program_type.value
                        save_file["system_eval_count"] = self.system_eval_count
                    #ENDWITH
                #ENDWITH
            except Timeout:
                print("Timeout while locking {}."
                      "".format(self.save_file_lock_path))

    
    def save_intermediate_states(self, iteration, states, system_eval_step,):
        """
        Save intermediate states to the save file.
        """
        if self.save_file_path is not None:
            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "a") as save_file:
                        #save_file["intermediate_states"][system_eval_step, :, :, :] = states.astype(np.complex128)
                        save_file["intermediate_states"][system_eval_step, :, :, :] = states.astype(jnp.complex128)
            except Timeout:
                print("Timeout while locking {} while saving intermediate states on iteration {}."
                      "".format(self.save_file_lock_path, iteration))
        #ENDIF


class EvolveSchroedingerResultUnitary(object):
    """
    This class encapsulates the result of the
    qoc.core.schroedingerdiscrete.evolve_schroedinger_discrete
    program.
    
    Fields:
    error
    final_states
    """

    def __init__(self, error=None,
                 final_states=None,):
        """
        See the class fields for arguments not listed here.
        """
        super().__init__()
        self.error = error
        self.final_states = final_states


class GrapeSchroedingerDiscreteStateUnitary(GrapeState):
    """
    This class encapsulates the data fields used by the
    qoc.core.schroedingerdiscrete.grap_schroedinger_discrete
    program.

    Fields:
    complex_controls
    control_count
    control_eval_count
    control_eval_times
    controls_shape
    cost_eval_step
    costs
    dt
    evolution_time
    final_iteration
    final_system_eval_step
    hamiltonian
    hilbert_size
    impose_control_conditions
    initial_controls
    initial_states
    interpolation_policy
    iteration_count
    log_iteration_step
    max_control_norms
    magnus_policy
    method
    min_error
    optimizer
    program_type
    save_file_lock_path
    save_file_path
    save_intermediate_states_
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    system_eval_count
    inner_propagator
    step_propagator
    checkpoint_interval
    """
    method = "grape_schroedinger_discrete"

    def __init__(self, complex_controls, control_count,
                 control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian,
                 impose_control_conditions,
                 initial_controls,
                 UNITARY_SIZE,
                 SYSTEM_HAMILTONIAN,
                 CONTROL,
                 #initial_states,
                 #initial_densities,
                 initial_unitaries,
                 interpolation_policy, iteration_count,
                 log_iteration_step, max_control_norms,
                 magnus_policy, min_error, optimizer,
                 save_file_path, save_intermediate_states_,
                 save_iteration_step,
                 system_eval_count,
                 use_custom_inner,
                 use_custom_step,
                 checkpoint_interval):
        """
        See class fields for arguments not listed here.
        """
        super().__init__(complex_controls, control_count,
                         control_eval_count, cost_eval_step, costs,
                         evolution_time, hamiltonian,
                         impose_control_conditions,
                         initial_controls,
                         interpolation_policy, iteration_count,
                         log_iteration_step, max_control_norms,
                         min_error, optimizer,
                         save_file_path, save_iteration_step,
                         system_eval_count,)
        self.hilbert_size = initial_unitaries[0].shape[0]
        #self.initial_states = initial_states
        #self.initial_densities = initial_densities
        self.initial_unitaries = initial_unitaries
        # added initial_densities, initial_unitaries
        self.magnus_policy = magnus_policy
        self.save_intermediate_states_ = (self.should_save
                                          and save_intermediate_states_)
        self.UNITARY_SIZE = UNITARY_SIZE
        self.SYSTEM_HAMILTONIAN = SYSTEM_HAMILTONIAN
        self.CONTROL = CONTROL
        self.use_custom_inner = use_custom_inner
        self.use_custom_step = use_custom_step
        self.checkpoint_interval = checkpoint_interval
        #print(self.should_save)

    def log_and_save(self, controls, error, final_unitaries, grads, iteration,):
        """
        If necessary, log to stdout and save to the save file.

        Arguments:
        controls :: ndarray - the optimization parameters
        error :: ndarray - the total error at the last time step
            of evolution
        final_states :: ndarray - the states at the last time step
            of evolution
        grads :: ndarray - the current gradients of the cost function
            with resepct to controls
        iteration :: int - the optimization iteration

        Returns: none
        """
        # Don't log if the iteration number is invalid.
        if iteration > self.final_iteration:
            return
        
        # Determine decision parameters.
        is_final_iteration = iteration == self.final_iteration
        
        if (self.should_log
            and ((jnp.mod(iteration, self.log_iteration_step) == 0)
                 or is_final_iteration)):
            grads_norm = jnp.linalg.norm(grads)
            print("{:^6d} | {:^1.8e} | {:^1.8e}"
                  "".format(iteration, error,
                            grads_norm))

        if (self.should_save
            and ((jnp.mod(iteration, self.save_iteration_step) == 0)
                 or is_final_iteration)):
            save_step, _ = jnp.divmod(iteration, self.save_iteration_step)
            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "a") as save_file:
                        save_file["controls"][save_step,] = controls
                        #print('controls', controls)
                        save_file["error"][save_step,] = error
                        save_file["final_unitaries"][save_step,] = final_unitaries.val
                        save_file["grads"][save_step,] = grads
                    #ENDWITH
                #ENDWITH
            except Timeout:
                print("Timeout while locking {} to save after iteration {}."
                      "".format(self.save_file_lock_path, iteration))


    def log_and_save_initial(self):
        """
        Perform the initial log and save.
        """
        if self.should_save:
            # Notify the user where the file is being saved.
            print("QOC is saving this optimization run to {}."
                  "".format(self.save_file_path))

            save_count, save_count_remainder = jnp.divmod(self.iteration_count,
                                                         self.save_iteration_step)
            state_count = len(self.initial_unitaries)
            # If the final iteration doesn't fall on a save step, add a save step.
            if save_count_remainder != 0:
                save_count += 1

            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "w") as save_file:
                        
                        save_file["complex_controls"] = self.complex_controls
                        save_file["control_count"] = self.control_count
                        save_file["control_eval_count"] = self.control_eval_count
                        save_file["controls"] = jnp.zeros((save_count, self.control_eval_count,
                                                          self.control_count,),
                                                         dtype=self.initial_controls.dtype)
                        save_file["cost_eval_step"] = self.cost_eval_step
                        save_file["cost_names"] = onp.array([onp.string_("{}".format(cost))
                                                            for cost in self.costs])
                        save_file["error"] = jnp.repeat(jnp.finfo(jnp.float64).max, save_count)
                        save_file["evolution_time"]= self.evolution_time
                        save_file["final_unitaries"] = jnp.zeros((save_count, state_count,
                                                              self.hilbert_size, self.hilbert_size),
                                                             dtype=jnp.complex128)
                        save_file["grads"] = jnp.zeros((save_count, self.control_eval_count,
                                                       self.control_count), dtype=self.initial_controls.dtype)
                        save_file["initial_controls"] = self.initial_controls
                        #save_file["initial_states"] = self.initial_states
                        if self.save_intermediate_states_:
                            save_file["intermediate_unitaries"] = jnp.zeros((save_count,
                                                                         self.system_eval_count,
                                                                         *self.initial_unitaries.shape),
                                                                        dtype=jnp.complex128)
                        save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                        save_file["iteration_count"] = self.iteration_count
                        save_file["magnus_policy"] = "{}".format(self.magnus_policy)
                        save_file["max_control_norms"] = self.max_control_norms
                        save_file["method"] = self.method
                        save_file["optimizer"] = "{}".format(self.optimizer)
                        save_file["program_type"] = self.program_type.value
                        save_file["system_eval_count"] = self.system_eval_count
                    #ENDWITH
                #ENDWITH
            except Timeout:
                print("Timeout while locking {}, could not perform initial save."
                      "".format(self.save_file_lock_path))
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")

    
    def save_intermediate_states(self, iteration,
                                 unitaries, system_eval_step,):
        """
        Save intermediate states to the save file.
        """
        # Don't log if the iteration number is invalid.
        if iteration > self.final_iteration:
            return

        # Determine decision parameters.
        is_final_iteration = iteration == self.final_iteration
        
        if (self.should_save
            and ((jnp.mod(iteration, self.save_iteration_step) == 0)
                 or is_final_iteration)):
            save_step, _ = jnp.divmod(iteration, self.save_iteration_step)
            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "a") as save_file:
                        save_file["intermediate_unitaries"][save_step, system_eval_step, :, :, :] = unitaries.astype(jnp.complex128)
            except Timeout:
                print("Timeout while locking {}, could not save intermediate states on iteration {} and "
                      "system_eval_step {}."
                      "".format(self.save_file_lock_path, iteration, system_eval_step))
        #ENDIF


class GrapeSchroedingerResultUnitary(object):
    """
    This class encapsulates the result of the
    qoc.core.lindbladdiscrete.grape_schroedinger_discrete
    program.

    Fields:
    best_controls
    best_error
    best_final_states
    best_iteration
    """
    def __init__(self, best_controls=None,
                 best_error=jnp.finfo(jnp.float64).max,
                 best_final_unitaries=None,
                 best_iteration=None,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__()
        self.best_controls = best_controls
        self.best_error = best_error
        self.best_final_unitaries = best_final_unitaries
        self.best_iteration = best_iteration
