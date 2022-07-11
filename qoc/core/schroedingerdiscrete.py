"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

import jax
import jax.numpy as jnp
import math

from qoc.core.common import (initialize_controls,
                             slap_controls, strip_controls,
                             clip_control_norms,)
from qoc.core.mathmethods import (interpolate_linear_set,
                                  magnus_m2,
                                  magnus_m4,
                                  magnus_m6,)
from qoc.models import (Dummy, EvolveSchroedingerDiscreteState,
                        EvolveSchroedingerResult,
                        GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerResult,
                        InterpolationPolicy,
                        MagnusPolicy,
                        ProgramType,)
from qoc.standard import (Adam, ans_jacobian,
                          matmuls,TargetDensityInfidelity)

### MAIN METHODS ###

def evolve_schroedinger_discrete(evolution_time, hamiltonian,
                                 initial_states, system_eval_count,
                                 controls=None,
                                 cost_eval_step=1, costs=list(), 
                                 interpolation_policy=InterpolationPolicy.LINEAR,
                                 magnus_policy=MagnusPolicy.M2,
                                 save_file_path=None,
                                 save_intermediate_states=False,):
    """
    Evolve a set of state vectors under the schroedinger equation
    and compute the optimization error.

    Args:
    evolution_time :: float - This value specifies the duration of the
        system's evolution.
    hamiltonian :: (controls :: ndarray (control_count), time :: float)
                   -> hamiltonian_matrix :: ndarray (hilbert_size x hilbert_size)
        - This function provides the system's hamiltonian given a set
        of control parameters and a time value.
    initial_states :: ndarray (state_count x hilbert_size x 1)
        - This array specifies the states that should be evolved under the
        specified system. These are the states at the beginning of the evolution.
    system_eval_count :: int >= 2 - This value determines how many times
        during the evolution the system is evaluated, including the
        initial value of the system. For the schroedinger evolution,
        this value determines the time step of integration.
        This value is used as:
        `system_eval_times` = numpy.linspace(0, `evolution_time`, `system_eval_count`).

    controls :: ndarray (control_step_count x control_count)
        - This array specifies the control parameter values at each
          control step. These values will be used to determine the `controls`
          argument passed to the `hamiltonian` function.
    cost_eval_step :: int >= 1- This value determines how often step-costs are evaluated.
         The units of this value are in system_eval steps. E.g. if this value is 2,
         step-costs will be computed every 2 system_eval steps.
    costs :: iterable(qoc.models.cost.Cost) - This list specifies all
        the cost functions that the optimizer should evaluate. This list
        defines the criteria for an "optimal" control set.
    interpolation_policy :: qoc.models.interpolationpolicy.InterpolationPolicy
        - This value specifies how control parameters should be
        interpreted at points where they are not defined.
    magnus_policy :: qoc.models.magnuspolicy.MagnusPolicy - This value
        specifies what method should be used to perform the magnus expansion
        of the system matrix for ode integration. Choosing a higher order
        magnus expansion will yield more accuracy, but it will
        result in a longer compute time.
    save_file_path :: str - This is the full path to the file where
        information about program execution will be stored.
        E.g. "./out/foo.h5"
    save_intermediate_states :: bool - If this value is set to True,
        qoc will write the states to the save file after every
        system_eval step.

    Returns:
    result :: qoc.models.schroedingermodels.EvolveSchroedingerResult
    """
    if controls is not None:
        control_eval_count = controls.shape[0]
    else:
        control_eval_count = 0
    
    pstate = EvolveSchroedingerDiscreteState(control_eval_count,
                                             cost_eval_step,
                                             costs, evolution_time,
                                             hamiltonian, initial_states,
                                             interpolation_policy,
                                             magnus_policy,
                                             save_file_path,
                                             save_intermediate_states,
                                             system_eval_count,)
    pstate.save_initial(controls)
    result = EvolveSchroedingerResult()
    _ = _evaluate_schroedinger_discrete(controls, pstate, result)

    return result


def grape_schroedinger_discrete(control_count, control_eval_count,
                                costs, evolution_time, hamiltonian,
                                UNITARY_SIZE,
                                SYSTEM_HAMILTONIAN,
                                CONTROL_0, CONTROL_0_DAGGER,
                                CONTROL_1, CONTROL_1_DAGGER,
                                initial_states, initial_densities, system_eval_count,
                                complex_controls=False,
                                cost_eval_step=1,
                                impose_control_conditions=None,
                                initial_controls=None,
                                interpolation_policy=InterpolationPolicy.LINEAR,
                                iteration_count=1000, 
                                log_iteration_step=10,
                                magnus_policy=MagnusPolicy.M2,
                                max_control_norms=None,
                                min_error=0,
                                optimizer=Adam(),
                                save_file_path=None,
                                save_intermediate_states=False,
                                save_iteration_step=0,
                                use_multilevel=True,
                                use_custom_inner=0,
                                use_custom_step=False,
                                checkpoint_interval=10):
    """
    This method optimizes the evolution of a set of states under the schroedinger
    equation for time-discrete control parameters.

    Args:
    control_count :: int - This is the number of control parameters that qoc should
        optimize over. I.e. it is the length of the `controls` array passed
        to the hamiltonian.
    control_eval_count :: int >= 2 - This value determines where definite values
        of the control parameters are evaluated. This value is used as:
        `control_eval_times`= numpy.linspace(0, `evolution_time`, `control_eval_count`).
    costs :: iterable(qoc.models.cost.Cost) - This list specifies all
        the cost functions that the optimizer should evaluate. This list
        defines the criteria for an "optimal" control set.
    evolution_time :: float - This value specifies the duration of the
        system's evolution.
    hamiltonian :: (controls :: ndarray (control_count), time :: float)
                   -> hamiltonian_matrix :: ndarray (hilbert_size x hilbert_size)
        - This function provides the system's hamiltonian given a set
        of control parameters and a time value.
    initial_states :: ndarray (state_count x hilbert_size x 1)
        - This array specifies the states that should be evolved under the
        specified system. These are the states at the beginning of the evolution.
    system_eval_count :: int >= 2 - This value determines how many times
        during the evolution the system is evaluated, including the
        initial value of the system. For the schroedinger evolution,
        this value determines the time step of integration.
        This value is used as:
        `system_eval_times` = numpy.linspace(0, `evolution_time`, `system_eval_count`).

    complex_controls :: bool - This value determines if the control parameters
        are complex-valued. If some controls are real only or imaginary only
        while others are complex, real only and imaginary only controls
        can be simulated by taking the real or imaginary part of a complex control.
    cost_eval_step :: int >= 1- This value determines how often step-costs are evaluated.
         The units of this value are in system_eval steps. E.g. if this value is 2,
         step-costs will be computed every 2 system_eval steps.
    impose_control_conditions :: (controls :: (control_eval_count x control_count))
                                 -> (controls :: (control_eval_count x control_count))
        - This function is called after every optimization update. Example uses
        include setting boundary conditions on the control parameters.                             
    initial_controls :: ndarray (control_step_count x control_count)
        - This array specifies the control parameters at each
        control step. These values will be used to determine the `controls`
        argument passed to the `hamiltonian` function at each time step for
        the first iteration of optimization.
    interpolation_policy :: qoc.models.interpolationpolicy.InterpolationPolicy
        - This value specifies how control parameters should be
        interpreted at points where they are not defined.
    iteration_count :: int - This value determines how many total system
        evolutions the optimizer will perform to determine the
        optimal control set.
    log_iteration_step :: int - This value determines how often qoc logs
        progress to stdout. This value is specified in units of system steps,
        of which there are `control_step_count` * `system_step_multiplier`.
        Set this value to 0 to disable logging.
    magnus_policy :: qoc.models.magnuspolicy.MagnusPolicy - This value
        specifies what method should be used to perform the magnus expansion
        of the system matrix for ode integration. Choosing a higher order
        magnus expansion will yield more accuracy, but it will
        result in a longer compute time.
    max_control_norms :: ndarray (control_count) - This array
        specifies the element-wise maximum norm that each control is
        allowed to achieve. If, in optimization, the value of a control
        exceeds its maximum norm, the control will be rescaled to
        its maximum norm. Note that for non-complex values, this
        feature acts exactly as absolute value clipping.
    min_error :: float - This value is the threshold below which
        optimization will terminate.
    optimizer :: class instance - This optimizer object defines the
        gradient-based procedure for minimizing the total contribution
        of all cost functions with respect to the control parameters.
    save_file_path :: str - This is the full path to the file where
        information about program execution will be stored.
        E.g. "./out/foo.h5"
    save_intermediate_densities :: bool - If this value is set to True,
        qoc will write the densities to the save file after every
        system_eval step.
    save_intermediate_states :: bool - If this value is set to True,
        qoc will write the states to the save file after every
        system_eval step.
    save_iteration_step :: int - This value determines how often qoc
        saves progress to the save file specified by `save_file_path`.
        This value is specified in units of system steps, of which
        there are `control_step_count` * `system_step_multiplier`.
        Set this value to 0 to disable saving.

    Returns:
    result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
    """
    # Initialize the controls.
    initial_controls, max_control_norms = initialize_controls(complex_controls,
                                                              control_count,
                                                              control_eval_count,
                                                              evolution_time,
                                                              initial_controls,
                                                              max_control_norms)
        
    # Construct the program state.
    pstate = GrapeSchroedingerDiscreteState(complex_controls, control_count,
                                            control_eval_count, cost_eval_step,
                                            costs, evolution_time, hamiltonian,
                                            impose_control_conditions,
                                            initial_controls,
                                            UNITARY_SIZE,
                                            SYSTEM_HAMILTONIAN,
                                            CONTROL_0, CONTROL_0_DAGGER,
                                            CONTROL_1, CONTROL_1_DAGGER,
                                            initial_states, initial_densities, interpolation_policy,
                                            iteration_count,
                                            log_iteration_step,
                                            max_control_norms, magnus_policy,
                                            min_error, optimizer,
                                            save_file_path,
                                            save_intermediate_states,
                                            save_iteration_step,
                                            system_eval_count,
                                            use_custom_inner,
                                            use_custom_step,
                                            checkpoint_interval)
    pstate.log_and_save_initial()

    # Autograd does not allow multiple return values from
    # a differentiable function.
    # Scipy's minimization algorithms require us to provide
    # functions that they evaluate on their own schedule.
    # The best solution to track mutable objects, that I can think of,
    # is to use a reporter object.
    reporter = Dummy()
    reporter.iteration = 0
    result = GrapeSchroedingerResult()
    # Convert the controls from cost function format to optimizer format.
    initial_controls = strip_controls(pstate.complex_controls, pstate.initial_controls)
    # Choose the propagator
    propagator = _evaluate_schroedinger_discrete
    if use_multilevel:
        propagator = _evaluate_schroedinger_discrete_multilevel
    # Run the optimization.
    pstate.optimizer.run(_esd_wrap, pstate.iteration_count, initial_controls,
                         _esdj_wrap, args=(pstate, reporter, result, propagator))

    return result


### HELPER METHODS ###

def _esd_wrap(controls, pstate, reporter, result, propagator):
    """
    Do intermediary work between the optimizer feeding controls
    to propagator().

    Args:
    controls
    pstate
    reporter
    result

    Returns:
    error
    """
    # Convert the controls from optimizer format to cost function format.
    controls = slap_controls(pstate.complex_controls, controls,
                             pstate.controls_shape)
    # Rescale the controls to their maximum norm.
    clip_control_norms(controls,
                       pstate.max_control_norms)
    # Impose user boundary conditions.
    if pstate.impose_control_conditions:
        controls = pstate.impose_control_conditions(controls)

    # Evaluate the cost function.
    error = propagator(controls, pstate, reporter)

    # Determine if optimization should terminate.
    if error <= pstate.min_error:
        terminate = True
    else:
        terminate = False

    return error, terminate


def _esdj_wrap(controls, pstate, reporter, result, propagator):
    """
    Do intermediary work between the optimizer feeding controls to 
    the jacobian of propagator().

    Args:
    controls
    pstate
    reporter
    result

    Returns:
    grads
    """
    # Convert the controls from optimizer format to cost function format.
    controls = slap_controls(pstate.complex_controls, controls,
                             pstate.controls_shape)
    # Rescale the controls to their maximum norm.
    clip_control_norms(controls,
                       pstate.max_control_norms)
    # Impose user boundary conditions.
    if pstate.impose_control_conditions is not None:
        controls = pstate.impose_control_conditions(controls)

    # Evaluate the jacobian.
    error, grads = (ans_jacobian(propagator, 0)
                          (controls, pstate, reporter))
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        grads = jnp.conjugate(grads)

    # The states need to be unwrapped from their autograd box.
    final_states = reporter.final_states

    # Update best configuration.
    if error < result.best_error:
        result.best_controls = controls
        result.best_error = error
        result.best_final_states = final_states
        result.best_iteration = reporter.iteration
    
    # Save and log optimization progress.
    pstate.log_and_save(controls, error, final_states,
                        grads, reporter.iteration)
    reporter.iteration += 1

    # Convert the gradients from cost function to optimizer format.
    grads = strip_controls(pstate.complex_controls, grads)

    # Determine if optimization should terminate.
    if error <= pstate.min_error:
        terminate = True
    else:
        terminate = False
    
    return grads, terminate


def _evaluate_schroedinger_discrete(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Arguments:
    controls :: ndarray (control_eval_count x control_count)
        - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter :: any - a reporter for mutable objects

    Returns:
    error :: float - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    control_eval_times = pstate.control_eval_times
    cost_eval_step = pstate.cost_eval_step
    costs = pstate.costs
    dt = pstate.dt
    evolution_time = pstate.evolution_time
    final_system_eval_step = pstate.final_system_eval_step
    hamiltonian = pstate.hamiltonian
    interpolation_policy = pstate.interpolation_policy
    magnus_policy = pstate.magnus_policy
    program_type = pstate.program_type
    if program_type == ProgramType.GRAPE:
        iteration = reporter.iteration
    else:
        iteration = 0
    save_intermediate_states = pstate.save_intermediate_states_
    states = pstate.initial_states
    densities = pstate.initial_densities
    step_costs = pstate.step_costs
    system_eval_count = pstate.system_eval_count
    error = 0

    # Evolve the states to `evolution_time`.
    # Compute step-costs along the way.
    for system_eval_step in range(system_eval_count):
        # If applicable, save the current states.
        """
        if save_intermediate_states:
            intermediate_states = states
            pstate.save_intermediate_states(iteration,
                                            intermediate_states,
                                            system_eval_step,)
        """
        
        # Determine where we are in the mesh.
        cost_step, cost_step_remainder = divmod(system_eval_step, cost_eval_step)
        is_cost_step = cost_step_remainder == 0
        is_first_system_eval_step = system_eval_step == 0
        is_final_system_eval_step = system_eval_step == final_system_eval_step
        time = system_eval_step * dt
        
        # Compute step costs every `cost_step`.
        if is_cost_step and not is_first_system_eval_step:
            for i, step_cost in enumerate(step_costs):
                cost_error = step_cost.cost(controls, states, system_eval_step)
                error = error + cost_error
            #ENDFOR
        
        # Evolve the states to the next time step.
        if not is_final_system_eval_step:
            states,densities = _evolve_step_schroedinger_discrete(dt,
                                                        pstate.SYSTEM_HAMILTONIAN,
                                                        pstate.CONTROL_0, pstate.CONTROL_0_DAGGER,
                                                        pstate.CONTROL_1, pstate.CONTROL_1_DAGGER,
                                                        states,densities, time,
                                                        control_eval_times=control_eval_times,
                                                        controls=controls,)
    #ENDFOR

    # Compute non-step-costs.
    for i, cost in enumerate(costs):
        if not cost.requires_step_evaluation:
            cost_error = cost.cost(controls, states, final_system_eval_step)
            error = error + cost_error

    # Report reults.
    reporter.error = error
    reporter.final_states = states
    
    return error


def _evolve_step_schroedinger_discrete(dt,
                                       SYSTEM_HAMILTONIAN,
                                       CONTROL_0, CONTROL_0_DAGGER,
                                       CONTROL_1, CONTROL_1_DAGGER,
                                       states, densities, time,
                                       control_eval_times=None,
                                       controls=None,):
    """
    Use the exponential series method via magnus expansion to evolve the state vectors
    to the next time step under the schroedinger equation for time-discrete controls.
    Magnus expansions are implemented using the methods described in
    https://arxiv.org/abs/1709.06483.

    Arguments:
    dt
    hamiltonian
    states
    time

    control_eval_times
    controls
    interpolation_policy
    magnus_policy
    
    Returns:
    states
    """
    t1 = time + dt * 0.5
    index = jnp.argmax(t1 <= control_eval_times)
     
    controls_ = controls[index - 1] + (((controls[index] - controls[index - 1]) / (control_eval_times[index] - control_eval_times[index - 1])) * (t1 - control_eval_times[index - 1]))
    hamiltonian_ = (SYSTEM_HAMILTONIAN
             + controls_[0] * CONTROL_0
             + jnp.conjugate(controls_[0]) * CONTROL_0_DAGGER
             + controls_[1] * CONTROL_1
             + jnp.conjugate(controls_[1]) * CONTROL_1_DAGGER)
    a1 = -1j * hamiltonian_
    magnus = dt * a1
    step_unitary = jax.scipy.linalg.expm(magnus)
    states = jnp.matmul(step_unitary, states)
    densities = jnp.matmul(step_unitary, densities)
    return (states, densities)

def _evaluate_schroedinger_discrete_multilevel(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Arguments:
    controls :: ndarray (control_eval_count x control_count)
        - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter :: any - a reporter for mutable objects

    Returns:
    error :: float - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    control_eval_times = pstate.control_eval_times
    cost_eval_step = pstate.cost_eval_step
    costs = pstate.costs
    dt = pstate.dt
    evolution_time = pstate.evolution_time
    final_system_eval_step = pstate.final_system_eval_step
    hamiltonian = pstate.hamiltonian
    interpolation_policy = pstate.interpolation_policy
    magnus_policy = pstate.magnus_policy
    program_type = pstate.program_type
    if program_type == ProgramType.GRAPE:
        iteration = reporter.iteration
    else:
        iteration = 0
    save_intermediate_states = pstate.save_intermediate_states_
    states = pstate.initial_states
    densities = pstate.initial_densities
    step_costs = pstate.step_costs
    system_eval_count = pstate.system_eval_count
    error = 0
    
    
    step_propagator = _evolve_step_schroedinger_discrete
    if pstate.use_custom_step:
        step_propagator = _evolve_step_schroedinger_discrete_custom
    
    """
    BEGIN ITERATION 0
    """
    system_eval_step = 0
    cost_step, cost_step_remainder = jnp.divmod(system_eval_step, cost_eval_step)
    is_cost_step = cost_step_remainder == 0
    time = system_eval_step * dt
    states, densities = step_propagator(dt,
           pstate.SYSTEM_HAMILTONIAN,
           pstate.CONTROL_0, pstate.CONTROL_0_DAGGER,
           pstate.CONTROL_1, pstate.CONTROL_1_DAGGER,
        states, densities, time,
        control_eval_times=control_eval_times,
        controls=controls,)
    """
    END ITERATION 0
    """
    """
    BEGIN main iteration block
    """
    states, densities = _evaluate_schroedinger_discrete_loop_outer(
                                         system_eval_count,cost_eval_step,
                                         dt, pstate,
                                         states, densities, control_eval_times,controls,
                                         pstate.checkpoint_interval)
    """
    END main iteration block
    """
    """
    BEGIN ITERATION system_eval_count-1
    """
    cost_step, cost_step_remainder = jnp.divmod(system_eval_count-1, cost_eval_step)
    is_cost_step = cost_step_remainder == 0
    time = (system_eval_count-1) * dt
    # Compute step costs every `cost_step`.
    if is_cost_step:
        for i, step_cost in enumerate(step_costs):
            cost_error = step_cost.cost(controls, states, system_eval_step)
            error = error + cost_error
        #ENDFOR
    """
    END ITERATION system_eval_count-1
    """
    # Compute non-step-costs.
    for i, cost in enumerate(costs):
        if not cost.requires_step_evaluation:
            if cost.cost == TargetDensityInfidelity.cost:
              cost_error = cost.cost(controls, densities, final_system_eval_step)
            else:
              cost_error = cost.cost(controls, states, final_system_eval_step)
            error = error + cost_error

    # Report reults.
    reporter.error = error
    reporter.final_states = states
    
    return error

def _evaluate_schroedinger_discrete_loop_outer(system_eval_count,cost_eval_step,
                                         dt, pstate,
                                         states, densities, control_eval_times,controls, checkpoint_interval):
                                         
    UNITARY_SIZE = pstate.UNITARY_SIZE
    SYSTEM_HAMILTONIAN = pstate.SYSTEM_HAMILTONIAN
    CONTROL_0 = pstate.CONTROL_0
    CONTROL_0_DAGGER = pstate.CONTROL_0_DAGGER
    CONTROL_1 = pstate.CONTROL_1
    CONTROL_1_DAGGER = pstate.CONTROL_1_DAGGER
    densities_store=None
    magnus_store=None
    index_store=None
    state_store = None
    if pstate.use_custom_inner==1:
      state_store=jnp.zeros((checkpoint_interval,UNITARY_SIZE,UNITARY_SIZE,
                               1),dtype=states.dtype)
      densities_store=jnp.zeros((checkpoint_interval,UNITARY_SIZE,UNITARY_SIZE
                               ),dtype=states.dtype)
      magnus_store=jnp.zeros((checkpoint_interval,
                        UNITARY_SIZE,UNITARY_SIZE),dtype=states.dtype)
      index_store=jnp.zeros((checkpoint_interval),dtype=jnp.integer)
    
    def _evaluate_schroedinger_discrete_loop_inner(start, stop,cost_eval_step,
                                             dt,
                                             states, densities, control_eval_times,controls):
        # Evolve the states to `evolution_time`.
        # Compute step-costs along the way.
        for system_eval_step in range(start,stop):
            
            # Determine where we are in the mesh.
            cost_step, cost_step_remainder = jnp.divmod(system_eval_step, cost_eval_step)
            is_cost_step = cost_step_remainder == 0
            time = system_eval_step * dt
            
            # Evolve the states to the next time step.
            states, densities  = _evolve_step_schroedinger_discrete(dt,
            SYSTEM_HAMILTONIAN,
            CONTROL_0, CONTROL_0_DAGGER,
            CONTROL_1, CONTROL_1_DAGGER,
                                                                states, densities, time,
                                                            control_eval_times=control_eval_times,
                                                                controls=controls,)
        return states, densities

    @jax.custom_vjp
    def _evaluate_schroedinger_discrete_loop_inner_custom_store(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls):
        states, densities = _evaluate_schroedinger_discrete_loop_inner(start, stop,cost_eval_step,
                                             dt,
                                             states, densities, control_eval_times,controls)
        return states, densities
                                             

    @jax.profiler.trace_function
    def _evaluate_schroedinger_discrete_loop_inner_custom_store_fwd(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls):
        start_states=states
        start_densities=densities
        
        states, densities=_evaluate_schroedinger_discrete_loop_inner(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls)
        #Here we store the stating state for use in the backward pass
        return (states,densities), (start, stop,cost_eval_step,
                                             dt,
                                             start_states, start_densities,
                                             control_eval_times,controls)
    @jax.profiler.trace_function
    def _evaluate_schroedinger_discrete_loop_inner_custom_store_bwd(res,g_prod):
        nonlocal index_store
        nonlocal magnus_store
        nonlocal state_store
        nonlocal densities_store
        start, stop,cost_eval_step, dt,states,densities,control_eval_times,controls=res
        #Go forward in timesteps storing the states and the unitary
        _M2_C1 = 0.5
        for i in range(start,stop):
            time = i * dt
            t1 = time + dt * 0.5
            index = jnp.argmax(t1 <= control_eval_times)
            index_store = index_store.at[i-start].set(index)
            #index_store=jax.ops.index_update(index_store, jax.ops.index[i-start],index)
            # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
            controls_ = controls[index - 1] + (((controls[index] - controls[index - 1]) / (control_eval_times[index] - control_eval_times[index - 1])) * (t1 - control_eval_times[index - 1]))
            hamiltonian_ = (SYSTEM_HAMILTONIAN
                     + controls_[0] * CONTROL_0
                     + jnp.conjugate(controls_[0]) * CONTROL_0_DAGGER
                     + controls_[1] * CONTROL_1
                     + jnp.conjugate(controls_[1]) * CONTROL_1_DAGGER)
            a1 = -1j * hamiltonian_
            magnus = dt * a1
            magnus_store = magnus_store.at[i-start].set(magnus)
            #magnus_store=jax.ops.index_update(magnus_store, jax.ops.index[i-start],magnus)
            # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
            step_unitary, f_expm_grad = jax.vjp(jax.scipy.linalg.expm, (magnus), has_aux=False)
            state_store = state_store.at[i-start].set(states)
            densities_store = densities_store.at[i-start].set(densities)
            #state_store=jax.ops.index_update(state_store, jax.ops.index[i-start],states)
            #densities_store=jax.ops.index_update(densities_store, jax.ops.index[i-start],densities)
            # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
            states, f_matmul = jax.vjp(jnp.matmul,step_unitary, states)
            densities, f_matmul_densities = jax.vjp(jnp.matmul,step_unitary, densities)
        controlsb = jnp.zeros(controls.shape, states.dtype)
        #Begin U-Turn Go backwards in timesteps
        for i in range(stop-1,start-1,-1):
            time = i * dt
            t1 = time + dt * _M2_C1
            index=index_store[i-start]
            step_unitary, f_expm_grad = jax.vjp(jax.scipy.linalg.expm, magnus_store[i-start], has_aux=False)
            _, f_matmul = jax.vjp(jnp.matmul,step_unitary, state_store[i-start])
            _, f_matmul_densities = jax.vjp(jnp.matmul,step_unitary, densities_store[i-start])
            step_unitaryb,densitiesb=f_matmul_densities(g_prod[1])
            step_unitaryb,statesb=f_matmul(g_prod[0])
            magnusb = f_expm_grad(step_unitaryb)
            a1b=dt*magnusb[0]
            hamiltonian_b = jnp.conjugate(-1j)*a1b
            controls1b=jnp.array((jnp.sum(jnp.conjugate(CONTROL_0)*hamiltonian_b) +
               jnp.conjugate(jnp.sum(jnp.conjugate(CONTROL_0_DAGGER)*hamiltonian_b)),
               jnp.sum(jnp.conjugate(CONTROL_1)*hamiltonian_b) +
               jnp.conjugate(jnp.sum(jnp.conjugate(CONTROL_1_DAGGER)*hamiltonian_b))),
               dtype=hamiltonian_b.dtype)
            tempb = (t1-control_eval_times[index-1])*controls1b/(control_eval_times[index]-control_eval_times[index-1])
            controlsb = controlsb.at[index-1].set(controlsb[index-1]+controls1b - tempb)
            controlsb = controlsb.at[index].set(controlsb[index]+tempb)
            #controlsb=jax.ops.index_update(controlsb, jax.ops.index[index-1],controlsb[index-1]+controls1b - tempb)
            #controlsb=jax.ops.index_update(controlsb, jax.ops.index[index],controlsb[index]+tempb)
            # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
            g_prod=(statesb,densitiesb)
        return (0.0,0.0,0.0,0.0,statesb,densitiesb,0.0,-1*controlsb)

    _evaluate_schroedinger_discrete_loop_inner_custom_store.defvjp(_evaluate_schroedinger_discrete_loop_inner_custom_store_fwd, _evaluate_schroedinger_discrete_loop_inner_custom_store_bwd)

    @jax.custom_vjp
    def _evaluate_schroedinger_discrete_loop_inner_custom_inv(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls):
        states, densities = _evaluate_schroedinger_discrete_loop_inner(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls)
        return states,densities

    @jax.profiler.trace_function
    def _evaluate_schroedinger_discrete_loop_inner_custom_inv_fwd(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls):
        states,densities=_evaluate_schroedinger_discrete_loop_inner(start, stop,cost_eval_step,
                                             dt,
                                             states,densities,control_eval_times,controls)
        #Here we store the final state for use in the backward pass
        return (states,densities), (start, stop,cost_eval_step,
                                             dt,
                                             states,densities,
                                             control_eval_times,controls)
    def _evaluate_schroedinger_discrete_loop_inner_custom_inv_bwd(res,g_prod):
        start, stop,cost_eval_step, dt, states,densities,control_eval_times,controls=res
        #Go forward in timesteps storing the controls only
        _M2_C1 = 0.5
        controlsb = jnp.zeros(controls.shape, states.dtype)
        #Go backwards in timesteps
        for i in range(stop-1,start-1,-1):
            #Reapply controls to compute a step unitary matrix
            time = i * dt
            t1 = time + dt * _M2_C1
            x = t1
            xs = control_eval_times
            ys = controls
            index = jnp.argmax(x <= xs)
            y = ys[index - 1] + (((ys[index] - ys[index - 1]) / (xs[index] - xs[index - 1])) * (x - xs[index - 1]))
            controls_ = y

            hamiltonian_ = (SYSTEM_HAMILTONIAN
                     + controls_[0] * CONTROL_0
                     + jnp.conjugate(controls_[0]) * CONTROL_0_DAGGER
                     + controls_[1] * CONTROL_1
                     + jnp.conjugate(controls_[1]) * CONTROL_1_DAGGER)
            a1 = -1j * hamiltonian_
            magnus = dt * a1
            step_unitary, f_expm_grad = jax.vjp(jax.scipy.linalg.expm, (magnus), has_aux=False)
            #Exploit invertibility of unitary matrix and calculate previous sta
            step_unitary_inv=jnp.conj(jnp.transpose(step_unitary))
            states=(jnp.matmul(step_unitary_inv,states))
            densities=(jnp.matmul(step_unitary_inv,densities))
            _, f_matmul = jax.vjp(jnp.matmul,step_unitary, states)
            _, f_matmul_densities = jax.vjp(jnp.matmul,step_unitary, densities)
            #Go backwards for the timestep
            step_unitaryb,densitiesb=f_matmul_densities(g_prod[1])
            step_unitaryb,statesb=f_matmul(g_prod[0])
            magnusb = f_expm_grad(step_unitaryb)
            a1b=dt*magnusb[0]
            hamiltonian_b = jnp.conjugate(-1j)*a1b
            controls1b=jnp.array((jnp.sum(jnp.conjugate(CONTROL_0)*hamiltonian_b) +
               jnp.conjugate(jnp.sum(jnp.conjugate(CONTROL_0_DAGGER)*hamiltonian_b)),
               jnp.sum(jnp.conjugate(CONTROL_1)*hamiltonian_b) +
               jnp.conjugate(jnp.sum(jnp.conjugate(CONTROL_1_DAGGER)*hamiltonian_b))),
               dtype=hamiltonian_b.dtype)
            tempb = (x-control_eval_times[index-1])*controls1b/(control_eval_times[index]-control_eval_times[index-1])
            controlsb = controlsb.at[index-1].set(controlsb[index-1]+controls1b - tempb)
            controlsb = controlsb.at[index].set(controlsb[index]+tempb)
            #controlsb=jax.ops.index_update(controlsb, jax.ops.index[index-1],controlsb[index-1]+controls1b - tempb)
            #controlsb=jax.ops.index_update(controlsb, jax.ops.index[index],controlsb[index]+tempb)
            # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
            g_prod=statesb,densitiesb
        return (0.0,0.0,0.0,0.0,statesb,densitiesb,0.0,-1*controlsb)

    _evaluate_schroedinger_discrete_loop_inner_custom_inv.defvjp(_evaluate_schroedinger_discrete_loop_inner_custom_inv_fwd, _evaluate_schroedinger_discrete_loop_inner_custom_inv_bwd)

    @jax.custom_vjp
    def _evolve_step_schroedinger_discrete_custom(dt,
                                           states, densities, time,
                                           control_eval_times,
                                           controls):
        """
        Utility to call _evolve_step_schroedinger_discrete()
        It is decorated with jax.custom_vjp. This way
        _evolve_step_schroedinger_discrete() does not have to
        be decorated and can be used in a non custom manner
        """
        return _evolve_step_schroedinger_discrete(dt,
                                                  states, densities, time,
                                                  control_eval_times,
                                                  controls)

    def _evolve_step_schroedinger_discrete_custom_fwd(dt,
                                                states, densities, time,
                                                control_eval_times,
                                                controls):
        states_start = states
        densities_start = densities
        states,densities=_evolve_step_schroedinger_discrete(dt,
                                                states, densities, time,
                                                control_eval_times,
                                                controls)
        return (states,densities), (dt,
                            states_start, densities_start, time,
                            control_eval_times,
                            controls)

    def _evolve_step_schroedinger_discrete_custom_bwd(res,g_prod):
        dt,states,densities, time,control_eval_times,controls=res
        #Reapply the control signals and recompute unitary matrix
        t1 = time + dt * 0.5
        index = jnp.argmax(t1 <= control_eval_times)
        controls_ = controls[index - 1] + (((controls[index] - controls[index - 1]) / (control_eval_times[index] - control_eval_times[index - 1])) * (t1 - control_eval_times[index - 1]))
        hamiltonian_ = (SYSTEM_HAMILTONIAN
                 + controls_[0] * CONTROL_0
                 + jnp.conjugate(controls_[0]) * CONTROL_0_DAGGER
                 + controls_[1] * CONTROL_1
                 + jnp.conjugate(controls_[1]) * CONTROL_1_DAGGER)
        a1 = -1j * hamiltonian_
        magnus = dt * a1
        step_unitary, f_expm_grad = jax.vjp(jax.scipy.linalg.expm, (magnus), has_aux=False)
        _, f_matmul = jax.vjp(jnp.matmul,step_unitary, states)
        _, f_matmul_densities = jax.vjp(jnp.matmul,step_unitary, densities)

        #Go backwards for the timestep
        step_unitaryb,densitiesb=f_matmul_densities(g_prod[1])
        step_unitaryb,statesb=f_matmul(g_prod[0])
        magnusb = f_expm_grad(step_unitaryb)
        a1b=dt*magnusb[0]
        hamiltonian_b = jnp.conjugate(-1j)*a1b
        controls1b=jnp.array((jnp.sum(jnp.conjugate(CONTROL_0)*hamiltonian_b) +
           jnp.conjugate(jnp.sum(jnp.conjugate(CONTROL_0_DAGGER)*hamiltonian_b)),
           jnp.sum(jnp.conjugate(CONTROL_1)*hamiltonian_b) +
           jnp.conjugate(jnp.sum(jnp.conjugate(CONTROL_1_DAGGER)*hamiltonian_b))),
           dtype=hamiltonian_b.dtype)
        tempb = (t1-control_eval_times[index-1])*controls1b/(control_eval_times[index]-control_eval_times[index-1])
        controlsb = jnp.zeros(controls.shape, controls1b.dtype)
        controlsb = controlsb.at[index-1].set(controls1b - tempb)
        controlsb = controlsb.at[index].set(tempb)
        #controlsb=jax.ops.index_update(controlsb, jax.ops.index[index-1],controls1b - tempb)
        #controlsb=jax.ops.index_update(controlsb, jax.ops.index[index],tempb)
        # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
        return (0.0,statesb,densitiesb,0.0,0.0,-1*controlsb)
        
    _evolve_step_schroedinger_discrete_custom.defvjp(_evolve_step_schroedinger_discrete_custom_fwd, _evolve_step_schroedinger_discrete_custom_bwd)
    
    if pstate.use_custom_inner==3: #This is the use the default inner and ignore checkpooint distance
        inner_propagator = _evaluate_schroedinger_discrete_loop_inner
        states,densities = inner_propagator(1,system_eval_count-1,cost_eval_step,
                                    dt,
                                    states,densities,control_eval_times,controls)
    elif pstate.use_custom_inner==4: #This is the use the invertibility inner and ignore checkpooint distance
        inner_propagator = _evaluate_schroedinger_discrete_loop_inner_custom_inv
        states,densities = inner_propagator(1,system_eval_count-1,cost_eval_step,
                                    dt,
                                    states,densities,control_eval_times,controls)
    
    else:
      inner_propagator = _evaluate_schroedinger_discrete_loop_inner
      if pstate.use_custom_inner==1:
          inner_propagator = _evaluate_schroedinger_discrete_loop_inner_custom_store
      if pstate.use_custom_inner==2:
          inner_propagator = _evaluate_schroedinger_discrete_loop_inner_custom_inv
      if pstate.use_custom_inner==5:
          inner_propagator = _evaluate_schroedinger_discrete_loop_inner
      # Evolve the states to `evolution_time`.
      # Compute step-costs along the way.
      valslen=math.ceil(len(range(1,system_eval_count-1))/checkpoint_interval)
      states,densities = inner_propagator(1,checkpoint_interval,cost_eval_step,
                                    dt,
                                    states,densities,control_eval_times,controls)
    
      for i in range(1,valslen-1):
          states,densities = inner_propagator(checkpoint_interval*i,checkpoint_interval*(i+1),cost_eval_step,
                                    dt,
                                    states,densities,control_eval_times,controls)
             
                                                            
      #ENDFOR
    
      i=valslen-1
      states,densities = inner_propagator(checkpoint_interval*i,system_eval_count-1,cost_eval_step,
                                    dt,
                                    states,densities,control_eval_times,controls)
    
    return states,densities

