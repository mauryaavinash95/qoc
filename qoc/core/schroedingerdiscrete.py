"""
schroedingerdiscrete.py - a module to expose the grape schroedinger discrete
optimization algorithm
"""

from autograd.extend import Box
import numpy as np

from qoc.core.common import (initialize_controls,
                             slap_controls, strip_controls,
                             clip_control_norms,)
from qoc.models import (Dummy, EvolveSchroedingerDiscreteState,
                        EvolveSchroedingerResult,
                        GrapeSchroedingerDiscreteState,
                        GrapeSchroedingerResult,
                        interpolate_linear,
                        InterpolationPolicy,
                        magnus_m2, magnus_m4, magnus_m6,
                        MagnusPolicy, OperationPolicy,
                        PerformancePolicy,)
from qoc.standard import (Adam, ans_jacobian,
                          conjugate, expm,
                          conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def evolve_schroedinger_discrete(control_step_count, evolution_time,
                                 hamiltonian, initial_states,
                                 controls=None, costs=list(),
                                 interpolation_policy=InterpolationPolicy.LINEAR,
                                 magnus_policy=MagnusPolicy.M2,
                                 operation_policy=OperationPolicy.CPU,
                                 system_step_multiplier=1,):
    """
    Evolve a set of state vectors under the schroedinger equation
    and compute the optimization error.

    Arguments:
    control_step_count :: int - This value is the number of positions
        in time where control parameters are fit. For example, if the
        evolution time is 10ns and the control step count is 2,
        control parameters will be fit at 0ns and 5ns. The control
        parameters at other times will be interpolated based on the ones
        fit at those values.
    controls :: ndarray (control_step_count x control_count)
        - This array specifies the control parameter values at each
          control step. These values will be used to determine the `controls`
          argument passed to the `hamiltonian` function.
    costs :: iterable(qoc.models.cost.Cost) - This list specifies all
        the cost functions that the optimizer should evaluate. This list
        defines the criteria for an "optimal" control set.
    evolution_time :: float - This value specifies the duration of the
        system's evolution.
    hamiltonian :: (controls :: ndarray (control_count), time :: float)
                   -> hamiltonian :: ndarray (hilbert_size x hilbert_size)
        - This function provides the system's hamiltonian given a set
        of control parameters and a time value. This function must
        use operations registered with autograd. I.e. use
        autograd.numpy instead of numpy.
    initial_states :: ndarray (state_count x hilbert_size x 1)
        - This array specifies the states that should be evolved under the
        specified system. These are the states at the beggining of the evolution.
    interpolation_policy :: qoc.models.interpolationpolicy.InterpolationPolicy
        - This value specifies how control parameters should be
        interpreted at points where they are not defined.
    magnus_policy :: qoc.models.magnuspolicy.MagnusPolicy - This value
        specifies what method should be used to perform the magnus expansion
        of the system matrix for ode integration. Choosing a higher order
        magnus expansion will yield more accuracy, but it will
        result in a longer compute time.
    operation_policy :: qoc.models.operationpolicy.OperationPolicy
        - This value specifies which computation backend should be used
        to execute the program.
    system_step_multiplier :: int - By default, qoc will integrate the
        ode at each time interval produced by dividing `evolution_time`
        into `control_step_count`intervals. Setting the
        `system_step_multiplier` value will change qoc's behavior to
        integrate the ode at each time interval produced by dividing
        `evolution_time`into `control_step_count` * `system_step_multiplier`
        intervals. This will result in higher accuracy, but more compute time.
        Additionaly, each cost function that requires evaluation at each
        time step will also be evaluated at `control_step_count`
        * `system_step_multiplier` time steps.

    Returns:
    result :: qoc.models.schroedingermodels.EvolveSchroedingerResult
    """
    pstate = EvolveSchroedingerDiscreteState(control_step_count,
                                             costs, evolution_time,
                                             hamiltonian, initial_states,
                                             interpolation_policy,
                                             magnus_policy, operation_policy,
                                             system_step_multiplier,)
    result = EvolveSchroedingerResult()
    _ = _evaluate_schroedinger_discrete(controls, pstate, result)

    return result


def grape_schroedinger_discrete(control_count, control_step_count,
                                costs, evolution_time, hamiltonian, initial_states,
                                complex_controls=False,
                                initial_controls=None,
                                interpolation_policy=InterpolationPolicy.LINEAR,
                                iteration_count=1000, 
                                log_iteration_step=10,
                                magnus_policy=MagnusPolicy.M2,
                                max_control_norms=None,
                                minimum_error=0,
                                operation_policy=OperationPolicy.CPU,
                                optimizer=Adam(),
                                performance_policy=PerformancePolicy.TIME,
                                save_file_path=None, save_iteration_step=0,
                                system_step_multiplier=1,):
    """
    This method optimizes the evolution of a set of states under the schroedinger
    equation for time-discrete control parameters.

    Arguments:
    complex_controls :: bool - This value determines if the control parameters
        are complex-valued. If some controls are real only or imaginary only
        while others are complex, real only and imaginary only controls
        can be simulated by taking the real or imaginary part of a complex control.
    control_count :: int - This value specifies how many control parameters
        should be supplied to the hamiltonian at one time.
    control_step_count :: int - This value is the number of positions
        in time where control parameters are fit. For example, if the
        evolution time is 10ns and the control step count is 2,
        control parameters will be fit at 0ns and 5ns. The control
        parameters at other times will be interpolated based on the ones
        fit at those values.
    costs :: iterable(qoc.models.cost.Cost) - This list specifies all
        the cost functions that the optimizer should evaluate. This list
        defines the criteria for an "optimal" control set.
    evolution_time :: float - This value specifies the duration of the
        system's evolution.
    hamiltonian :: (controls :: ndarray (control_count), time :: float)
                   -> hamiltonian :: ndarray (hilbert_size x hilbert_size)
        - This function provides the system's hamiltonian given a set
        of control parameters and a time value. This function must
        use operations registered with autograd. I.e. use
        autograd.numpy instead of numpy.
    initial_controls :: ndarray (control_step_count x control_count)
        - This array specifies the control parameters at each
        control step. These values will be used to determine the `controls`
        argument passed to the `hamiltonian` function at each time step for
        the first iteration of optimization.
    initial_states :: ndarray (state_count x hilbert_size x 1)
        - This array specifies the states that should be evolved under the
        specified system. These are the states at the beggining of the evolution.
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
    minimum_error :: float - This value is the threshold below which
        optimization will terminate.
    operation_policy :: qoc.models.operationpolicy.OperationPolicy
        - This value specifies which computation backend should be used
        to execute the program.
    optimizer :: class instance - This optimizer object defines the
        gradient-based procedure for minimizing the total contribution
        of all cost functions with respect to the control parameters.
    performance_policy :: qoc.models.performancepolicy.PerformancePolicy
        - This value determines which computing resources should be
        minimized during program execution.
    save_file_path :: str - This is the full path to the file where
        information about program execution will be stored.
        E.g. "./out/foo.h5"
    save_iteration_step :: int - This value determines how often qoc
    saves progress to the save file specified by `save_file_path`.
        This value is specified in units of system steps, of which
        there are `control_step_count` * `system_step_multiplier`.
        Set this value to 0 to disable saving.
    system_step_multiplier :: int - By default, qoc will integrate the
        ode at each time interval produced by dividing `evolution_time`
        into `control_step_count`intervals. Setting the
        `system_step_multiplier` value will change qoc's behavior to
        integrate the ode at each time interval produced by dividing
        `evolution_time`into `control_step_count` * `system_step_multiplier`
        intervals. This will result in higher accuracy, but more compute time.
        Additionaly, each cost function that requires evaluation at each
        time step will also be evaluated at `control_step_count`
        * `system_step_multiplier` time steps.

    Returns:
    result :: GrapeSchroedingerResult
    """
    # Initialize the controls.
    initial_controls, max_control_norms = initialize_controls(complex_controls,
                                                              control_count,
                                                              control_step_count,
                                                              evolution_time,
                                                              initial_controls,
                                                              max_control_norms,)
    # Construct the program state.
    pstate = GrapeSchroedingerDiscreteState(complex_controls, control_count,
                                            control_step_count, costs,
                                            evolution_time, hamiltonian,
                                            initial_controls,
                                            initial_states,
                                            interpolation_policy,
                                            iteration_count,
                                            log_iteration_step, magnus_policy,
                                            max_control_norms, minimum_error,
                                            operation_policy,
                                            optimizer, performance_policy,
                                            save_file_path, save_iteration_step,
                                            system_step_multiplier,)
    pstate.log_and_save_initial()

    # Switch on the GRAPE implementation method.
    if performance_policy == PerformancePolicy.TIME:
        result = _grape_schroedinger_discrete_time(pstate)
    else:
        result = _grape_schroedinger_discrete_memory(pstate)

    return result


### HELPER METHODS ###

def _esd_wrap(controls, pstate, reporter, result):
    """
    Do intermediary work between the optimizer feeding controls
    to _evaluate_schroedinger_discrete.

    Args:
    controls
    pstate
    reporter
    result

    Returns:
    total_error
    """
    # Convert the controls from optimizer format to cost function format.
    controls = slap_controls(pstate.complex_controls, controls,
                             pstate.controls_shape)
    clip_control_norms(pstate.max_control_norms, controls)

    # Evaluate the cost function.
    return _evaluate_schroedinger_discrete(controls, pstate, reporter)


def _esdj_wrap(controls, pstate, reporter, result):
    """
    Do intermediary work between the optimizer feeding controls to 
    the jacobian of _evaluate_schroedinger_discrete.

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
    clip_control_norms(pstate.max_control_norms, controls)

    # Evaluate the jacobian.
    total_error, grads = (ans_jacobian(_evaluate_schroedinger_discrete, 0)
                          (controls, pstate, reporter))
    # Autograd defines the derivative of a function of complex inputs as
    # df_dz = du_dx - i * du_dy for z = x + iy, f(z) = u(x, y) + iv(x, y).
    # For optimization, we care about df_dz = du_dx + i * du_dy.
    if pstate.complex_controls:
        grads = conjugate(grads)

    # The states need to be unwrapped from their autograd box.
    if isinstance(reporter.final_states, Box):
        final_states = reporter.final_states._value

    # Update best configuration.
    if total_error < result.best_total_error:
        result.best_controls = controls
        result.best_final_states = final_states
        result.best_iteration = reporter.iteration
        result.best_total_error = total_error
    
    # Save and log optimization progress.
    pstate.log_and_save(controls, total_error, grads, reporter.iteration,
                        final_states)
    reporter.iteration += 1

    # Convert the gradients from cost function to optimizer format.
    grads = strip_controls(pstate.complex_controls, grads)
    
    return grads


def _evaluate_schroedinger_discrete(controls, pstate, reporter):
    """
    Compute the value of the total cost function for one evolution.

    Args:
    controls :: ndarray - the control parameters
    pstate :: qoc.GrapeSchroedingerDiscreteState or qoc.EvolveSchroedingerDiscreteState
        - static objects
    reporter :: any - a reporter for mutable objects

    Returns:
    total_error :: ndarray - total error of the evolution
    """
    # Initialize local variables (heap -> stack).
    costs = pstate.costs
    dt = pstate.dt
    hamiltonian = pstate.hamiltonian
    final_control_step = pstate.final_control_step
    final_system_step = pstate.final_system_step
    interpolation_policy = pstate.interpolation_policy
    magnus_policy = pstate.magnus_policy
    operation_policy = pstate.operation_policy
    states = pstate.initial_states
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0
    
    # Compute the total error for this evolution.
    for system_step in range(final_system_step + 1):
        control_step, _ = divmod(system_step, system_step_multiplier)
        is_final_control_step = control_step == final_control_step
        is_final_system_step = system_step == final_system_step
        time = system_step * dt

        # Evolve the states.
        states = _evolve_step_schroedinger_discrete(dt, hamiltonian, states,
                                                    time,
                                                    control_sentinel=is_final_control_step,
                                                    control_step=control_step,
                                                    controls=controls,
                                                    interpolation_policy=interpolation_policy,
                                                    magnus_policy=magnus_policy,
                                                    operation_policy=operation_policy,)
        
        # Compute cost every time step.
        if is_final_system_step:
            for i, cost in enumerate(costs):
                error = cost.cost(controls, states, system_step)
                total_error = total_error + error
            #ENDFOR
            reporter.final_states = states
            reporter.total_error = total_error
        else:
            for i, step_cost in enumerate(step_costs):
                error = step_cost.cost(controls, states, system_step)
                total_error = total_error + error
            #ENDFOR
    #ENDFOR
    
    return total_error


_M4_T1 = 0.5 - np.divide(np.sqrt(3), 6)
_M4_T2 = 0.5 + np.divide(np.sqrt(3), 6)
_M6_T1 = 0.5 - np.divide(np.sqrt(15), 10)
_M6_T2 = 0.5
_M6_T3 = 0.5 + np.divide(np.sqrt(15), 10)

def _evolve_step_schroedinger_discrete(dt, hamiltonian, states, time,
                                       control_sentinel=False,
                                       control_step=0,
                                       controls=None,
                                       interpolation_policy=InterpolationPolicy.LINEAR,
                                       magnus_policy=MagnusPolicy.M2,
                                       operation_policy=OperationPolicy.CPU,):
    """
    Use the exponential series method via magnus expansion to evolve the state vectors
    to the next time step under the schroedinger equation for time-discrete controls.
    Magnus expansions are implemented using the methods described in
    https://arxiv.org/abs/1709.06483.

    Args:
    control_sentinel
    control_step
    controls
    dt
    hamiltonian
    interpolation_policy
    magnus_policy
    operation_policy
    states
    time
    
    Returns:
    states
    """
    controls_exist = not (controls is None)
    
    if magnus_policy == MagnusPolicy.M2:
        if controls_exist:
            c1 = controls[control_step]
        else:
            c1 = None
        a1 = -1j * hamiltonian(c1, time)
        magnus = magnus_m2(a1, dt, operation_policy=operation_policy)
    elif magnus_policy == MagnusPolicy.M4:
        t1 = time + dt * _M4_T1
        t2 = time + dt * _M4_T2
        if controls_exist:
            if control_sentinel:
                controls_left = controls[control_step - 1]
                controls_right = controls[control_step]
                time_left = time - dt
                time_right = time
            else:
                controls_left = controls[control_step]
                controls_right = controls[control_step + 1]
                time_left = time
                time_right = time + dt
            if interpolation_policy == InterpolationPolicy.LINEAR:
                c1 = interpolate_linear(time_left, time_right, t1,
                                        controls_left, controls_right)
                c2 = interpolate_linear(time_left, time_right, t2, controls_left,
                                        controls_right)
            else:
                raise ValueError("The interpolation policy {} is not implemented"
                                 "for this method.".format(interpolation_policy))
        else:
            c1 = c2 = None
        a1 = -1j * hamiltonian(c1, t1)
        a2 = -1j * hamiltonian(c2, t2)
        magnus = magnus_m4 (a1, a2, dt)
    elif magnus_policy == MagnusPolicy.M6:
        t1 = time + dt * _M6_T1
        t2 = time + dt * _M6_T2
        t3 = time + dt * _M6_T3
        if controls_exist:
            if control_sentinel:
                controls_left = controls[control_step - 1]
                controls_right = controls[control_step]
                time_left = time - dt
                time_right = time
            else:
                controls_left = controls[control_step]
                controls_right = controls[control_step + 1]
                time_left = time
                time_right = time + dt
            if interpolation_policy == InterpolationPolicy.LINEAR:
                c1 = interpolate_linear(time_left, time_right, t1,
                                        controls_left, controls_right,
                                        operation_policy=operation_policy)
                c2 = interpolate_linear(time_left, time_right, t2, controls_left,
                                        controls_right,
                                        operation_policy=operation_policy)
                c3 = interpolate_linear(time_left, time_right, t2, controls_left,
                                        controls_right,
                                        operation_policy=operation_policy)
            else:
                raise ValueError("The interpolation policy {} is not implemented"
                                 "for this method.".format(interpolation_policy))
        else:
            c1 = c2 = c3 = None
        a1 = -1j * hamiltonian(c1, t1)
        a2 = -1j * hamiltonian(c2, t2)
        a3 = -1j * hamiltonian(c3, t3)
        magnus = magnus_m6(a1, a2, a3, dt,
                           operation_policy=operation_policy)
    #ENDIF

    step_unitary = expm(magnus, operation_policy=operation_policy)
    states = matmuls(step_unitary, states,
                     operation_policy=operation_policy)

    return states


def _grape_schroedinger_discrete_time(pstate):
    """
    Perform GRAPE for the schroedinger equation with time discrete parameters.
    Use autograd to compute evolution gradients.

    Args:
    pstate :: qoc.models.GrapeSchroedingerDiscreteState - information required to
         perform the optimization

    Returns: 
    result :: qoc.models.GrapeResult - an object that tracks important information
        about the optimization
    """
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
    # Run the optimization.
    pstate.optimizer.run(_esd_wrap, pstate.iteration_count, initial_controls,
                         _esdj_wrap, args=(pstate, reporter, result))
    return result


### MODULE TESTS ###

_BIG = 10
_MAGNUS_POLICIES = (MagnusPolicy.M2, MagnusPolicy.M4, MagnusPolicy.M6,)

def _random_complex_matrix(matrix_size):
    """
    Generate a random, square, complex matrix of size `matrix_size`.
    """
    return (np.random.rand(matrix_size, matrix_size)
            + 1j * np.random.rand(matrix_size, matrix_size))


def _random_hermitian_matrix(matrix_size):
    """
    Generate a random, square, hermitian matrix of size `matrix_size`.
    """
    matrix = _random_complex_matrix(matrix_size)
    return np.divide(matrix + conjugate_transpose(matrix), 2)


def _test_evolve_schroedinger_discrete():
    """
    Run end-to-end test on the evolve_schroedinger_discrete
    function.
    """
    from qutip import mesolve, Qobj, Options
    
    from qoc.standard import (matrix_to_column_vector_list,
                              SIGMA_X, SIGMA_Y,)

    # Test that evolving states under a hamiltonian yields
    # a known result. Use e.q. 109 of 
    # https://arxiv.org/abs/1904.06560.
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    iswap_unitary = np.array(((1,   0,   0, 0),
                              (0,   0, -1j, 0),
                              (0, -1j,   0, 0),
                              (0,   0,   0, 1)))
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(SIGMA_X, SIGMA_X)
                                     + np.kron(SIGMA_Y, SIGMA_Y))
    hamiltonian = lambda controls, time: hamiltonian_matrix
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(iswap_unitary)
    evolution_time = np.divide(np.pi, 2)
    control_step_count = int(1e3)
    for magnus_policy in _MAGNUS_POLICIES:
        result = evolve_schroedinger_discrete(control_step_count, evolution_time,
                                              hamiltonian, initial_states,
                                              magnus_policy=magnus_policy)
        final_states = result.final_states
        assert(np.allclose(final_states, target_states))
    #ENDFOR
    # Note that qutip only gets the same result within 1e-5 error.
    tlist = np.linspace(0, evolution_time, control_step_count)
    c_ops = list()
    e_ops = list()
    hamiltonian_qutip = Qobj(hamiltonian_matrix)
    options = Options(nsteps=control_step_count)
    for i, initial_state in enumerate(initial_states):
        initial_state_qutip = Qobj(initial_state)
        result = mesolve(hamiltonian_qutip,
                         initial_state_qutip,
                         tlist, c_ops, e_ops,
                         options=options)
        final_state = result.states[-1].full()
        target_state = target_states[i]
        assert(np.allclose(final_state, target_state, atol=1e-5))
    #ENDFOR

    # Test that evolving states under a random hamiltonian yields
    # a result similar to qutip.
    hilbert_size = 4
    initial_state = np.divide(np.ones((hilbert_size, 1)),
                              np.sqrt(hilbert_size))
    initial_states = np.stack((initial_state,))
    initial_state_qutip = Qobj(initial_state)
    control_step_count = int(1e3)
    evolution_time = 1
    tlist = np.linspace(0, evolution_time, control_step_count)
    c_ops = e_ops = list()
    options = Options(nsteps=control_step_count)
    for _ in range(_BIG):
        hamiltonian_matrix = _random_hermitian_matrix(hilbert_size)
        hamiltonian = lambda controls, time: hamiltonian_matrix
        hamiltonian_qutip = Qobj(hamiltonian_matrix)
        result = mesolve(hamiltonian_qutip,
                         initial_state_qutip,
                         tlist, c_ops, e_ops,
                         options=options)
        final_state_qutip = result.states[-1].full()
        for magnus_policy in _MAGNUS_POLICIES:
            result = evolve_schroedinger_discrete(control_step_count,
                                                  evolution_time, hamiltonian,
                                                  initial_states,
                                                  magnus_policy=magnus_policy)
            final_state = result.final_states[0]
            assert(np.allclose(final_state, final_state_qutip, atol=1e-4))
        #ENDFOR
    #ENDFOR
        

def _test_grape_schroedinger_discrete():
    """
    Run end-to-end test on the grape_schroedinger_discrete function.

    NOTE: We mostly care about the tests for evolve_schroedinger_discrete.
    For grape_schroedinger_discrete we care that everything is being passed
    through functions properly, but autograd has a really solid testing
    suite and we trust that their gradients are being computed
    correctly.
    """
    from qoc.standard import (ForbidStates, SIGMA_X, SIGMA_Y,)
    
    # Test that parameters are clipped if they grow too large.
    hilbert_size = 4
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(SIGMA_X, SIGMA_X)
                                            + np.kron(SIGMA_Y, SIGMA_Y))
    hamiltonian = lambda controls, t: (controls[0] * hamiltonian_matrix)
    initial_states = np.array([[[0], [1], [0], [0]]])
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    control_count = 1
    evolution_time = 10
    control_step_count = 10
    max_norm = 1e-10
    max_control_norms = np.repeat(max_norm, control_count)
    costs = [ForbidStates(forbidden_states, control_step_count)]
    iteration_count = 100
    log_iteration_step = 0
    result = grape_schroedinger_discrete(control_count, control_step_count,
                                         costs, evolution_time,
                                         hamiltonian, initial_states,
                                         iteration_count=iteration_count,
                                         log_iteration_step=log_iteration_step,
                                         max_control_norms=max_control_norms)
    for i in range(result.best_controls.shape[1]):
        assert(np.less_equal(np.abs(result.best_controls[:,i]),
                             max_control_norms[i]).all())


def _test():
    """
    Run tests on the module.
    """
    _test_evolve_schroedinger_discrete() 
    _test_grape_schroedinger_discrete()


if __name__ == "__main__":
    _test()
