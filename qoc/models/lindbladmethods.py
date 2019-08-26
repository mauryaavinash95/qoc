"""
lindbladmethods.py - a module for lindblad math

NOTE:
This module's methods are tested by the
tests on the evolve_lindblad_discrete
method in qoc.core.gld.py.
"""

from qoc.models.mathmethods import interpolate_linear
from qoc.models.interpolationpolicy import (InterpolationPolicy,)
from qoc.models.operationpolicy import (OperationPolicy,)
from qoc.standard import (commutator, conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def evolve_step_lindblad_discrete(densities, dt,
                                  time, control_sentinel=False,
                                  control_step=0, 
                                  controls=None,
                                  hamiltonian=None,
                                  interpolation_policy=InterpolationPolicy.LINEAR,
                                  lindblad_data=None,
                                  operation_policy=OperationPolicy.CPU,):
    """
    Use Runge-Kutta 4th order to evolve the density matrices to the next time step
    under the lindblad master equation. This RK4 implementation follows the definition:
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods. Runge-Kutta was
    chosen over matrix exponential by the suggestion of: https://arxiv.org/abs/1609.03170.
    

    NOTATION:
     - t is time, c is controls, h is hamiltonian, g is dissipation constants,
       l is lindblad operators, k are the runge-kutta increments

    Args:
    control_sentinel :: bool - set to True if this is the final control step,
        in which case control interpolation is performed on the last two
        control sets in the controls array
    control_step :: int - the index into the control array at which control
        interpolation should be performed
    controls :: ndarray - the controls that should be provided to the
        hamiltonian for the evolution    
    densities :: ndarray - the probability density matrices to evolve
    dt :: float - the time step
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_data :: (time :: float) -> (dissipartors :: ndarray, operators :: ndarray)
        - a function to generate the dissipation constants and lindblad operators
          for a given time
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    time :: float - the current evolution time

    Returns:
    densities :: ndarray - the densities evolved to `time + dt`
    """
    t1 = time
    t2 = time + 0.5 * dt
    t3 = t2
    t4 = time + dt
    if controls is None:
        c1 = c2 = c3 = c4 = None
    else:
        if control_sentinel:
            control_left = controls[control_step - 1]
            control_right = controls[control_step]
        else:
            control_left = controls[control_step]
            control_right = controls[control_step + 1]
        c1 = control_left
        if interpolation_policy == InterpolationPolicy.LINEAR:
            c2 = interpolate_linear(t1, t4, t2, control_left, control_right)
        else:
            raise ValueError("The interpolation policy {} is not "
                             "implemented for this method."
                             "".format(interpolation_policy))
        c3 = c2
        c4 = control_right
    #ENDIF
    if hamiltonian is None:
        h1 = h2 = h3 = h4 = None
    else:
        h1 = hamiltonian(c1, t1)
        h2 = hamiltonian(c2, t2)
        h3 = h2
        h4 = hamiltonian(c4, t4)
    if lindblad_data is None:
        (g1, l1) =  (g2, l2) = (g3, l3) = (g4, l4) = (None, None)
    else:
        g1, l1 = lindblad_data(t1)
        g2, l2 = lindblad_data(t2)
        g3, l3 = lindblad_data(t3)
        g4, l4 = lindblad_data(t4)
    #ENDIF
    k1 = dt * get_lindbladian(densities, g1, h1, l1,
                              operation_policy=operation_policy)
    k2 = dt * get_lindbladian(densities + 0.5 * k1, g2, h2, l2,
                              operation_policy=operation_policy)
    k3 = dt * get_lindbladian(densities + 0.5 * k2, g3, h3, l3,
                              operation_policy=operation_policy)
    k4 = dt * get_lindbladian(densities + k3, g4, h4, l4,
                              operation_policy=operation_policy)

    densities = densities + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return densities


def get_lindbladian(densities, dissipators=None, hamiltonian=None,
                    operators=None,
                    operation_policy=OperationPolicy.CPU):
    """
    Compute the action of the lindblad operator on a single (set of)
    density matrix (matrices). This implementation uses the definiton:
    https://en.wikipedia.org/wiki/Lindbladian.

    Args:
    densities :: ndarray - the probability density matrices
    dissipators :: ndarray - the lindblad dissipators
    hamiltonian :: ndarray
    operators :: ndarray - the lindblad operators
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.

    Returns:
    lindbladian :: ndarray - the lindbladian operator acting on the densities
    """
    if not (hamiltonian is None):
        lindbladian = -1j * commutator(hamiltonian, densities,
                                       operation_policy=operation_policy)
    else:
        lindbladian = 0
    if ((not (operators is None))
      and (not (dissipators is None))):
        operators_dagger = conjugate_transpose(operators,
                                               operation_policy=operation_policy)
        operators_product = matmuls(operators_dagger, operators,
                                    operation_policy=operation_policy)
        for i, operator in enumerate(operators):
            dissipator = dissipators[i]
            operator_dagger = operators_dagger[i]
            operator_product = operators_product[i]
            lindbladian = (lindbladian
                           + (dissipator
                              * (matmuls(operator, densities, operator_dagger,
                                         operation_policy=operation_policy)
                                 - 0.5 * matmuls(operator_product, densities,
                                                 operation_policy=operation_policy)
                                 - 0.5 * matmuls(densities, operator_product,
                                                 operation_policy=operation_policy))))
        #ENDFOR
    #ENDIF
    return lindbladian


### MODULE TESTS ###

def _test():
    import numpy as np
    
    # Test get_lindbladian on a hand verified solution.
    p = np.array(((1, 1), (1, 1)))
    ps = np.stack((p,))
    h = np.array(((0, 1), (1, 0)))
    g = 1
    gs = np.array((1,))
    l = np.array(((1, 0), (0, 0)))
    ls = np.stack((l,))
    lindbladian = get_lindbladian(p, gs, h, ls)
    expected_lindbladian = np.array(((0, -0.5),
                                     (-0.5, 0)))
    assert(np.allclose(lindbladian, expected_lindbladian))

if __name__ == "__main__":
    _test()