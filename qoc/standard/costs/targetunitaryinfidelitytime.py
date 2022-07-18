"""
targetdensityinfidelitytime.py - This module defines a cost function
that penalizes the infidelity of evolved unitaries and their
respective target unitaries at each cost evaluation step.
"""

import jax
import jax.numpy as jnp

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetUnitaryInfidelityTime(Cost):
    """
    This class penalizes the infidelity of evolved unitaries
    and their respective target unitaries at each cost evaluation step.
    The intended result is that a lower infidelity is
    achieved earlier in the system evolution.

    Fields:
    cost_eval_count
    cost_multiplier
    unitary_count
    hilbert_size
    name
    requires_step_evaluation
    target_unitaries_dagger
    """
    name = "target_unitary_infidelity_time"
    requires_step_evaluation = False

    def __init__(self, system_eval_count, target_unitaries,
                 cost_eval_step=1, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_unitaries
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.cost_eval_count, _ = jnp.divmod(system_eval_count - 1, cost_eval_step)
        self.unitary_count = target_unitaries.shape[0]
        self.hilbert_size = target_unitaries.shape[1]
        self.target_unitaries_dagger = conjugate_transpose(jnp.stack(target_unitaries))


    def cost(self, controls, unitaries, sytem_eval_step):
        """
        Compute the penalty.

        Arguments:
        controls
        unitaries
        system_eval_step

        Returns:
        cost
        """
        # The cost is the infidelity of each evolved unitary and its target unitary.
        # NOTE: Autograd doesn't support vjps of np.trace with axis arguments.
        # Nor does it support the vjp of np.einsum(...ii->..., a).
        # Therefore, we must use a for loop to index the traces.
        # The following computations are equivalent to:
        # inner_products = (np.trace(np.matmul(self.target_densities_dagger, densities),
        #                             axis1=-1, axis2=-2) / self.hilbert_size)
        prods = jnp.matmul(self.target_unitaries_dagger, unitaries)
        fidelity_sum = 0
        for i, prod in enumerate(prods):
            inner_prod = jnp.trace(prod)
            fidelity = jnp.abs(inner_prod)
            fidelity_sum = fidelity_sum + fidelity
        fidelity_normalized = fidelity_sum / (self.unitary_count * self.hilbert_size)
        infidelity = 1 - fidelity_normalized
        cost_normalized = infidelity / self.cost_eval_count

        return cost_normalized * self.cost_multiplier
