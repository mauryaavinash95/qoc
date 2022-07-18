"""
targetdensityinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved unitary and a target unitary.
"""

import jax
import jax.numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetUnitaryInfidelity(Cost):
    """
    This cost penalizes the infidelity of an evolved unitary
    and a target unitary.

    Fields:
    cost_multiplier
    unitary_count
    hilbert_size
    name
    requires_step_evaluation
    target_unitaries_dagger
    """
    name = "target_unitary_infidelity"
    requires_step_evaluation = False

    def __init__(self, target_unitaries, cost_multiplier=1.):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_unitaries
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.unitary_count = target_unitaries.shape[0]
        self.hilbert_size = target_unitaries.shape[1]
        self.target_unitaries_dagger = conjugate_transpose(target_unitaries)


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
        # NOTE: Autograd doesn't support vjps of anp.trace with axis arguments.
        # Nor does it support the vjp of anp.einsum(...ii->..., a).
        # Therefore, we must use a for loop to index the traces.
        # The following computations are equivalent to:
        # inner_products = (anp.trace(anp.matmul(self.target_densities_dagger, densities),
        #                             axis1=-1, axis2=-2) / self.hilbert_size)
        prods = np.matmul(self.target_unitaries_dagger, unitaries)
        fidelity_sum = 0
        for i, prod in enumerate(prods):
            inner_prod = np.trace(prod)
            fidelity = np.abs(inner_prod)
            fidelity_sum = fidelity_sum + fidelity
        fidelity_normalized = fidelity_sum / (self.unitary_count * self.hilbert_size)
        infidelity = 1 - fidelity_normalized

        return infidelity * self.cost_multiplier
