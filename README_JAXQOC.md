# JAXQOC
Implementing QOC with JAX.

This README document records the customized coding for the lattice gauge theory project.

### Installation ###
You can install QOC locally via pip.
```
git clone https://github.com/SchusterLab/qoc.git
cd qoc
pip install -e .
```

### Updates to the QOC code for LGT project ###
1) jax.ops.index_update was removed in JAX. Replace with x.at[idx].set(y)
2) Define target unitary infidelity in qoc/standard/costs/targetunitaryinfidelity.py and qoc/standard/costs/targetunitaryinfidelitytime.py
3) Debug: call self in qoc/models/cost.py
4) Coded for unitary cost in qoc/core/schroedingerdiscrete_unitary.py
5) Coded for unitary cost in qoc/models/schroedingermodels_unitary.py
6) Debug: Replace cost.cost == CostClass.cost with isinstance(cost, CostClass) in qoc/core/schroedingerdiscrete_unitary.py
7) Debug: replace jnp.random.normal with onp.random.normal when initializing controls with white noise in qoc/core/common.py
8) Debug: replace np with jnp in qoc/models/schroedingermodels.py
9) Update definition of infidelity: fidelity_sum = fidelity_sum + fidelity**2
    fidelity_normalized = fidelity_sum / (self.density_count * (self.hilbert_size ** 2) )
10) Update plots when controls are real in qoc/standard/plot.py
11) Define plotting infidelity vs. iteration in qoc/standard/plot.py
12) Define hamiltonians with lattice size, qoc/standard/functions/generatingHamiltonian.py
13) Load all new functions in __init__.py
14) All qoc_HamiltonianSimulation launchers are deprecated. Use qoc_LGT.py instead
15) All files and functions ending with CONTROL_list are deprecated
   
### Contact ###
[Xian Wang](mailto:xwang056@ucr.edu) or (mailto:wangthehero@gmail.com)
