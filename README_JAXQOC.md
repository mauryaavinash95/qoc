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

### Parameters for LGT project ###
1) # 4 qubits:
   DEVICE_HAMILTONIAN = ( 0.5 * single_qubit_terms_2(2, 2, sigma = np.array(qt.operators.sigmaz().data.toarray()) )  
                   + 0.25 * corner_terms_2(2, 2, sigma = np.array(qt.operators.sigmax().data.toarray()) )
                   + 0.25 * corner_terms_2(2, 2, sigma = np.array(qt.operators.sigmay().data.toarray()) )  )
   # Control all the 4 qubits:
   CONTROL = 0.5 * single_qubit_terms_custom_2(2, 2, 4, [0, 1, 2, 3, ], sigma = np.array(qt.operators.sigmax().data.toarray()) )
   coupling_J = 1.0 
   coupling_V = 1.0
   MODEL_HAMILTONIAN = - coupling_J * plaquette_terms_2(2, 2 ) + coupling_V * corner_terms_2(2, 2, sigma = np.array(qt.operators.sigmaz().data.toarray()) )
   Time_system = 0.01 #ns
   TARGET_UNITARIES = np.array([jax.scipy.linalg.expm(-1j * Time_system * MODEL_HAMILTONIAN)], dtype=np.complex128)
   # There are 4 qubits, 4 controls, so:
   QUBIT_COUNT = 4
   CONTROL_COUNT = 4
   def hamiltonian(controls, time):
       return (DEVICE_HAMILTONIAN
            + controls[0] * CONTROL[0]
            + controls[1] * CONTROL[1]
            + controls[2] * CONTROL[2]
            + controls[3] * CONTROL[3] )
   # EVOLUTION_TIME = 100 ns suffices:
   CONTROL_EVAL_COUNT = int(500)
   EVOLUTION_TIME = 100

2) # 6 qubits:
   
### Contact ###
[Xian Wang](mailto:xwang056@ucr.edu) or (mailto:wangthehero@gmail.com)
