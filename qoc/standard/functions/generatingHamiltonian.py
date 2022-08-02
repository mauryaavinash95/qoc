#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:29:23 2022

@author: xian.wang
"""

# This is the code for generating the Hamiltonians of 2D Lattice Gauge Theory.
# It works for rectangular lattice only. The number of vertices along the two sides are required.
# Qubits sit at the sides.

import jax.numpy as jnp
import numpy as onp
import qutip as qt
from functools import reduce

def corner_terms(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( (length - 1) * width + (width - 1) * length )
    plaquette_count = int( (length - 1) * (width - 1) )
    corner_count = int( plaquette_count * 4 )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((corner_count, qubit_count), dtype=jnp.bool_)
    corner_idx = 0
    for plaquette_idx in range(0, plaquette_count):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length - 1 ) )
        first_qubit_idx = (length + (length - 1) ) * plaquette_row + plaquette_remainder
        flags[corner_idx, first_qubit_idx] = True
        flags[corner_idx, first_qubit_idx + (length - 1)] = True
        flags[corner_idx + 1, first_qubit_idx] = True
        flags[corner_idx + 1, first_qubit_idx + length] = True
        flags[corner_idx + 2, first_qubit_idx + (length - 1)] = True
        flags[corner_idx + 2, first_qubit_idx + length + (length - 1)] = True
        flags[corner_idx + 3, first_qubit_idx + length] = True
        flags[corner_idx + 3, first_qubit_idx + length + (length - 1)] = True
        corner_idx += 4
    
    assert(corner_idx == corner_count)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for corner_idx in range(0, corner_count):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[corner_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
    return hamiltonian

def plaquette_terms(length, width, 
                    S_plus = 0.5 * (jnp.array(qt.operators.sigmax().data.toarray()) + 1j * jnp.array(qt.operators.sigmay().data.toarray()) ),
                    S_minus = 0.5 * (jnp.array(qt.operators.sigmax().data.toarray()) - 1j * jnp.array(qt.operators.sigmay().data.toarray()) ) ):
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( (length - 1) * width + (width - 1) * length )
    plaquette_count = int( (length - 1) * (width - 1) )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((plaquette_count, qubit_count), dtype=jnp.int8)
    for plaquette_idx in range(0, plaquette_count):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length - 1 ) )
        first_qubit_idx = (length + (length - 1) ) * plaquette_row + plaquette_remainder
        flags[plaquette_idx, first_qubit_idx] = 1
        flags[plaquette_idx, first_qubit_idx + (length - 1)] = -1
        flags[plaquette_idx, first_qubit_idx + length] = -1
        flags[plaquette_idx, first_qubit_idx + length + (length - 1)] = 1
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for plaquette_idx in range(0, plaquette_count):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[plaquette_idx, qubit_idx] == 1:
                matrices.append(S_plus)
            elif flags[plaquette_idx, qubit_idx] == -1:
                matrices.append(S_minus)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[plaquette_idx, qubit_idx] == 1:
                matrices.append(S_minus)
            elif flags[plaquette_idx, qubit_idx] == -1:
                matrices.append(S_plus)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
    return hamiltonian