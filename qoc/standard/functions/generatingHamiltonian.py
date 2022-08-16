#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:29:23 2022

@author: xian.wang
"""

# This is the code for generating the Hamiltonians of 2D Lattice Gauge Theory.
# See the following papers for details.
# (1) Marcos, D., Widmer, P., Rico, E., Hafezi, M., Rabl, P., Wiese, U. J., & Zoller, P. (2014). 
# Two-dimensional lattice gauge theories with superconducting quantum circuits. 
# Annals of physics, 351, 634-654.
# (2) Kairys, P., & Humble, T. S. (2021). 
# Parametrized Hamiltonian simulation using quantum optimal control. 
# Physical Review A, 104(4), 042602.

# plaquette_terms: sum of (S_+^(i) S_-^(i+1) S_+^(i+2) S_-^(i+3) ) + (S_-^(i) S_+^(i+1) S_-^(i+2) S_+^(i+3) ) over all plaquettes
# corner_terms: sum of (S^(i) S^(i+1) ) over all corners
# single_qubit_terms: sum of S^(i) over all qubits indexed i
# single_qubit_terms_custom: sum of S^(i) over customized qubits

# The first 4 functions: The lattice is a rectangular defined by length and width, qubits sit on edges
# The next 4 functions: The lattice is a rectangular defined by length and width, qubits sit on vertices
# The next 4 functions: The lattice has zigzag boundaries as defined in Fig. 4 of ref (1)

import jax.numpy as jnp
import numpy as onp
import qutip as qt
from functools import reduce

def corner_terms(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the edges.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    # Plaquettes are defined as in Fig. 4 of the 2D LGT paper. Boundaries are smooth
    
    qubit_count = int( (length - 1) * width + (width - 1) * length )
    plaquette_count = int( (length - 1) * (width - 1) )
    corner_count = int( plaquette_count * 4 )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((corner_count, qubit_count), dtype=jnp.bool_)
    corner_idx = 0
    for plaquette_idx in range(0, plaquette_count):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length - 1 ) )
        first_qubit_idx = (length + (length - 1) ) * plaquette_row + plaquette_remainder
        flags = flags.at[corner_idx, first_qubit_idx].set(True)
        flags = flags.at[corner_idx, first_qubit_idx + (length - 1)].set(True)
        flags = flags.at[corner_idx + 1, first_qubit_idx].set(True)
        flags = flags.at[corner_idx + 1, first_qubit_idx + length].set(True)
        flags = flags.at[corner_idx + 2, first_qubit_idx + (length - 1)].set(True)
        flags = flags.at[corner_idx + 2, first_qubit_idx + length + (length - 1)].set(True)
        flags = flags.at[corner_idx + 3, first_qubit_idx + length].set(True)
        flags = flags.at[corner_idx + 3, first_qubit_idx + length + (length - 1)].set(True)
        corner_idx += 4
    print('corner_terms_flags', flags)
    
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
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the edges.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( (length - 1) * width + (width - 1) * length )
    plaquette_count = int( (length - 1) * (width - 1) )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((plaquette_count, qubit_count), dtype=jnp.int8)
    for plaquette_idx in range(0, plaquette_count):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length - 1 ) )
        first_qubit_idx = (length + (length - 1) ) * plaquette_row + plaquette_remainder
        flags = flags.at[plaquette_idx, first_qubit_idx].set(1)
        flags = flags.at[plaquette_idx, first_qubit_idx + (length - 1)].set(-1)
        flags = flags.at[plaquette_idx, first_qubit_idx + length].set(-1)
        flags = flags.at[plaquette_idx, first_qubit_idx + length + (length - 1)].set(1)
    print('plaquette_terms_flags', flags)
    
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

def single_qubit_terms(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the edges.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( (length - 1) * width + (width - 1) * length )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((qubit_count, qubit_count), dtype=jnp.bool_)
    for qubit_idx in range(0, qubit_count):
        flags = flags.at[qubit_idx, qubit_idx].set(True)
    print('single_qubit_flags', flags)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for qubit_idx in range(0, qubit_count):        
        matrices = []
        for qubit_idx2 in range(0, qubit_count):
            if flags[qubit_idx, qubit_idx2] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
    return hamiltonian

def single_qubit_terms_custom(length, width, control_count, control_qubit_list, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the edges.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( (length - 1) * width + (width - 1) * length )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((control_count, qubit_count), dtype=jnp.bool_)
    for control_idx in range(0, control_count):
        qubit_idx = control_qubit_list[control_idx]
        flags = flags.at[control_idx, qubit_idx].set(True)
    print('single_qubit_flags', flags)
    
    hamiltonian = jnp.zeros((control_count, hilbert_size, hilbert_size), dtype=jnp.complex128)
    for control_idx in range(0, control_count):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[control_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        #hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        hamiltonian = hamiltonian.at[control_idx].set( reduce(jnp.kron, matrices) )
        
    return hamiltonian

def corner_terms_2(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( length * width )
    edge_count = int( (length - 1) * width + (width - 1) * length )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((edge_count, qubit_count), dtype=jnp.bool_)
    edge_idx = 0
    for qubit_idx in range(0, qubit_count):
        (qubit_row, qubit_remainder) = onp.divmod(qubit_idx, length)
        if qubit_remainder == int(length - 1) and qubit_row < int(width - 1):
            flags = flags.at[edge_idx, qubit_idx].set(True)
            flags = flags.at[edge_idx, qubit_idx + length].set(True)
            edge_idx += 1
        elif qubit_remainder < int(length - 1) and qubit_row < int(width - 1):
            flags = flags.at[edge_idx, qubit_idx].set(True)
            flags = flags.at[edge_idx, qubit_idx + 1].set(True)
            flags = flags.at[edge_idx + 1, qubit_idx].set(True)
            flags = flags.at[edge_idx + 1, qubit_idx + length].set(True)
            edge_idx += 2
        elif qubit_remainder < int(length - 1) and qubit_row == int(width - 1):
            flags = flags.at[edge_idx, qubit_idx].set(True)
            flags = flags.at[edge_idx, qubit_idx + 1].set(True)
            edge_idx += 1
    print('corner_terms_flags', flags)
    
    assert(edge_idx == edge_count)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for edge_idx in range(0, edge_count):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[edge_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
    return hamiltonian

def plaquette_terms_2(length, width, 
                      S_plus = 0.5 * (jnp.array(qt.operators.sigmax().data.toarray()) + 1j * jnp.array(qt.operators.sigmay().data.toarray()) ),
                      S_minus = 0.5 * (jnp.array(qt.operators.sigmax().data.toarray()) - 1j * jnp.array(qt.operators.sigmay().data.toarray()) ) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( length * width )
    plaquette_count = int( (length - 1) * (width - 1) )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((plaquette_count, qubit_count), dtype=jnp.int8)
    for plaquette_idx in range(0, plaquette_count):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length - 1 ) )
        first_qubit_idx = length * plaquette_row + plaquette_remainder
        flags = flags.at[plaquette_idx, first_qubit_idx].set(1)
        flags = flags.at[plaquette_idx, first_qubit_idx + 1].set(-1)
        flags = flags.at[plaquette_idx, first_qubit_idx + length].set(-1)
        flags = flags.at[plaquette_idx, first_qubit_idx + 1 + length].set(1)
    print('plaquette_terms_flags', flags)
    
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

def single_qubit_terms_2(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( length * width )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((qubit_count, qubit_count), dtype=jnp.bool_)
    for qubit_idx in range(0, qubit_count):
        flags = flags.at[qubit_idx, qubit_idx].set(True)
    print('single_qubit_flags', flags)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for qubit_idx in range(0, qubit_count):        
        matrices = []
        for qubit_idx2 in range(0, qubit_count):
            if flags[qubit_idx, qubit_idx2] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
    return hamiltonian

def single_qubit_terms_custom_2(length, width, control_count, control_qubit_list, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    qubit_count = int( length * width )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((control_count, qubit_count), dtype=jnp.bool_)
    for control_idx in range(0, control_count):
        qubit_idx = control_qubit_list[control_idx]
        flags = flags.at[control_idx, qubit_idx].set(True)
    print('single_qubit_flags', flags)
    
    hamiltonian = jnp.zeros((control_count, hilbert_size, hilbert_size), dtype=jnp.complex128)
    for control_idx in range(0, control_count):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[control_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        #hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        hamiltonian = hamiltonian.at[control_idx].set( reduce(jnp.kron, matrices) )
        
    return hamiltonian

def corner_terms_3(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # For figure 4 in the 2D Lattice Gauge Theories paper.
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side. It must be even number
    # width is the number of vertices along the other side. It must be even number
    
    assert(length % 2 == 0)
    assert(width % 2 == 0)
    
    qubit_count = int( length * width )
    plaquette_count_1 = int( (length / 2) * (width / 2) )
    plaquette_count_2 = int( (length / 2 - 1) * (width / 2 - 1) )
    corner_count_1 = int( plaquette_count_1 * 4 )
    corner_count_2 = int( plaquette_count_2 * 4 )
    hilbert_size = 2 ** qubit_count
    flags_1 = jnp.zeros((corner_count_1, qubit_count), dtype=jnp.bool_)
    flags_2 = jnp.zeros((corner_count_2, qubit_count), dtype=jnp.bool_)
    corner_idx = 0
    for plaquette_idx in range(0, plaquette_count_1):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length / 2 ) )
        first_qubit_idx = (length * 2) * plaquette_row + 2 * plaquette_remainder
        flags_1 = flags_1.at[corner_idx, first_qubit_idx].set(True)
        flags_1 = flags_1.at[corner_idx, first_qubit_idx + 1].set(True)
        flags_1 = flags_1.at[corner_idx + 1, first_qubit_idx].set(True)
        flags_1 = flags_1.at[corner_idx + 1, first_qubit_idx + length].set(True)
        flags_1 = flags_1.at[corner_idx + 2, first_qubit_idx + 1].set(True)
        flags_1 = flags_1.at[corner_idx + 2, first_qubit_idx + 1 + length].set(True)
        flags_1 = flags_1.at[corner_idx + 3, first_qubit_idx + length].set(True)
        flags_1 = flags_1.at[corner_idx + 3, first_qubit_idx + 1 + length].set(True)
        corner_idx += 4
    
    assert(corner_idx == corner_count_1)
    
    corner_idx = 0
    for plaquette_idx in range(0, plaquette_count_2):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length / 2 - 1 ) )
        first_qubit_idx = length + 1 + (length * 2) * plaquette_row + 2 * plaquette_remainder
        flags_2 = flags_2.at[corner_idx, first_qubit_idx].set(True)
        flags_2 = flags_2.at[corner_idx, first_qubit_idx + 1].set(True)
        flags_2 = flags_2.at[corner_idx + 1, first_qubit_idx].set(True)
        flags_2 = flags_2.at[corner_idx + 1, first_qubit_idx + length].set(True)
        flags_2 = flags_2.at[corner_idx + 2, first_qubit_idx + 1].set(True)
        flags_2 = flags_2.at[corner_idx + 2, first_qubit_idx + 1 + length].set(True)
        flags_2 = flags_2.at[corner_idx + 3, first_qubit_idx + length].set(True)
        flags_2 = flags_2.at[corner_idx + 3, first_qubit_idx + 1 + length].set(True)
        corner_idx += 4
    
    assert(corner_idx == corner_count_2)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for corner_idx in range(0, corner_count_1):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags_1[corner_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
    
    for corner_idx in range(0, corner_count_2):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags_2[corner_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
    
    return hamiltonian

def plaquette_terms_3(length, width, 
                      S_plus = 0.5 * (jnp.array(qt.operators.sigmax().data.toarray()) + 1j * jnp.array(qt.operators.sigmay().data.toarray()) ),
                      S_minus = 0.5 * (jnp.array(qt.operators.sigmax().data.toarray()) - 1j * jnp.array(qt.operators.sigmay().data.toarray()) ) ):
    # For figure 4 in the 2D Lattice Gauge Theories paper.
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side. It must be even number
    # width is the number of vertices along the other side. It must be even number
    
    assert(length % 2 == 0)
    assert(width % 2 == 0)
    
    qubit_count = int( length * width )
    plaquette_count_1 = int( (length / 2) * (width / 2) )
    plaquette_count_2 = int( (length / 2 - 1) * (width / 2 - 1) )
    hilbert_size = 2 ** qubit_count
    flags_1 = jnp.zeros((plaquette_count_1, qubit_count), dtype=jnp.int8)
    flags_2 = jnp.zeros((plaquette_count_2, qubit_count), dtype=jnp.int8)
    for plaquette_idx in range(0, plaquette_count_1):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length / 2 ) )
        first_qubit_idx = (length * 2) * plaquette_row + 2 * plaquette_remainder
        flags_1 = flags_1.at[plaquette_idx, first_qubit_idx].set(1)
        flags_1 = flags_1.at[plaquette_idx, first_qubit_idx + 1].set(-1)
        flags_1 = flags_1.at[plaquette_idx, first_qubit_idx + length].set(-1)
        flags_1 = flags_1.at[plaquette_idx, first_qubit_idx + 1 + length].set(1)
    
    for plaquette_idx in range(0, plaquette_count_2):
        (plaquette_row, plaquette_remainder) = onp.divmod(plaquette_idx, ( length / 2 - 1 ) )
        first_qubit_idx = length + 1 + (length * 2) * plaquette_row + 2 * plaquette_remainder
        flags_2 = flags_2.at[plaquette_idx, first_qubit_idx].set(1)
        flags_2 = flags_2.at[plaquette_idx, first_qubit_idx + 1].set(-1)
        flags_2 = flags_2.at[plaquette_idx, first_qubit_idx + length].set(-1)
        flags_2 = flags_2.at[plaquette_idx, first_qubit_idx + 1 + length].set(1)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for plaquette_idx in range(0, plaquette_count_1):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags_1[plaquette_idx, qubit_idx] == 1:
                matrices.append(S_plus)
            elif flags_1[plaquette_idx, qubit_idx] == -1:
                matrices.append(S_minus)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags_1[plaquette_idx, qubit_idx] == 1:
                matrices.append(S_minus)
            elif flags_1[plaquette_idx, qubit_idx] == -1:
                matrices.append(S_plus)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
    
    for plaquette_idx in range(0, plaquette_count_2):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags_2[plaquette_idx, qubit_idx] == 1:
                matrices.append(S_plus)
            elif flags_2[plaquette_idx, qubit_idx] == -1:
                matrices.append(S_minus)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags_2[plaquette_idx, qubit_idx] == 1:
                matrices.append(S_minus)
            elif flags_2[plaquette_idx, qubit_idx] == -1:
                matrices.append(S_plus)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
    
    return hamiltonian

def single_qubit_terms_3(length, width, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    assert(length % 2 == 0)
    assert(width % 2 == 0)
    
    qubit_count = int( length * width )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((qubit_count, qubit_count), dtype=jnp.bool_)
    for qubit_idx in range(0, qubit_count):
        flags = flags.at[qubit_idx, qubit_idx].set(True)
    
    hamiltonian = jnp.zeros((hilbert_size, hilbert_size), dtype=jnp.complex128)
    for qubit_idx in range(0, qubit_count):        
        matrices = []
        for qubit_idx2 in range(0, qubit_count):
            if flags[qubit_idx, qubit_idx2] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        
    return hamiltonian

def single_qubit_terms_custom_3(length, width, control_count, control_qubit_list, sigma = jnp.array(qt.operators.sigmaz().data.toarray()) ):
    # It works for rectangular lattice only. The number of vertices along the two sides are required.
    # Qubits sit at the vertices.
    # length is the number of vertices along one side
    # width is the number of vertices along the other side
    
    assert(length % 2 == 0)
    assert(width % 2 == 0)
    
    qubit_count = int( length * width )
    hilbert_size = 2 ** qubit_count
    flags = jnp.zeros((control_count, qubit_count), dtype=jnp.bool_)
    for control_idx in range(0, control_count):
        qubit_idx = control_qubit_list[control_idx]
        flags = flags.at[control_idx, qubit_idx].set(True)
    print('single_qubit_flags', flags)
    
    hamiltonian = jnp.zeros((control_count, hilbert_size, hilbert_size), dtype=jnp.complex128)
    for control_idx in range(0, control_count):        
        matrices = []
        for qubit_idx in range(0, qubit_count):
            if flags[control_idx, qubit_idx] == True:
                matrices.append(sigma)
            else:
                matrices.append(jnp.identity(2,dtype=jnp.complex128) )
        #hamiltonian = hamiltonian + reduce(jnp.kron, matrices)
        hamiltonian = hamiltonian.at[control_idx].set( reduce(jnp.kron, matrices) )
        
    return hamiltonian



