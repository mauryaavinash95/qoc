"""
expm.py - a module for all things e^M
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as la

@jax.custom_transforms
def expm(A):
  return la.expm(A)

def _expm_vjp_f(matrix,g):
    _,dfinal_dmatrix = la.expm_frechet(matrix,g)
    return dfinal_dmatrix
    
jax.defvjp(expm, lambda g,exp_matrix, matrix: _expm_vjp_f(matrix, g))
