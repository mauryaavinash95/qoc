"""
autogutil.py - This module provides utilities for interfacing with autograd.
"""

import jax

def ans_jacobian(function, argnum):
    """
    Get the value and the jacobian of a function.
    This differential operator follows autograd's jacobian implementation:
    https://github.com/HIPS/autograd/blob/master/autograd/differential_operators.py

    Args:
    function :: any -> any - the function to differentiate
    argnum :: int - the argument number to differentiate with respect to

    Returns:
    ans_jacobian any -> tuple(any :: any, jacobian :: ndarray) - a function
        that returns the value of `function` and the jacobian
        of `function` evaluated at a given argument of `function`
    """
    value_and_grad = jax.value_and_grad(function, argnum)
    return value_and_grad
