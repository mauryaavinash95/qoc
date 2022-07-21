"""
common.py - This module defines methods that are used by
multiple core functionalities.
"""

import jax
import jax.numpy as jnp
import numpy as onp

def clip_control_norms(controls, max_control_norms):
    """
    Me: I need the entry-wise norms of the column entries of my
        control array to each be scaled to a fixed
        maximum norm if they exceed that norm
    Barber: Say no more fam

    Arguments:
    controls
    max_control_norms

    Returns: None
    """
    for i, max_control_norm in enumerate(max_control_norms):
        control = controls[:, i]
        control_norm = jnp.abs(control)
        offending_indices = onp.nonzero(onp.less(max_control_norm, control_norm))
        bool_mcn_lt_cn = jnp.less(max_control_norm, control_norm)
        offending_control_points = control * bool_mcn_lt_cn
        resolved_control_points = ((offending_control_points / control_norm)
                                   * max_control_norm)
        resolved_control_points1 = jnp.where(resolved_control_points!=0.0, resolved_control_points, controls[:,i])
        controls = controls.at[:,i].set(resolved_control_points1)
        #controls = jax.ops.index_update(controls, jax.ops.index[:,i], resolved_control_points1)
        # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
    #ENDFOR
    return controls


def gen_controls_cos(complex_controls, control_count, control_eval_count,
                     evolution_time, max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a cosine function.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    max_control_norms
    
    periods

    Returns:
    controls
    """
    period = jnp.divide(control_eval_count, periods)
    b = jnp.divide(2 * jnp.pi, period)
    controls = jnp.zeros((control_eval_count, control_count))
    
    # Create a wave for each control over all time
    # and add it to the controls.
    for i in range(control_count):
        # Generate a cosine wave about y=0 with amplitude
        # half of the max.
        max_norm = max_control_norms[i]
        _controls = (jnp.divide(max_norm, 2)
                   * jnp.cos(b * jnp.arange(control_eval_count)))
        # Replace all controls that have zero value
        # with small values.
        small_norm = max_norm * 1e-1
        _controls = jnp.where(_controls, _controls, small_norm)
        controls[:, i] = _controls
    #ENDFOR

    # Mimic the cosine fit for the imaginary parts and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / jnp.sqrt(2)

    return controls

def gen_controls_white(complex_controls, control_count, control_eval_count,
                      evolution_time, max_control_norms, periods=10.):
    """
    Create a discrete control set of random white noise.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    max_control_norms
    
    periods
    
    Returns:
    controls
    """
    controls = jnp.zeros((control_eval_count, control_count))

    # Make each control a random distribution of white noise.
    for i in range(control_count):
        max_norm = max_control_norms[i]
        stddev = max_norm/5.0
        control = onp.random.normal(0, stddev, control_eval_count)
        controls = controls.at[:, i].set(control)
    #ENDFOR

    # Mimic the white noise for the imaginary parts, and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / jnp.sqrt(2)

    return controls

def gen_controls_flat(complex_controls, control_count, control_eval_count,
                      evolution_time, max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a flat line with small amplitude.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    max_control_norms
    
    periods
    
    Returns:
    controls
    """
    controls = jnp.zeros((control_eval_count, control_count))

    # Make each control a flat line for all time.
    for i in range(control_count):
        max_norm = max_control_norms[i]
        small_norm = max_norm * 1e-1
        control = jnp.repeat(small_norm, control_eval_count)
        controls = controls.at[:,i].set(control)
        #controls = jax.ops.index_update(controls, jax.ops.index[:,i], control)
        # jax.ops.index_update was removed. Replace with x.at[idx].set(y)
    #ENDFOR

    # Mimic the flat line for the imaginary parts, and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / jnp.sqrt(2)

    return controls


_NORM_TOLERANCE = 1e-10
def initialize_controls(complex_controls,
                        control_count,
                        control_eval_count, evolution_time,
                        initial_controls, max_control_norms):
    """
    Sanitize `initial_controls` with `max_control_norms`.
    Generate both if either was not specified.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    initial_controls
    max_control_norms

    Returns:
    controls
    max_control_norms
    """
    if max_control_norms is None:
        max_control_norms = jnp.ones(control_count)
        
    if initial_controls is None:
        controls = gen_controls_flat(complex_controls, control_count, control_eval_count,
                                     evolution_time, max_control_norms)
    else:
        # Check that the user-specified controls match the specified data type.
        if complex_controls:
            if not jnp.iscomplexobj(initial_controls):
                raise ValueError("The program expected that the initial_controls specified by "
                                 "the user conformed to complex_controls, but "
                                 "the program found that the initial_controls were not complex "
                                 "and complex_controls was set to True.")
        else:
            if jnp.iscomplexobj(initial_controls):
                raise ValueError("The program expected that the initial_controls specified by "
                                 "the user conformed to complex_controls, but "
                                 "the program found that the initial_controls were complex "
                                 "and complex_controls was set to False.")
        
        # Check that the user-specified controls conform to max_control_norms.
        for control_step, step_controls in enumerate(initial_controls):
            if not (jnp.less_equal(jnp.abs(step_controls), max_control_norms + _NORM_TOLERANCE).all()):
                raise ValueError("The program expected that the initial_controls specified by "
                                 "the user conformed to max_control_norms, but the program "
                                 "found a conflict at initial_controls[{}]={} and "
                                 "max_control_norms={}."
                                 "".format(control_step, step_controls, max_control_norms))
        #ENDFOR
        controls = initial_controls

    return controls, max_control_norms


def slap_controls(complex_controls, controls, controls_shape,):
    """
    Reshape and transform controls in optimizer format
    to controls in cost function format.

    Arguments:
    complex_controls :: bool - whether or not the controls in cost function
         format are complex
    controls :: ndarray (2 * controls_size if COMPLEX else controls_size)
        - the controls in optimizer format
    controls_shape :: tuple(int) - 
    
    Returns:
    controls :: ndarray (controls_shape)- the controls in cost function format
    """
    # Transform the controls to C if they are complex.
    if complex_controls:
        real, imag = jnp.split(controls, 2)
        controls = real + 1j * imag
    # Reshape the controls.
    controls = jnp.reshape(controls, controls_shape)
    
    return controls


def strip_controls(complex_controls, controls):
    """
    Reshape and transform controls in cost function format
    to controls in optimizer format.

    Arguments:
    complex_controls :: bool - whether or not the controls in cost function
        format are complex
    controls :: ndarray (controls_shape) - the controls in cost function format

    Returns:
    controls :: ndarray (2 * controls_size if COMPLEX else controls_size)
        - the controls in optimizer format
    """
    # Flatten the controls.
    controls = jnp.ravel(controls)
    # Transform the controls to R2 if they are complex.
    if complex_controls:
        controls = jnp.hstack((jnp.real(controls), jnp.imag(controls)))
    
    return controls
