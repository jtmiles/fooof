"""Functions that can be used for model fitting.

NOTES
-----
- FOOOF currently (only) uses the exponential and gaussian functions.
- Linear & Quadratic functions are from previous versions of FOOOF.
    - They are left available for easy swapping back in, if desired.
"""

import numpy as np

from fooof_PLA.core.errors import InconsistentDataError
###################################################################################################
###################################################################################################

def gaussian_function(xs, *params):
    """Gaussian fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define gaussian function.

    Returns
    -------
    ys : 1d array
        Output values for gaussian function.
    """

    ys = np.zeros_like(xs)

    for ii in range(0, len(params), 3):

        ctr, hgt, wid = params[ii:ii+3]

        ys = ys + hgt * np.exp(-(xs-ctr)**2 / (2*wid**2))

    return ys


def mod_expo_function(xs, offset, log_knee, exp):
    """Hill-knee Exponential fitting function, for fitting aperiodic component with a 'log knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    offset, log_knee, exp : floats
        Parameters (offset, knee, exp) that define Lorentzian function:
        y = offset + log_knee * exp - np.log10(10**(log_knee * exp) + xs**exp

    Returns
    -------
    ys : 1d array
        Output values for exponential function.
    """

    ys = np.zeros_like(xs)
    fmin = 1 #np.min(xs)

    ys = ys + offset + np.log10(10**(log_knee * exp) + fmin**exp) - np.log10(10**(log_knee * exp) + xs**exp)

    return ys


def get_fixed_mod_func(offset_fixed = False, knee_fixed = False, exp_fixed = False):
    #breakpoint()
    log_knee_fixed = knee_fixed
    fmin = 1 #np.min(xs)
    try:
        if offset_fixed and not (knee_fixed or exp_fixed): # only fixed offset
            def fixed_function(xs, log_knee, exp):
                ys = np.zeros_like(xs)
                #return ys + offset_fixed + log_knee * exp - np.log10(10**(log_knee * exp) + xs**exp)
                return ys + offset_fixed + np.log10(10**(log_knee * exp) + fmin**exp) - np.log10(10**(log_knee * exp) + xs**exp)
        elif knee_fixed and not (offset_fixed or exp_fixed): # only fixed knee
            def fixed_function(xs, offset, exp):
                ys = np.zeros_like(xs)
                #return ys + offset + log_knee_fixed * exp - np.log10(10**(log_knee_fixed * exp) + xs**exp)
                return ys + offset + np.log10(10**(log_knee_fixed * exp) + fmin**exp) - np.log10(10**(log_knee_fixed * exp) + xs**exp)
        elif exp_fixed and not (offset_fixed or knee_fixed): # only fixed exponent
            def fixed_function(xs, offset, log_knee):
                ys = np.zeros_like(xs)
                #return ys + offset + log_knee * exp_fixed - np.log10(10**(log_knee * exp_fixed) + xs**exp_fixed)
                return ys + offset + np.log10(10**(log_knee * exp_fixed) + fmin**exp_fixed) - np.log10(10**(log_knee * exp_fixed) + xs**exp_fixed)
        elif offset_fixed and knee_fixed and not exp_fixed: # only optimize exponent
            def fixed_function(xs, exp):
                ys = np.zeros_like(xs)
                #return ys + offset_fixed + log_knee_fixed * exp - np.log10(10**(log_knee_fixed * exp) + xs**exp)
                return ys + offset_fixed + np.log10(10**(log_knee_fixed * exp) + fmin**exp) - np.log10(10**(log_knee_fixed * exp) + xs**exp)
        elif offset_fixed and exp_fixed and not knee_fixed: # only optimize knee
            def fixed_function(xs, log_knee):
                ys = np.zeros_like(xs)
                #return ys + offset_fixed + log_knee * exp_fixed - np.log10(10**(log_knee * exp_fixed) + xs**exp_fixed)
                return ys + offset_fixed + np.log10(10**(log_knee * exp_fixed) + fmin**exp_fixed) - np.log10(10**(log_knee * exp_fixed) + xs**exp_fixed)
        elif knee_fixed and exp_fixed and not offset_fixed: # only optimize offset
            def fixed_function(xs, offset):
                ys = np.zeros_like(xs)
                #return ys + offset + log_knee_fixed * exp_fixed - np.log10(10**(log_knee_fixed * exp_fixed) + xs**exp_fixed)
                return ys + offset + np.log10(10**(log_knee_fixed * exp_fixed) + fmin**exp_fixed) - np.log10(10**(log_knee_fixed * exp_fixed) + xs**exp_fixed)
        return fixed_function
    except:
        if not (offset or knee or exp):
            print('no parameters are fixed')
        elif offset and knee and exp:
            print('all parameters are fixed')


def linear_function(xs, *params):
    """Linear fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define linear function.

    Returns
    -------
    ys : 1d array
        Output values for linear function.
    """

    ys = np.zeros_like(xs)

    offset, slope = params

    ys = ys + offset + (xs*slope)

    return ys


def quadratic_function(xs, *params):
    """Quadratic fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define quadratic function.

    Returns
    -------
    ys : 1d array
        Output values for quadratic function.
    """

    ys = np.zeros_like(xs)

    offset, slope, curve = params

    ys = ys + offset + (xs*slope) + ((xs**2)*curve)

    return ys


def get_pe_func(periodic_mode):
    """Select and return specified function for periodic component.

    Parameters
    ----------
    periodic_mode : {'gaussian'}
        Which periodic fitting function to return.

    Returns
    -------
    pe_func : function
        Function for the periodic component.

    Raises
    ------
    ValueError
        If the specified periodic mode label is not understood.

    """

    if periodic_mode == 'gaussian':
        pe_func = gaussian_function
    else:
        raise ValueError("Requested periodic mode not understood.")

    return pe_func


def get_mod_ap_func(aperiodic_mode):
    if aperiodic_mode == 'fixed':
        raise ValueError("No non-knee parametrization for Alan's version")
    elif aperiodic_mode == 'knee':
        ap_func = mod_expo_function
    else:
        raise ValueError("Requested aperiodic mode not understood.")

    return ap_func


def infer_ap_func(aperiodic_params):
    """Infers which aperiodic function was used, from parameters.

    Parameters
    ----------
    aperiodic_params : list of float
        Parameters that describe the aperiodic component of a power spectrum.

    Returns
    -------
    aperiodic_mode : {'fixed', 'knee'}
        Which kind of aperiodic fitting function the given parameters are consistent with.

    Raises
    ------
    InconsistentDataError
        If the given parameters are inconsistent with any available aperiodic function.
    """

    if len(aperiodic_params) == 2:
        aperiodic_mode = 'fixed'
    elif len(aperiodic_params) == 3:
        aperiodic_mode = 'knee'
    else:
        raise InconsistentDataError("The given aperiodic parameters are "
                                    "inconsistent with available options.")

    return aperiodic_mode
