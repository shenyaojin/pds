# This script is to define the time sampling optimizer for the 1D pressure diffusion problem. It can help me to use a
# dynamic time step for the simulation which will be more efficient and accurate.

# Provide utilities for the time sampling optimizer
import numpy as np
from core import pds

def time_sampling_optimizer(pds1d, full_step_solution, half_step_solution, dt, tol=1e-3, *args, **kwargs):
    """
    Optimize the time sampling for the 1D pressure diffusion problem.
    Args:
        pds1d: the 1D pressure diffusion problem object
        full_step_solution: the solution at the full time step calculated by the solver
        half_step_solution: the solution at the half time step calculated by the solver
        tol: the tolerance for the optimization, default is 1e-3
        *args: additional arguments
        **kwargs: additional keyword arguments
    Returns:
        No return value. All the results will be stored/modifed in the pds1d object.
    """
    # Calculate the error
    error = np.linalg.norm(full_step_solution - half_step_solution, ord=2) / np.linalg.norm(full_step_solution, ord=2)

    # Update the time parameter
    updated_dt = adjust_dt(dt, tol)

    # Decide whether to accept the snapshot
    if error <= tol:
        # The error is small enough, then accept it and append the new snapshot/taxis
        pds1d.snapshot.append(full_step_solution)
        pds1d.taxis.append(pds1d.taxis[-1] + dt)
        pds1d.record_log("Dynamic time sampling updated. dt = ", dt, "Next loop dt will be", updated_dt, "\n")

    return dt

def adjust_dt(dt, tol):
    return 0.01