import numpy as np
from DSS_analyzer_Mariner import Data1D_GAUGE
from core import pds


def MatrixBuilder_1D_SingleSource(pds1d, dt):
    """
    Build the matrix for the 1D diffusion problem with a single source term.
    Args:
    mesh: 1D numpy array
    diffusivity: 1D numpy array
    lbc: str, "Dirichlet" or "Neumann"
    rbc: str, "Dirichlet" or "Neumann"
    source: float
    sourceidx: int
    Returns:
    A: 2D numpy array
    b: 1D numpy array
    """
    # Calculate alpha; in this case, the dx is constant
    dx = pds1d.mesh[1] - pds1d.mesh[0]
    alpha = pds1d.diffusivity * dt / dx**2
    nx = len(pds1d.mesh)

    # Define the matrix A
    A = np.zeros((nx, nx))

    # Define the vector b
    b = np.zeros(nx)

    # Fill the matrix A and vector b
    for iter in range(1, nx-1):
        A[iter, iter-1] = -alpha[iter]
        A[iter, iter] = 1 + 2*alpha[iter]
        A[iter, iter+1] = -alpha[iter]

    # Fill the vector b, should be the snapshot of the previous time step
    b = pds.snapshot[-1]

    # Apply the boundary conditions
    # Scan the left boundary
    if pds1d.lbc == 'Dirichlet':
        A[0, 0] = 1 # No need to initialize for the for loop above
        b[0] = 0
    elif pds1d.lbc == 'Neumann':
        A[0, 0] = -1
        A[0, 1] = 1
        b[0] = 0

    # Scan the right boundary
    if pds1d.rbc == 'Dirichlet':
        A[nx-1, nx-1] = 1
        b[nx-1] = 0
    elif pds1d.rbc == 'Neumann':
        A[nx-1, nx-1] = -1
        A[nx-1, nx-2] = 1
        b[nx-1] = 0

    # Apply the source term
    # Get the source value from PG data
    source_value = pds1d.source.get_value_by_time(pds1d.taxis[-1])
    A[pds1d.sourceidx, :] = 0 # Initialize the row
    A[pds1d.sourceidx, pds1d.sourceidx] = 1
    b[pds1d.sourceidx] = source_value

    return A, b