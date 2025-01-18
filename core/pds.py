# This script is to provide the utilities for 1D pressure diffusion problem
import numpy as np
import matplotlib.pyplot as plt 
from DSS_analyzer_Mariner import Data1D_GAUGE # Load gauge data; use the dataframe to process the data
import optimizer.tso as tso # Load the time sampling optimizer

# Define the class for the 1D pressure diffusion problem; this class will only support single source term.
class PDS1D_SingleSource:
    def __init__(self):
        self.mesh = None # Mesh
        self.source = None # Source term
        self.lbc = None # Boundary conditions: left boundary condition
        self.rbc = None # Boundary conditions: right boundary condition
        self.initial = None # Initial condition
        self.diffusivity = None # Diffusivity
        self.t0 = 0 # Initial time

    # Define the parameters for the problem
    def set_mesh(self, mesh):
        self.mesh = mesh # Set mesh, 1D numpy array

    def set_source(self, source):
        self.source = source # Set source term, which would be the pressure gauge dataframe.
        print("Message from pds: Source set done.\nAlso, just a reminder: please make sure the data is cropped properly.")

    def set_bcs(self, lbc='Neumann', rbc='Neumann'):
        # Set boundary conditions, str: "Dirichlet" or "Neumann"
        # If it's not these two, return an error message
        if lbc not in ['Dirichlet', 'Neumann']:
            print("Left boundary condition must be either Dirichlet or Neumann")
            return
        if rbc not in ['Dirichlet', 'Neumann']:
            print("Right boundary condition must be either Dirichlet or Neumann")
            return
        self.lbc = lbc # Set left boundary condition
        self.rbc = rbc

    def set_initial(self, initial):
        # Define the initial condition
        self.initial = initial # 1D array of initial condition having same length as mesh array.

    def set_diffusivity(self, diffusivity):
        # Set diffusivity
        self.diffusivity = diffusivity # 1D array of diffusivity having same length as mesh array.

    def set_t0(self, t0):
        # Set initial time
        self.t0 = t0 # Initial time, float

    # Define the function to solve the problem
    
    def solve(self, optimizer = False, **kwargs):
        # If optimizer is false, then solve the problem with the given parameters and a given time step "dt", pass this to PDE solver.
        if not optimizer:
            # Generate the time array using t_total. If not given, then use the time array from the source term.
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                dt = kwargs['dt']
                time = np.arange(self.t0, t_total, dt)
                print("Time array generated using t_total.")
            else:
                dt = kwargs['dt']
                t_total = (self.source.calculate_time())[-1] # get the last time from the source term
                time = np.arange(self.t0, t_total, dt)
                print("Time array generated using the source term.")
            # Use the PDE solved defined in this library to solve the problem
            self.snapshot = np.zeros((len(time), len(self.mesh)))

            # Set the initial condition
            self.snapshot[0] = self.initial


        # TMP: Return 0 for now
        return 0