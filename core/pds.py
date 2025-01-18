# This script is to provide the utilities for 1D pressure diffusion problem
# Developed by Shenyao Jin, shenyaojin@mines.edu
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from time import time

from DSS_analyzer_Mariner import Data1D_GAUGE # Load gauge data; use the dataframe to process the data
import optimizer.tso as tso # Load the time sampling optimizer
from solver import matbuilder, PDEsolver_IMP, PDESolver_EXP # Load the matrix builder and PDE solver

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
        self.taxis = None # Time axis
        self.sourceidx = None # Source index in the mesh; would be a single index
        self.snapshot = None # Snapshot of the solution
        self.history = [] # History of the solution

    # Define the parameters for the problem

    def set_mesh(self, mesh):
        self.mesh = mesh # Set mesh, 1D numpy array
        self.history.append("Mesh set done.")

    def set_source(self, source):
        self.source = source # Set source term, which would be the pressure gauge dataframe.
        print("Message from pds: Source set done.\nAlso, just a reminder: please make sure the data is cropped properly.")
        self.history.append("Source set done.")

    def set_bcs(self, lbc='Neumann', rbc='Neumann'):
        # Set boundary conditions, str: "Dirichlet" or "Neumann"
        # If it's not these two, return an error message
        if lbc not in ['Dirichlet', 'Neumann']:
            print("Left boundary condition must be either Dirichlet or Neumann")
            self.history.append("Left boundary condition must be either Dirichlet or Neumann")
            return
        if rbc not in ['Dirichlet', 'Neumann']:
            print("Right boundary condition must be either Dirichlet or Neumann")
            self.history.append("Right boundary condition must be either Dirichlet or Neumann")
            return
        self.lbc = lbc # Set left boundary condition
        self.rbc = rbc
        self.history.append("Boundary conditions set done.")

    def set_initial(self, initial):
        # Define the initial condition
        self.initial = initial # 1D array of initial condition having same length as mesh array.
        self.history.append("Initial condition set done.")

    def set_diffusivity(self, diffusivity):
        # Set diffusivity
        self.diffusivity = diffusivity # 1D array of diffusivity having same length as mesh array.
        self.history.append("Diffusivity set done.")

    def set_t0(self, t0):
        # Set initial time
        self.t0 = t0 # Initial time, float
        self.history.append("Initial time set done.")

    def set_sourceidx(self, sourceidx):
        # Set source index
        self.sourceidx = sourceidx
        # Check if the source index is in the mesh
        if sourceidx not in self.mesh:
            print("Source index is not in the mesh.")
            self.history.append("Source index is not in the mesh. Source index initialization failed.")
            return
        self.history.append("Source index set done.")

    # Print the all the parameters
    def print(self):
        for key, value in self.__dict__.items():
            print(key, ":", value)

    # Define the function to solve the problem

    def solve(self, optimizer = False, **kwargs):
        # If optimizer is false, then solve the problem with the given parameters and a given time step "dt", pass this to PDE solver.
        # Implicit solver

        # Future plan: use decorator to enhance the function here.
        if not optimizer:
            # Generate the time array using t_total. If not given, then use the time array from the source term.
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                dt = kwargs['dt']
                time = np.arange(self.t0, t_total, dt)
                print("Time array generated using t_total.")
                self.history.append("Time array generated using t_total.")
            else:
                dt = kwargs['dt']
                t_total = (self.source.calculate_time())[-1] # get the last time from the source term
                time = np.arange(self.t0, t_total, dt)
                print("Time array generated using the source term.")
                self.history.append("Time array generated using the source term.")
            # Initialize the snapshot
            self.snapshot = []

            # Set the initial condition
            self.snapshot[0] = self.initial
        else:
            dt_init = kwargs['dt_init']
            if 't_total' in kwargs:
                t_total = kwargs['t_total']
                time = np.arange(self.t0, t_total, dt_init)
                print("Time array generated using t_total.")
            else:
                t_total = (self.source.calculate_time())[-1]

            # Initialize the snapshot
            self.snapshot = []

            # Set the initial condition
            self.snapshot[0] = self.initial

        # get the intermediate time parameter, optimizer = True -> dt; optimizer = False -> dt_init
        time_parameter = -1
        # Initialize the time parameter
        if optimizer:
            time_parameter = dt_init
        else:
            time_parameter = dt

        # initialize the taxis
        taxis_tmp = [self.t0]

        self.record_log("Start to solve the problem.")

        # start to loop through the time array
        while taxis_tmp[-1] < t_total:
            # Full step solution. For the non-optimizer case, only full step solution is needed.
            # Call the Matrix builder
            A, b = matbuilder.MatrixBuilder_1D(self.mesh, self.diffusivity,
                                               self.lbc, self.rbc, self.source, self.sourceidx, time_parameter)
            # Call the solver, get an updated snapshot
            if 'mode' in kwargs:
                if kwargs['mode'] == 'implicit':
                    snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, mode='crank-nicolson') #call the implicit solver
                else:
                    snapshot_upd = PDESolver_EXP.solver_explicit(A, b) #call the explicit solver
            else:
                # Default mode is implicit solver
                snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, mode='crank-nicolson')

            # If no optimizer, append the snapshot to the list
            if not optimizer:
                self.snapshot.append(snapshot_upd)
                # Update the time
                taxis_tmp.append(taxis_tmp[-1] + time_parameter)
                self.record_log("Full step solution done. Time:", taxis_tmp[-1])
            else:
                # Call the half step optimizer, then estimate the error
                # I'll implement this later
                return 0
        # After the loop, update the taxis
        self.taxis = taxis_tmp
        self.record_log("Problem solved.")

    # Message function -> History
    def record_log(self, text):
        # get time
        time_now = time()
        # append the text to the history
        msg = text + "time:", time_now
        self.history.append(msg)

    # Solution data processing