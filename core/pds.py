# This script is to provide the utilities for 1D pressure diffusion problem
# Developed by Shenyao Jin, shenyaojin@mines.edu
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
        """
        Set diffusivity for the problem.
        If a single value is provided, broadcast it to the length of the mesh.
        """
        if isinstance(diffusivity, (int, float)):  # Check if diffusivity is a single number
            self.diffusivity = np.full(len(self.mesh), diffusivity)
            print("Diffusivity is a single scalar value, broadcasted to the mesh length.")
        elif isinstance(diffusivity, (list, np.ndarray)) and len(diffusivity) == len(
                self.mesh):  # Check for proper length
            self.diffusivity = np.array(diffusivity)
        else:
            raise ValueError(
                "Diffusivity must be either a single scalar value or an array of the same length as the mesh.")

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

    # Define the function to solve the problem; core function
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
            self.snapshot.append(self.initial)
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
            self.snapshot.append(self.initial)

        # get the intermediate time parameter, optimizer = True -> dt; optimizer = False -> dt_init
        time_parameter = -1
        # Initialize the time parameter
        if optimizer:
            time_parameter = dt_init
        else:
            time_parameter = dt

        # initialize the taxis
        self.taxis = [self.t0]

        self.record_log("Start to solve the problem.")

        # start to loop through the time array
        while self.taxis[-1] < t_total:
            # Before the loop, get the value of source term at the current time
            source_val = self.source.get_value_by_time(self.taxis[-1])
            self.record_log("Time:", self.taxis[-1], "Source term:", source_val)

            # Full step solution. For the non-optimizer case, only full step solution is needed.
            # Call the Matrix builder
            A, b = matbuilder.matrix_builder_1d_single_source(self, time_parameter)
            # Call the solver, get an updated snapshot
            if 'mode' in kwargs:
                if kwargs['mode'] == 'implicit':
                    snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, solver='numpy') #call the implicit solver
                else:
                    snapshot_upd = PDESolver_EXP.solver_explicit(A, b) #call the explicit solver
            else:
                # Default mode is implicit solver
                snapshot_upd = PDEsolver_IMP.solver_implicit(A, b, solver='numpy')

            # If no optimizer, append the snapshot to the list
            if not optimizer:
                self.snapshot.append(snapshot_upd)
                # Update the time
                self.taxis.append(self.taxis[-1] + time_parameter)
                self.record_log("Full step solution done. taxis[-1]=", self.taxis[-1])
                # Print the progress if print_progress is True
                if 'print_progress' in kwargs:
                    if kwargs['print_progress']:
                        print("Time:", self.taxis[-1], "Source term:", source_val)
            else:
                # Call the half step optimizer, then estimate the error
                # Call the Matrix builder to get the matrix for the half step
                A_half, b_half = matbuilder.matrix_builder_1d_single_source(self, time_parameter / 2)
                snapshot_middle_tmp = PDEsolver_IMP.solver_implicit(A_half, b_half, solver='numpy')
                # Store this tmp snapshot to the list
                self.snapshot.append(snapshot_middle_tmp)
                # then call the matrix builder again to get the matrix for the full step
                A_full, b_full = matbuilder.matrix_builder_1d_single_source(self, time_parameter / 2)
                snapshot_full = PDEsolver_IMP.solver_implicit(A_full, b_full, solver='numpy')
                # Delete the tmp snapshot
                del self.snapshot[-1]
                # Call the optimizer to 1. decide whether to accept the full step solution; 2. decide the next time
                # step size; 3. update the snapshot list (if accepted); if not accepted, decrease the time step size
                # and redo the full step solution. 4. Record the log.
                time_parameter = tso.time_sampling_optimizer(self, snapshot_full, snapshot_middle_tmp, time_parameter,
                                                                tol=1e-3, **kwargs)
                # Print the progress if print_progress is True
                if 'print_progress' in kwargs:
                    if kwargs['print_progress']:
                        print("Time:", self.taxis[-1], "Source term:", source_val)

        # Convert the snapshot/taxis to numpy array
        self.snapshot = np.array(self.snapshot)
        self.taxis = np.array(self.taxis)

        self.record_log("Problem solved")
        print("Problem solved.")

    # History recording
    def record_log(self, *args):
        time_now = datetime.now()
        # Concatenate all arguments
        msg = " ".join(map(str, args)) + f" | Time: {time_now}"
        # Append the formatted message to the history
        self.history.append(msg)

    # Solution data processing
    def get_solution(self):
        return self.snapshot, self.taxis

    def get_val_at_source_idx(self):
        return self.snapshot[:, self.sourceidx]

    def get_val_at_idx(self, idx):
        return self.snapshot[:, idx]

    def get_val_at_time(self, time):
        idx = np.argmin(np.abs(self.taxis - time))
        print("Closest time = ", self.taxis[idx])
        return self.snapshot[idx]

    # Plot the solution
    def plot_solution(self, **kwargs):

        # Extract cmap if given
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
        else:
            cmap = 'bwr'

        plt.figure()
        plt.imshow(self.snapshot.T, aspect='auto', cmap=cmap,
                   extent=[self.taxis[0], self.taxis[-1], self.mesh[0], self.mesh[-1]])

        # Invert the y-axis
        plt.gca().invert_yaxis()
        plt.xlabel("Time/s")
        plt.ylabel("Distance/ft")
        plt.show()