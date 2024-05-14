import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from models.differentialSim import DifferentialDrive, DifferentialSimulation
from typing import Tuple

def export_casadi_model():
    # create mpc to formulate mpc 
    MPC = AcadosOcp()

    # differential model setup 
    ## Define the states
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    yaw = ca.SX.sym('yaw')
    states = ca.vertcat(x, y, yaw)

    ## Define the inputs
    v = ca.SX.sym('v')
    w = ca.SX.sym('w')
    controls = ca.vertcat(v, w)

    ## Define the system equations \dot(x) = f(x,u)
    rhs = ca.vertcat(
        v * ca.cos(yaw),
        v * ca.sin(yaw),
        w
    )

    ## Define implicit DAE
    xdot = ca.SX.sym('xdot')
    ydot = ca.SX.sym('ydot')
    yawdot = ca.SX.sym('yawdot')
    dae = ca.vertcat(xdot, ydot, yawdot)

    ## Define dynamics system
    ### Explicit
    f_expl = rhs
    ### Implicit
    f_impl = dae - rhs

    # set model
    model = AcadosModel()
    model.x = states
    model.u = controls
    model.z = []
    model.p = []
    model.xdot = dae
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.name = "Differential_Drive"

    return model


class MPCController:

    def __init__(
            self,
            x0: np.ndarray,
            u0: np.ndarray,
            # goal_trajectory: np.ndarray,
            # goal_control: np.ndarray,
            state_cost_matrix: np.ndarray,
            control_cost_matrix: np.ndarray,
            terminal_cost_matrix: np.ndarray,
            state_lower_bound: np.ndarray,
            state_upper_bound: np.ndarray,
            control_lower_bound: np.ndarray,
            control_upper_bound: np.ndarray,
            N: int,
            dt: float
              
        ) -> None:
        """
        Initialize the MPC controller with the given parameters.
        """

        self.state_cost_matrix = state_cost_matrix
        self.control_cost_matrix = control_cost_matrix
        self.terminal_const_matrix = terminal_cost_matrix

        self.state_constraints = {
            'lbx': state_lower_bound,
            'ubx': state_upper_bound,
        }

        self.control_constraints = {
            'lbu': control_lower_bound,
            'ubu': control_upper_bound
        }

        self.N = N
        self.Ts = dt
        self.dt = self.N/self.Ts

        # Differential Drive Model
        self.differential_drive = DifferentialDrive(
            x0_initial=x0
        )

        # Export the Casadi Model
        self.model = export_casadi_model()

        # Create the ACADOS solver
        self.mpc_solver, self.mpc = self.create_mpc_solver(x0)

    def create_mpc_solver(self, x0: np.ndarray) -> Tuple[AcadosOcpSolver, AcadosOcp]:
        mpc = AcadosOcp()

        model = self.model
        mpc.model = model
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # set dimensions
        mpc.dims.N = self.N
        mpc.dims.nx = nx
        mpc.dims.nu = nu

        # Set cost
        Q_mat = self.state_cost_matrix
        Q_mat_e = self.terminal_const_matrix
        R_mat = self.control_cost_matrix    

        # External cost function
        mpc.cost.cost_type = 'NONLINEAR_LS'
        mpc.cost.cost_type_e = 'NONLINEAR_LS'
        mpc.model.cost_y_expr = ca.vertcat(model.x, model.u)
        mpc.model.cost_y_expr_e = model.x

        mpc.cost.yref = np.zeros((ny, ))
        mpc.cost.yref_e = np.zeros((ny_e, ))

        mpc.cost.W = block_diag(Q_mat, R_mat)
        mpc.cost.W_e = Q_mat_e

        # Set constraints
        mpc.constraints.lbx = self.state_constraints['lbx']
        mpc.constraints.ubx = self.state_constraints['ubx']
        mpc.constraints.idxbx = np.arange(nx)

        mpc.constraints.lbx_e = self.state_constraints['lbx']
        mpc.constraints.ubx_e = self.state_constraints['ubx']
        mpc.constraints.idxbx_e = np.arange(nx)

        mpc.constraints.lbu = self.control_constraints['lbu']
        mpc.constraints.ubu = self.control_constraints['ubu']
        mpc.constraints.idxbu = np.arange(nu)

        # Set solver options
        mpc.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'IRK'
        mpc.solver_options.nlp_solver_type = 'SQP_RTI'

        # Set prediction horizons
        mpc.solver_options.tf = self.Ts

        try:
            print("Creating MPC solver...")
            mpc_solver = AcadosOcpSolver(mpc, json_file='mpc_differential_drive.json')
            print("MPC solver created successfully.")
            return mpc_solver, mpc
        except Exception as e:
            print("Error creating MPC solver:", e)
            raise

        # mpc_solver = AcadosOcpSolver(mpc, json_file='mpc_differential_drive.json'

        return mpc_solver, mpc


    def solve_mpc(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem with the given initial state.
        """

        # Update the initial state in the solver
        # self.mpc_solver.set(0, "lbx", x0)
        # self.mpc_solver.set(0, "ubx", x0)

        # Solve the MPC problem
        status = self.mpc_solver.solve()

        if status != 0:
            raise Exception(f'Acados returned status {status}')

        # Get the optimal state and control trajectories
        simX = np.zeros((self.mpc.dims.N+1, self.mpc.dims.nx))
        simU = np.zeros((self.mpc.dims.N, self.mpc.dims.nu))

        for i in range(self.mpc.dims.N):
            simX[i, :] = self.mpc_solver.get(i, "x")
            simU[i, :] = self.mpc_solver.get(i, "u")

        simX[self.mpc.dims.N, :] = self.mpc_solver.get(self.mpc.dims.N, "x")

        return simX, simU


if __name__ == "__main__":
    # Initialize the MPC Controller     
    ## State and Control
    state_init = np.array([3.0, 0.0, 0.0])
    control_init = np.array([0.0, 0.0])

    ## Current State and Control
    state_current = state_init
    control_current = control_init

    ## Cost Matrices
    state_cost_matrix = np.diag([7.0, 7.0, 1.0])
    control_cost_matrix = np.diag([1.0, 1.0])
    terminal_cost_matrix = np.diag([7.0, 7.0, 1.0])

    ## Constraints
    state_lower_bound = np.array([0.0, 0.0, 0.0])
    state_upper_bound = np.array([5.0, 5.0, 0.0])
    control_lower_bound = np.array([-30.0, -30.0])
    control_upper_bound = np.array([30.0, 30.0])

    # Differential Drive Robot
    diffRobot = DifferentialDrive(
        x0_initial=state_init
    )

    ## Prediction Horizon
    N = 20
    sampling_time = 0.1
    Tsim = int(N/sampling_time)

    ## Tracks history
    xs = []
    us = []

    # MPC modules
    mpc = MPCController(
        x0=state_init,
        u0=control_init,
        state_cost_matrix=state_cost_matrix,
        control_cost_matrix=control_cost_matrix,
        terminal_cost_matrix=terminal_cost_matrix,
        state_lower_bound=state_lower_bound,
        state_upper_bound=state_upper_bound,
        control_lower_bound=control_lower_bound,
        control_upper_bound=control_upper_bound,
        N=N,
        dt=sampling_time
    )

    for step in range(Tsim):
        if step % (1/sampling_time) == 0:
            print('t =', step*sampling_time)

        state_ref = np.array([5, 0, 0])  # Desired state reference
        control_ref = np.array([0, 0])   # Desired control reference
        yref = np.concatenate((state_ref, control_ref))

        simX, simU = mpc.solve_mpc(state_current)

        # Get the first control input from the optimal solution
        u = simU[0, :]

        print('Control:', u)
        print('State:', state_current)

        # Apply the control input to the system
        x1 = state_current + sampling_time * diffRobot.forward_kinematic(u)

        xs.append(state_current)
        us.append(u)

        # Update state
        state_current = x1
        