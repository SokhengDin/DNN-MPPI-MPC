import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from models.raceCarSim import RacecarModel
from utils.plot_differential_drive import Simulation
from typing import Tuple

def export_casadi_model():

    L = 0.325
    ## Define the states
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    yaw = ca.SX.sym('yaw')
    v = ca.SX.sym('v')
    states = ca.vertcat(x, y, yaw, v)

    ## Define the inputs
    a = ca.SX.sym('a')
    delta = ca.SX.sym('delta')
    controls = ca.vertcat(a, delta)

    ## Define the system equations \dot(x) = f(x,u)
    rhs = ca.vertcat(
        v * ca.cos(yaw),
        v * ca.sin(yaw),
        v * ca.tan(delta) / L,
        a
    )


    ## Define implicit DAE
    xdot = ca.SX.sym('xdot')
    ydot = ca.SX.sym('ydot')
    yawdot = ca.SX.sym('yawdot')
    vdot = ca.SX.sym('vdot')
    dae = ca.vertcat(xdot, ydot, yawdot, vdot)

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
    model.name = "RaceCar"

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
            dt: float,
            Ts: float,
            cost_type: str
              
        ) -> None:
        """
        Initialize the MPC controller with the given parameters.
        """

        # Cost type
        self.cost_type = cost_type

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
        self.dt = dt
        self.Ts = Ts

        # RaceCar  Model
        self.racecar = RacecarModel(x0, L=0.325, dt=self.dt)

        # Export the Casadi Model
        self.model = export_casadi_model()

        # Create the ACADOS solver
        self.mpc_solver, self.mpc = self.create_mpc_solver(x0)

        # Create the ACADOS simulator
        # self.sim_solver = self.create_sim_solver()

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
        if self.cost_type == 'LINEAR_LS':
            unscale = self.N / self.Ts
            mpc.cost.cost_type = 'LINEAR_LS'
            mpc.cost.cost_type_e = 'LINEAR_LS'
            mpc.cost.W = block_diag(Q_mat, R_mat)
            mpc.cost.W_e = Q_mat_e
            Vx = np.zeros((ny, nx))
            Vx[:nx, :nx] = np.eye(nx)
            mpc.cost.Vx = Vx
            mpc.cost.Vx_e = np.eye(nx)
            Vu = np.zeros((ny, nu))
            Vu[-nu:, -nu:] = np.eye(nu)
            mpc.cost.Vu = Vu

        elif self.cost_type == 'NONLINEAR_LS':
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

        mpc.constraints.x0  = x0

        mpc.constraints.lbu = self.control_constraints['lbu']
        mpc.constraints.ubu = self.control_constraints['ubu']
        mpc.constraints.idxbu = np.arange(nu)

        # Set solver options
        mpc.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'ERK'
        mpc.solver_options.nlp_solver_type = 'SQP_RTI'
        mpc.solver_options.sim_method_num_stages = 4
        mpc.solver_options.sim_method_num_steps = 3

        # Set prediction horizons
        mpc.solver_options.tf = self.Ts

        try:
            print("Creating MPC solver...")
            mpc_solver = AcadosOcpSolver(mpc, json_file='race_car_mpc.json')
            print("MPC solver created successfully.")
            return mpc_solver, mpc
        except Exception as e:
            print("Error creating MPC solver:", e)
            raise

    def create_sim_solver(self) -> AcadosSimSolver:
        sim = AcadosSim()

        model = self.model
        sim.model = model

        # Set simulation options
        sim.solver_options.T = self.dt
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 3

        try:
            print("Creating Sim solver...")
            sim_solver = AcadosSimSolver(sim, json_file='sim_differential_drive.json')
            print("Sim solver created successfully.")
            return sim_solver
        except Exception as e:
            print("Error creating Sim solver:", e)
            raise



    def solve_mpc(self, x0: np.ndarray,
                        simX: np.ndarray,
                        simU: np.ndarray,
                        yref: np.ndarray,
                        yref_N: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem with the given initial state.
        """

        nx = self.model.x.size()[0]

        # Ensure x0 is the correct shape
        if x0.shape[0] != nx:
            raise ValueError(f"Initial state x0 must have dimension {nx}, but got {x0.shape[0]}.")

        # Solve the MPC problem
        status = self.mpc_solver.solve()

        if status != 0:
            raise Exception(f'Acados returned status {status}')

        for i in range(self.mpc.dims.N):
            self.mpc_solver.set(i, "yref", yref)

            simX[i, :] = self.mpc_solver.get(i, "x")
            simU[i, :] = self.mpc_solver.get(i, "u")  

        # Update the initial state in the solver
        self.mpc_solver.set(self.mpc.dims.N, "yref", yref_N)

        self.mpc_solver.set(0, "lbx", x0)
        self.mpc_solver.set(0, "ubx", x0)   

        simX[self.mpc.dims.N, :] = self.mpc_solver.get(self.mpc.dims.N, "x")

        return simX, simU
    
    def runge_kutta(self, f: np.ndarray, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute the state update using the Runge-Kutta method.
        """

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    

    def update_stateRungeKutta(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Use the Runge-Kutta method to update the state based on the current state and control inputs.
        """

        x1 = self.runge_kutta(self.racecar.forward_kinematic, x0, u, self.dt)
        return x1


    def update_state(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Use the AcadosSimSolver to update the state based on the current state and control inputs.
        """

        self.sim_solver.set("x", x0)
        self.sim_solver.set("u", u)
        status = self.sim_solver.solve()

        if status != 0:
            raise Exception(f'Simulation failed with status {status}')

        x1 = self.sim_solver.get('x')
        return x1

if __name__ == "__main__":
    # Initialize the MPC Controller     
    ## State and Control
    state_init = np.array([0.0, 0.0, 0.0, 0.0])
    control_init = np.array([0.0, 0.0])

    ## Current State and Control
    state_current = state_init
    control_current = control_init

    ## Cost Matrices
    state_cost_matrix = np.diag([100.0, 100.0, 50.0, 20.0]) 
    control_cost_matrix = np.diag([1.0, 0.1])                
    terminal_cost_matrix = np.diag([100.0, 100.0, 50.0, 20.0]) 

    ## Constraints
    state_lower_bound = np.array([-5.0, -5.0, -np.pi, -10.0])
    state_upper_bound = np.array([5.0, 5.0, np.pi, 10.0])
    control_lower_bound = np.array([-2.0, -1.0])  
    control_upper_bound = np.array([2.0, 1.0])


    # RaceCar robot
    raceCar = RacecarModel(
        initial_state=state_init
    )

    ## Prediction Horizon
    N = 50
    sampling_time = 0.05
    Ts = 3.0
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
        dt=sampling_time,
        Ts=Ts,
        cost_type='NONLINEAR_LS'
    )


    # Get the optimal state and control trajectories
    simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
    simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

    for step in range(Tsim):
        if step % (1/sampling_time) == 0:
            print('t =', step*sampling_time)

        state_ref = np.array([2.0, 4.0, 0.0, 4.0])  
        control_ref = np.array([1.0, 0.5])   
        yref = np.concatenate([state_ref, control_ref])
        yref_N = np.array([2.0, 4.0, 0.0, 4.0])

        simX, simU = mpc.solve_mpc(state_current, simX, simU, yref, yref_N)

        # Get the first control input from the optimal solution
        u = simU[0, :]

        print('Control:', u)
        print('State:', state_current)

        # Apply the control input to the system
        # x1 = state_current + sampling_time * raceCar.forward_kinematic(u)
        # x1 = mpc.update_state(state_current, u)
        x1 = mpc.update_stateRungeKutta(state_current, u)

        xs.append(state_current)
        us.append(u)

        # Update state
        state_current = x1