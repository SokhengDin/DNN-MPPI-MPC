import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from scipy.linalg import block_diag
from typing import Tuple

def export_casadi_model():
    # Race car model setup
    ## Define the states
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    yaw = ca.SX.sym('yaw')
    v = ca.SX.sym('v')
    states = ca.vertcat(x, y, yaw, v)

    ## Define the controls
    a = ca.SX.sym('a')
    delta = ca.SX.sym('delta')
    controls = ca.vertcat(a, delta)

    ## Define the model parameters
    L = 0.325 
    W = 0.2 
    m = 4.0 
    Iz = 0.05865
    Cf = 1000.0  
    Cr = 1000.0  
    lf = 0.325 / 2 
    lr = 0.325 / 2

    ## Define the system equations \dot{x} = f(x, u)
    beta = ca.arctan(lr / (lf + lr) * ca.tan(delta))
    f_x = a
    f_y = 2 * (Cf * ca.sin(ca.arctan((v * ca.sin(beta) + lf * yaw) / (v * ca.cos(beta)))) * ca.cos(delta) +
               Cr * ca.sin(ca.arctan((v * ca.sin(beta) - lr * yaw) / (v * ca.cos(beta)))))
    rhs = ca.vertcat(
        v * ca.cos(yaw + beta),
        v * ca.sin(yaw + beta),
        v * ca.sin(beta) / lr,
        (f_x - f_y * ca.sin(delta)) / m
    )

    ## Define implicit DAE
    xdot = ca.SX.sym('xdot', 4)
    dae = ca.vertcat(xdot)

    ## Define dynamics system
    ### Explicit
    f_expl = rhs
    ### Implicit
    f_impl = dae - rhs

    # Set model
    model = AcadosModel()
    model.x = states
    model.u = controls
    model.z = []
    model.xdot = dae
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.name = "Race_Car_Dynamics"

    return model

class NMPCController:
    def __init__(
            self,
            x0: np.ndarray,
            u0: np.ndarray,
            xref: np.ndarray,
            uref: np.ndarray,
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
            cost_type: str = 'LINEAR_LS'
    ) -> None:
        """
        Initialize the NMPC controller with the given parameters.
        """

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

        # Export the Casadi Model
        self.model = export_casadi_model()

        # Create the ACADOS solver
        self.mpc_solver, self.mpc = self.create_ocp_solver(x0, xref, uref)

        # Create the ACADOS simulator
        # self.sim_solver = self.create_sim_solver()

    def create_ocp_solver(self, x0: np.ndarray, xref: np.ndarray, uref: np.ndarray) -> Tuple[AcadosOcpSolver, AcadosOcp]:
        ocp = AcadosOcp()

        model = self.model
        ocp.model = model
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # Set dimensions
        ocp.dims.N = self.N
        ocp.dims.nx = nx
        ocp.dims.nu = nu

        # Set cost
        Q_mat = self.state_cost_matrix
        Q_mat_e = self.terminal_const_matrix
        R_mat = self.control_cost_matrix

        # External cost function
        if self.cost_type == 'LINEAR_LS':
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'
            ocp.cost.W = block_diag(Q_mat, R_mat)
            ocp.cost.W_e = Q_mat_e
            Vx = np.zeros((ny, nx))
            Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vx = Vx
            ocp.cost.Vx_e = np.eye(nx)
            Vu = np.zeros((ny, nu))
            Vu[nx:nx+nu, 0:nu] = np.eye(nu)
            ocp.cost.Vu = Vu
            ocp.cost.yref = np.concatenate((xref, uref))
            ocp.cost.yref_e = xref

        elif self.cost_type == 'NONLINEAR_LS':
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'
            ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
            ocp.model.cost_y_expr_e = model.x
            ocp.cost.yref = np.concatenate((xref, uref))
            ocp.cost.yref_e = xref
            ocp.cost.W = block_diag(Q_mat, R_mat)
            ocp.cost.W_e = Q_mat_e

        # Set constraints
        ocp.constraints.x0 = x0

        ocp.constraints.lbx = self.state_constraints['lbx']
        ocp.constraints.ubx = self.state_constraints['ubx']
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.lbx_e = self.state_constraints['lbx']
        ocp.constraints.ubx_e = self.state_constraints['ubx']
        ocp.constraints.idxbx_e = np.arange(nx)

        ocp.constraints.lbu = self.control_constraints['lbu']
        ocp.constraints.ubu = self.control_constraints['ubu']
        ocp.constraints.idxbu = np.arange(nu)

        # Set solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.qp_solver_cond_N = self.N

        # Set prediction horizons
        ocp.solver_options.tf = self.Ts

        try:
            print("Creating OCP solver...")
            ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
            print("OCP solver created successfully.")
            return ocp_solver, ocp
        except Exception as e:
            print("Error creating OCP solver:", e)
            raise

    def create_sim_solver(self) -> AcadosSimSolver:
        sim = AcadosSim()

        model = self.model
        sim.model = model

        # Set simulation options
        sim.solver_options.T = self.dt
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 3

        try:
            print("Creating Sim solver...")
            sim_solver = AcadosSimSolver(sim, json_file='acados_sim.json')
            print("Sim solver created successfully.")
            return sim_solver
        except Exception as e:
            print("Error creating Sim solver:", e)
            raise

    def solve_mpc(self, x0: np.ndarray,
                        simX: np.ndarray,
                        simU: np.ndarray,
                        yref: np.ndarray,
                        yref_N: np.ndarray,
                        obstacle_positions: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem with the given initial state.
        """

        nx = self.model.x.size()[0]

        # Ensure x0 is the correct shape
        if x0.shape[0] != nx:
            raise ValueError(f"Initial state x0 must have dimension {nx}, but got {x0.shape[0]}.")

        # Update the initial state in the solver
        self.mpc_solver.set(0, "lbx", x0)
        self.mpc_solver.set(0, "ubx", x0)

        # Set the obstacle positions
        # param_values = obstacle_positions.flatten()

        # # Set the obstacle positions in the solver
        # for i in range(self.N+1):
        #     self.mpc_solver.set(i, 'p', param_values)

        # Set reference trajectory
        for i in range(self.mpc.dims.N):
            self.mpc_solver.set(i, "yref", yref)
        self.mpc_solver.set(self.mpc.dims.N, "yref", yref_N)

        # Warm up the solver
        for i in range(self.N):
            self.mpc_solver.set(i, "x", simX[i, :])
            self.mpc_solver.set(i, "u", simU[i, :])
        self.mpc_solver.set(self.N, "x", simX[self.N, :])

        # Solve the MPC problem
        status = self.mpc_solver.solve()

        # if status != 0:
        #     raise Exception(f'Acados returned status {status}')

        for i in range(self.mpc.dims.N):
            self.mpc_solver.set(i, "yref", yref)
            simX[i, :] = self.mpc_solver.get(i, "x")
            simU[i, :] = self.mpc_solver.get(i, "u")
        simX[self.mpc.dims.N, :] = self.mpc_solver.get(self.mpc.dims.N, "x")

        return simX, simU

    def runge_kutta(self, f, x, u, dt):
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def race_car_dynamics(self, x0, u):
        # Extract the model parameters
        L = 0.325 
        W = 0.2 
        m = 4.0 
        Iz = 0.05865
        Cf = 1000.0  
        Cr = 1000.0  
        lf = 0.325 / 2 
        lr = 0.325 / 2

        # Extract the states and controls
        x, y, yaw, v = x0
        a, delta = u

        # Compute the state derivatives
        beta = np.arctan(lr / (lf + lr) * np.tan(delta))
        f_x = a
        f_y = 2 * (Cf * np.sin(np.arctan((v * np.sin(beta) + lf * yaw) / (v * np.cos(beta)))) * np.cos(delta) +
                   Cr * np.sin(np.arctan((v * np.sin(beta) - lr * yaw) / (v * np.cos(beta)))))
        dx = v * np.cos(yaw + beta)
        dy = v * np.sin(yaw + beta)
        dyaw = v * np.sin(beta) / lr
        dv = (f_x - f_y * np.sin(delta)) / m

        return np.array([dx, dy, dyaw, dv])

    def update_state_runge_kutta(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Use the Runge-Kutta method to update the state based on the current state and control inputs.
        """

        x1 = self.runge_kutta(self.race_car_dynamics, x0, u, self.dt)
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