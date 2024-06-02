import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import casadi as ca

from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from models.raceCarSim import RacecarModel
from utils.plot_differential_drive import Simulation
from typing import Tuple

def export_casadi_model():

    L = 2.5
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
    model.name = "RaceCarObstacle"

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
            obstacle_positions: np.ndarray,
            obstacle_radii: np.ndarray,
            safe_distance: float,
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

        self.obstacle_positions = obstacle_positions
        self.obstacle_radii     = obstacle_radii
        self.safe_distance      = safe_distance

        self.N = N
        self.dt = dt
        self.Ts = Ts

        # RaceCar  Model
        self.racecar = RacecarModel(x0, L=2.5, dt=self.dt)

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

        # Initialize obstacle avoidance constraints
        num_obstacles = len(self.obstacle_positions)
        mpc.constraints.lh = np.zeros((num_obstacles,))
        finite_upper_bound = 100
        mpc.constraints.uh = np.full((num_obstacles,), finite_upper_bound)

        # Define nonlinear constraint expressions
        mpc.model.con_h_expr = ca.vertcat(*[
            (model.x[0] - obstacle_pos[0])**2 + (model.x[1] - obstacle_pos[1])**2
            for obstacle_pos in self.obstacle_positions
        ])

        # Set obstacle avoidance constraints
        for i in range(num_obstacles):
            obstacle_radius = self.obstacle_radii[i]
            mpc.constraints.lh[i] = (obstacle_radius + self.safe_distance)**2
            mpc.constraints.uh[i] =  finite_upper_bound



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
            mpc_solver = AcadosOcpSolver(mpc, json_file='race_car_obstacle_mpc.json')
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
            sim_solver = AcadosSimSolver(sim, json_file='race_car_obstacle_sim.json')
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

def animate(i, ax, mpc, state_current, simX, simU, goal_trajectory, obstacle_positions, obstacle_radii, xs, us):
    # Get the reference goal from the goal trajectory
    ref_goal = goal_trajectory[i]
    
    # Set the reference goal for the MPC problem
    yref = np.concatenate([ref_goal, np.zeros((mpc.mpc.dims.nu,))])
    yref_N = ref_goal
    
    # Solve MPC problem
    simX, simU = mpc.solve_mpc(state_current, simX, simU, yref, yref_N)
    
    # Get the first control input from the optimal solution
    u = simU[0, :]
    
    # Apply the control input to the system
    x1 = mpc.update_stateRungeKutta(state_current, u)
    
    # Update state
    state_current[:] = x1
    
    # Append current state and control to history
    xs.append(state_current)
    us.append(u)
    
    # Clear the previous plot
    ax.clear()
    
    # Plot the race car
    ax.plot(state_current[0], state_current[1], 'ro', markersize=10)
    
    # Plot the goal path
    ax.plot(goal_trajectory[:, 0], goal_trajectory[:, 1], 'b--')
    
    # Plot the obstacles
    for obstacle_pos, obstacle_radius in zip(obstacle_positions, obstacle_radii):
        obstacle = Circle(obstacle_pos, obstacle_radius, color='g', alpha=0.7)
        ax.add_patch(obstacle)
    
    # Plot the race car's trajectory
    xs_array = np.array(xs)
    ax.plot(xs_array[:, 0], xs_array[:, 1], 'r-', linewidth=1.5)
    
    # Set plot limits and labels
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Race Car Obstacle Avoidance')
    
    # Return the updated objects
    return ax


if __name__ == "__main__":
    # Initialize the MPC Controller
    ## State and Control
    state_init = np.array([0.0, 0.0, 0.0, 0.0])
    control_init = np.array([0.0, 0.0])

    ## Current State and Control
    state_current = state_init
    control_current = control_init

    ## Cost Matrices
    state_cost_matrix = np.diag([3.0, 3.0, 5.0, 5.0]) 
    control_cost_matrix = np.diag([1.0, 1.0])                
    terminal_cost_matrix = np.diag([3.0, 3.0, 5.0, 5.0]) 

    ## Constraints
    state_lower_bound = np.array([-5.0, -5.0, -np.pi, -10.0])
    state_upper_bound = np.array([5.0, 5.0, np.pi, 10.0])
    control_lower_bound = np.array([-5.0, -3.0])  
    control_upper_bound = np.array([5.0, 3.0])

    # RaceCar robot
    raceCar = RacecarModel(
        initial_state=state_init
    )

    ## Prediction Horizon
    N = 100
    sampling_time = 0.01
    Ts = 1.0
    Tsim = int(N/sampling_time)

    ## Tracks history
    xs = []
    us = []
    
    # Define multiple obstacles along the y-axis
    obstacle_positions = np.array([
        [5.0, y] for y in range(2, 10) 
    ])
    obstacle_radii = np.array([0.5] * len(obstacle_positions)) 
    safe_distance = 0.2


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
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        safe_distance=safe_distance,
        N=N,
        dt=sampling_time,
        Ts=Ts,
        cost_type='NONLINEAR_LS'
    )

    # Get the optimal state and control trajectories
    simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
    simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

    # Define the goal trajectory as a set of waypoints
    goal_trajectory = np.array([
        [i, 0.0] for i in range(11) 
    ])

    # Simulation loop
    for step in range(Tsim):
        try:
            # Determine the reference point from the goal trajectory
            ref_point_index = step % len(goal_trajectory)
            state_ref = np.array([goal_trajectory[ref_point_index, 0], goal_trajectory[ref_point_index, 1], 0.0, 4.0])  # Assuming a constant velocity of 4.0
            control_ref = np.array([1.0, 0.5])   
            yref = np.concatenate([state_ref, control_ref])
            yref_N = state_ref  # Terminal state reference

            simX, simU = mpc.solve_mpc(state_current, simX, simU, yref, yref_N)

            # Get the first control input from the optimal solution
            u = simU[0, :]

            print('Control:', u)
            print('State:', state_current)

            # Apply the control input to the system
            x1 = mpc.update_stateRungeKutta(state_current, u)

            xs.append(state_current)
            us.append(u)

            # Update state
            state_current = x1

        except Exception as e:
            print("Error in MPC solve step:", e)
            # Dump QP to JSON for debugging
            mpc.mpc_solver.dump_last_qp_to_json('last_qp.json')
            break