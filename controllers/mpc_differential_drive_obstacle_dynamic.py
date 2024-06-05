import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from models.differentialSim import DifferentialSimulation
from models.differentialSimV2 import DiffSimulation
from utils.plot_differential_drive import Simulation
from typing import Tuple

def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def export_casadi_model():

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
    model.p = ca.SX.sym('x', 6) 
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
            obstacle_positions: np.ndarray,
            obstacle_radii: float,
            safe_distance: float,
            N: int,
            dt: float,
            Ts: float,
            cost_type: str = 'LINEAR_LS'
              
        ) -> None:
        """
        Initialize the MPC controller with the given parameters.
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

        # Differential Drive Model
        # self.differential_drive = DifferentialDrive(
        #     x0_initial=x0
        # )

        # Export the Casadi Model
        self.model = export_casadi_model()


        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        self.safe_distance = safe_distance


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
        ny_p = model.p.size()[0]

        # set dimensions
        mpc.dims.N = self.N
        mpc.dims.nx = nx
        mpc.dims.nu = nu
        mpc.dims.np = ny_p

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
            Vu[nx : nx + nu, 0:nu] = np.eye(nu)
            mpc.cost.Vu = Vu
            mpc.cost.yref = np.zeros((ny, ))
            mpc.cost.yref_e = np.zeros((ny_e, ))

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

        mpc.constraints.x0  = x0

        mpc.constraints.lbx = self.state_constraints['lbx']
        mpc.constraints.ubx = self.state_constraints['ubx']
        mpc.constraints.idxbx = np.arange(nx)

        mpc.constraints.lbx_e = self.state_constraints['lbx']
        mpc.constraints.ubx_e = self.state_constraints['ubx']
        mpc.constraints.idxbx_e = np.arange(nx)

        mpc.constraints.lbu = self.control_constraints['lbu']
        mpc.constraints.ubu = self.control_constraints['ubu']
        mpc.constraints.idxbu = np.arange(nu)

        # Define obstacle avoidance constraints
        num_obstacles = len(self.obstacle_positions)
        mpc.constraints.lh = np.zeros((num_obstacles,))
        mpc.constraints.uh = np.full((num_obstacles,), 1e8)

        obstacle_pos_sym = ca.SX.sym('obstacle_pos', 2, num_obstacles)

        mpc.model.p = ca.vertcat(mpc.model.p, ca.vec(obstacle_pos_sym))
        mpc.model.con_h_expr = ca.vertcat(*[
            (model.x[0] - obstacle_pos_sym[0, i])**2 + (model.x[1] - obstacle_pos_sym[1, i])**2 - (self.obstacle_radii[i] + self.safe_distance)**2
            for i in range(num_obstacles)
        ])
        mpc.parameter_values = np.zeros(mpc.model.p.size()[0])

        for i in range(num_obstacles):
            mpc.constraints.lh[i] = (self.obstacle_radii[i] + self.safe_distance)**2


        # Set solver options
        mpc.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'ERK'
        mpc.solver_options.nlp_solver_type = 'SQP_RTI'
        mpc.solver_options.sim_method_num_stages = 4
        mpc.solver_options.sim_method_num_steps = 3
        mpc.solver_options.nlp_solver_max_iter = 100
        mpc.solver_options.qp_solver_cond_N = self.N

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
        param_values = np.concatenate((np.zeros(6), obstacle_positions.flatten()))

        for i in range(self.N+1):
            self.mpc_solver.set(i, 'p', param_values)

        # Warm up the solver
        for i in range(self.N):
            self.mpc_solver.set(i, "x", simX[i, :])
            self.mpc_solver.set(i, "u", simU[i, :])

        self.mpc_solver.set(self.N, "x", simX[self.N, :])

        # Solve the MPC problem
        status = self.mpc_solver.solve()

        if status != 0:
            raise Exception(f'Acados returned status {status}')

        for i in range(self.mpc.dims.N):
            self.mpc_solver.set(i, "yref", yref)
            simX[i, :] = self.mpc_solver.get(i, "x")
            simU[i, :] = self.mpc_solver.get(i, "u")

        self.mpc_solver.set(self.mpc.dims.N, "yref", yref_N)

        simX[self.mpc.dims.N, :] = self.mpc_solver.get(self.mpc.dims.N, "x")
        return simX, simU
    

    def runge_kutta(self, f, x, u, dt):

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        
    def differential_drive(self, x0, u):

        dx = u[0] * np.cos(x0[2])
        dy = u[0] * np.sin(x0[2])
        dyaw = u[1]

        return np.array([dx, dy, dyaw])
    
    def update_stateRungeKutta(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Use the Runge-Kutta method to update the state based on the current state and control inputs.
        """

        x1 = self.runge_kutta(self.differential_drive, x0, u, self.dt)
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
    state_init = np.array([0.0, 0.0, 0.0])
    control_init = np.array([0.0, 0.0])

    ## Current State and Control
    state_current = state_init
    control_current = control_init

    ## Cost Matrices
    state_cost_matrix = np.diag([75.0, 75.0, 150.0])
    control_cost_matrix = np.diag([1, 0.1])
    terminal_cost_matrix = np.diag([75.0, 75.0, 150.0])

    ## Constraints
    state_lower_bound = np.array([-10.0, -10.0, -3.14])
    state_upper_bound = np.array([10.0, 10.0, 3.14])
    control_lower_bound = np.array([-30.0, -31.4])
    control_upper_bound = np.array([30.0, 31.4])

    # Define multiple obstacles
    obstacle_positions = np.array([
        [2.0, 1.0],  # Obstacle 1 position (x, y)
        [3.0, 3.0],  # Obstacle 2 position (x, y)
        [2.0, 6.0]   # Obstacle 3 position (x, y)
    ])

    # Define obstacle velocities
    obstacle_velocities = 10.0*np.array([
        [0.3, .6],   # Velocity of obstacle 1 (vx, vy)
        [0.0, -0.6],  # Velocity of obstacle 2 (vx, vy)
        [0.5, 0.1]    # Velocity of obstacle 3 (vx, vy)
    ])

    obstacle_radii = np.array([0.5, 0.2, 0.4])  # Radii of the obstacles
    safe_distance = 0.7

    # Simulation
    simulation = DifferentialSimulation()

    ## Prediction Horizon
    N =10
    sampling_time = 0.01
    Ts = 0.5
    Tsim = int(N/sampling_time)

    ## Tracks history
    xs = [state_init.copy()]
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
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        safe_distance=safe_distance,
        N=N,
        dt=sampling_time,
        Ts=Ts,
        cost_type='LINEAR_LS'
    )

    # Set up the ffmpeg writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


    # Get the optimal state and control trajectories
    simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
    simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

    fig, ax = plt.subplots(figsize=(8, 8))

    differential_robot = DiffSimulation()

    # Simulation loop
    def animate(step):
        global state_current, simX, simU, xs, us, obstacle_positions
        try:
            # Determine the reference point from the goal trajectory
            state_ref = np.array([6.0, 6.0, 0.0])
            control_ref = np.array([30.0, 1.57])
            yref = np.concatenate([state_ref, control_ref])
            yref_N = state_ref  # Terminal state reference

            

            simX, simU = mpc.solve_mpc(state_current, simX, simU, yref, yref_N, obstacle_positions)

            # Get the first control input from the optimal solution
            u = simU[0, :]

            print('Control:', u)
            print('State:', state_current)

            # Apply the control input to the system
            x1 = mpc.update_stateRungeKutta(state_current, u)

            # Update obstacle positions based on their velocities
            obstacle_positions[:] += obstacle_velocities * sampling_time

            xs.append(state_current)
            us.append(u)

            # Update state
            state_current = x1

            # Clear the previous plot
            ax.clear()

            # Plot the race car
            plot_arrow(state_current[0], state_current[1], state_current[2], length=0.5, width=0.2, fc="r", ec="k")
            differential_robot.generate_each_wheel_and_draw(state_current[0], state_current[1], state_current[2])
            

            # Plot the goal point
            ax.plot(state_ref[0], state_ref[1], 'bo', markersize=10)

            # Plot the obstacles
            for obstacle_pos, obstacle_radius in zip(obstacle_positions, obstacle_radii):
                obstacle = plt.Circle(obstacle_pos, obstacle_radius, color='b', alpha=0.7)
                ax.add_patch(obstacle)

            # Plot the race car's trajectory
            xs_array = np.array(xs)
            ax.plot(xs_array[:, 0], xs_array[:, 1], 'r-', linewidth=1.5)

            # Plot the predicted trajectory using dot stars
            ax.plot(simX[:, 0], simX[:, 1], 'g*', markersize=8)

            # Set plot limits and labels
            ax.set_xlim(-2, 10)
            ax.set_ylim(-2, 10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Race Car Obstacle Avoidance')

            # Return the updated objects
            return ax

        except Exception as e:
            print("Error in MPC solve step:", e)
            return None

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=Tsim, interval=100, blit=False)

    # Save the animation as an MP4 file
    ani.save('differential_drive_obstacle_dynamics.mp4', writer='ffmpeg', fps=15)

    # Display the plot
    plt.show()