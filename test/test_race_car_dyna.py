import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from typing import Tuple

def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


class MPCController:
    def __init__(
            self,
            x0: np.ndarray,
            u0: np.ndarray,
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
            slack_weight: float,
            cost_type: str = 'LINEAR_LS'
    ) -> None:
        self.cost_type = cost_type
        self.state_cost_matrix = state_cost_matrix
        self.control_cost_matrix = control_cost_matrix
        self.terminal_cost_matrix = terminal_cost_matrix
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
        self.obstacle_positions = obstacle_positions
        self.obstacle_radii = obstacle_radii
        self.safe_distance = safe_distance
        self.slack_weight = slack_weight

        # Export the Casadi Model
        self.model = self.export_casadi_model()

        # Create the ACADOS solver
        self.mpc_solver, self.mpc = self.create_mpc_solver(x0)

    def export_casadi_model(self):
        # Dynamic model setup 
        m = 1500  # Mass of the vehicle (kg)
        Iz = 3000  # Moment of inertia about z-axis (kg·m²)
        lf = 1.0  # Distance from center of gravity to front axle (m)
        lr = 1.5  # Distance from center of gravity to rear axle (m)
        Cf = 19000  # Cornering stiffness of front tires (N/rad)
        Cr = 33000  # Cornering stiffness of rear tires (N/rad)

        # States
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        yaw = ca.MX.sym('yaw')
        v = ca.MX.sym('v')
        states = ca.vertcat(x, y, yaw, v)

        # Controls
        a = ca.MX.sym('a')  # Acceleration
        delta = ca.MX.sym('delta')  # Steering angle
        controls = ca.vertcat(a, delta)

        # Parameters for obstacles
        p = ca.MX.sym('p', 9)  # 6 for obstacle positions, 3 for obstacle radii

        # Dynamics
        beta = ca.arctan((lr * ca.tan(delta)) / (lf + lr))
        dx = v * ca.cos(yaw + beta)
        dy = v * ca.sin(yaw + beta)
        dyaw = (v / lr) * ca.sin(beta)
        dv = a

        rhs = ca.vertcat(dx, dy, dyaw, dv)

        # Define xdot
        xdot = ca.MX.sym('xdot', 4, 1)

        # Define implicit dynamics
        f_impl = xdot - rhs

        # Define obstacle avoidance constraints
        constraints = []
        for i in range(3):  # Assuming 3 obstacles
            obstacle_distance = (x - p[2*i])**2 + (y - p[2*i+1])**2
            obstacle_radius = p[6+i]  # Radius for this obstacle
            constraints.append(obstacle_distance - (obstacle_radius + self.safe_distance)**2)

        # Create an Acados model
        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = rhs
        model.x = states
        model.xdot = xdot
        model.u = controls
        model.p = p
        model.con_h_expr = ca.vertcat(*constraints)
        model.name = 'race_car_dynamics'

        return model
    
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
        mpc.dims.ny = ny
        mpc.dims.ny_e = ny_e

        # Set cost
        if self.cost_type == 'LINEAR_LS':
            mpc.cost.cost_type = 'LINEAR_LS'
            mpc.cost.cost_type_e = 'LINEAR_LS'
            mpc.cost.W = block_diag(self.state_cost_matrix, self.control_cost_matrix)
            mpc.cost.W_e = self.terminal_cost_matrix
            Vx = np.zeros((ny, nx))
            Vx[:nx, :nx] = np.eye(nx)
            mpc.cost.Vx = Vx
            mpc.cost.Vx_e = np.eye(nx)
            Vu = np.zeros((ny, nu))
            Vu[nx:nx + nu, 0:nu] = np.eye(nu)
            mpc.cost.Vu = Vu
            mpc.cost.yref = np.zeros((ny,))
            mpc.cost.yref_e = np.zeros((ny_e,))
        elif self.cost_type == 'NONLINEAR_LS':
            mpc.cost.cost_type = 'NONLINEAR_LS'
            mpc.cost.cost_type_e = 'NONLINEAR_LS'
            mpc.model.cost_y_expr = ca.vertcat(model.x, model.u)
            mpc.model.cost_y_expr_e = model.x
            mpc.cost.yref = np.zeros((ny,))
            mpc.cost.yref_e = np.zeros((ny_e,))
            mpc.cost.W = block_diag(self.state_cost_matrix, self.control_cost_matrix)
            mpc.cost.W_e = self.terminal_cost_matrix

        # Set constraints
        mpc.constraints.x0 = x0
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
        mpc.constraints.uh = np.full((num_obstacles,), 1e9)

        # Set up parameters
        mpc.parameter_values = np.zeros(model.p.size()[0])

        # Set solver options
        mpc.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'IRK'
        mpc.solver_options.nlp_solver_type = 'SQP_RTI'
        mpc.solver_options.sim_method_num_stages = 4
        mpc.solver_options.sim_method_num_steps = 3
        mpc.solver_options.nlp_solver_max_iter = 400
        mpc.solver_options.qp_solver_cond_N = self.N

        # Set prediction horizons
        mpc.solver_options.tf = self.Ts

        print("Creating MPC solver...")
        mpc_solver = AcadosOcpSolver(mpc, json_file='mpc_race_car_dynamics.json')
        print("MPC solver created successfully.")
        return mpc_solver, mpc

    def solve_mpc(self, x0: np.ndarray,
                simX: np.ndarray,
                simU: np.ndarray,
                yref: np.ndarray,
                yref_N: np.ndarray,
                obstacle_positions: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray]:
        nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]

        if x0.shape[0] != nx:
            raise ValueError(f"Initial state x0 must have dimension {nx}, but got {x0.shape[0]}.")

        self.mpc_solver.set(0, "lbx", x0)
        self.mpc_solver.set(0, "ubx", x0)

        # Prepare parameters: obstacle positions and radii
        param_values = np.concatenate([obstacle_positions.flatten(), self.obstacle_radii])

        for i in range(self.N+1):
            self.mpc_solver.set(i, 'p', param_values)

        # Set reference for all time steps
        for i in range(self.N):
            self.mpc_solver.set(i, "yref", yref)
        
        # Set terminal reference
        self.mpc_solver.set(self.N, "yref", yref_N)

        # Initialize state and control trajectories
        for i in range(self.N):
            self.mpc_solver.set(i, "x", simX[i, :])
            self.mpc_solver.set(i, "u", simU[i, :])
        self.mpc_solver.set(self.N, "x", simX[self.N, :])

        # Solve the optimization problem
        status = self.mpc_solver.solve()

        if status != 0:
            print(f'Warning: Acados returned status {status}')

        # Retrieve the optimized state and control trajectories
        for i in range(self.N):
            simX[i, :] = self.mpc_solver.get(i, "x")
            simU[i, :] = self.mpc_solver.get(i, "u")
        simX[self.N, :] = self.mpc_solver.get(self.N, "x")

        return simX, simU

    def update_state(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Update the state based on the current state and control inputs using RK4 integration.
        """
        k1 = self.dynamics(x0, u)
        k2 = self.dynamics(x0 + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(x0 + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(x0 + self.dt * k3, u)
        return x0 + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def dynamics(self, x, u):
        """
        Race car dynamics.
        """
        m = 1500  # Mass of the vehicle (kg)
        Iz = 3000  # Moment of inertia about z-axis (kg·m²)
        lf = 1.0  # Distance from center of gravity to front axle (m)
        lr = 1.5  # Distance from center of gravity to rear axle (m)
        Cf = 19000  # Cornering stiffness of front tires (N/rad)
        Cr = 33000  # Cornering stiffness of rear tires (N/rad)
        
        x_pos, y_pos, yaw, v = x
        a, delta = u
        
        beta = np.arctan((lr * np.tan(delta)) / (lf + lr))
        dx = v * np.cos(yaw + beta)
        dy = v * np.sin(yaw + beta)
        dyaw = (v / lr) * np.sin(beta)
        dv = a
        
        return np.array([dx, dy, dyaw, dv])

def update_obstacle_positions(step, initial_positions):
    """
    Update obstacle positions based on the current step.
    You can define any motion pattern here.
    """
    # Example: Circular motion for the first obstacle, linear motion for the second, static for the third
    positions = initial_positions.copy()
    
    # Circular motion for the first obstacle
    radius = 0.5
    angular_speed = 0.08
    positions[0, 0] = 2.0 + radius * np.cos(angular_speed * step)
    positions[0, 1] = 1.0 + radius * np.sin(angular_speed * step)
    
    # Linear motion for the second obstacle
    positions[1, 0] = 4.0 + 0.05 * step  
    positions[1, 1] = 2.5 + 0.03 * step
    # Third obstacle remains static
    
    return positions


if __name__ == "__main__":
    # Initialize the MPC Controller     
    ## State and Control
    state_init = np.array([0.0, 0.0, 0.0, 0.0]) 
    control_init = np.array([0.0, 0.0])  

    ## Current State and Control
    state_current = state_init.copy()
    control_current = control_init.copy()

    ## Cost Matrices
    state_cost_matrix = 10*np.diag([5, 5, 9, 10])
    control_cost_matrix = np.diag([0.1, 0.1])  
    terminal_cost_matrix = 2 * state_cost_matrix

    ## Constraints
    large_bound = 1e6  
    state_lower_bound = np.array([-large_bound, -large_bound, -np.pi, -15.0])
    state_upper_bound = np.array([large_bound, large_bound, np.pi, 15.0])
    control_lower_bound = np.array([-5.0, -np.pi])
    control_upper_bound = np.array([5.0, np.pi])

    # Define multiple obstacles
    initial_obstacle_positions = np.array([
        [2.0, 1.0],
        [4.0, 2.5], 
        [2.0, 3.0]   
    ])
    obstacle_radii = np.array([0.5, 0.3, 0.4])
    safe_distance = 0.5

    ## Prediction Horizon
    N = 10
    dt = 0.05
    Ts = N * dt  # Prediction horizon time

    # Simulation time
    sim_time = 15  # seconds
    num_sim_steps = 100

    ## Tracks history
    xs = [state_init.copy()]
    us = []

    # MPC controller
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
        obstacle_positions=initial_obstacle_positions,
        obstacle_radii=obstacle_radii,
        safe_distance=safe_distance,
        N=N,
        dt=dt,
        Ts=Ts,
        slack_weight=1000.0,
        cost_type='NONLINEAR_LS'
    )

    # Get the optimal state and control trajectories
    simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
    simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Target state
    target_state = np.array([4.0, 1.0, 0.0, 0.0])

    # Input control history
    accel = [0.0]
    steer = [0.0]

    # Simulation loop
    def animate(step):
        global state_current, simX, simU, xs, us

        try:
            # Update obstacle positions
            obstacle_positions = update_obstacle_positions(step, initial_obstacle_positions)
            # Determine the reference trajectory
            state_ref = target_state
            control_ref = np.array([0.0, 0.0]) 
            yref = np.concatenate([state_ref, control_ref])
            yref_N = state_ref  # Terminal state reference

            simX, simU = mpc.solve_mpc(state_current, simX, simU, yref, yref_N, obstacle_positions)

            # Get the first control input from the optimal solution
            u = simU[0, :]

            print(f'Step: {step}, Control: {u}, State: {state_current}')

            # Apply the control input to the system
            state_current = mpc.update_state(state_current, u)

            xs.append(state_current)
            us.append(u)

            accel.append(u[0])
            steer.append(u[1])

            # Clear the previous plot
            ax.clear()

            # Plot the target point
            ax.plot(target_state[0], target_state[1], 'g*', markersize=15, label='Target')

            # Plot the race car
            plot_arrow(state_current[0], state_current[1], state_current[2], length=0.5, width=0.2, fc="r", ec="k")

            # Plot obstacles
            for obstacle_pos, obstacle_radius in zip(obstacle_positions, obstacle_radii):
                obstacle = plt.Circle(obstacle_pos, obstacle_radius, color='gray', alpha=0.7)
                ax.add_patch(obstacle)

            # Plot the race car's trajectory
            xs_array = np.array(xs)
            ax.plot(xs_array[:, 0], xs_array[:, 1], 'b-', linewidth=1.5, label='Race Car trajectory')

            # Plot the predicted trajectory
            ax.plot(simX[:, 0], simX[:, 1], 'r--', linewidth=1.5, label='Predicted trajectory')

            # Set plot limits and labels
            ax.set_xlim(-1, 6)
            ax.set_ylim(-1, 6)
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.set_title(f'Race Car - Step {step}')
            ax.legend()

            # Return the updated objects
            return ax

        except Exception as e:
            print(f"Error in MPC solve step: {e}")
            return None

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=num_sim_steps, interval=100, blit=False, repeat=False)

    # Save the animation as an MP4 file
    ani.save('race_car_dynamics.mp4', writer='ffmpeg', fps=10)

    control_inputs = np.array(us)

    # Save control inputs plot to a file instead of displaying it
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(accel, label='Acceleration')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Acceleration (m/s^2)')
    ax1.set_title('Acceleration Control Input')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(steer, label='Steering angle')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Steering angle (rad)')
    ax2.set_title('Steering Angle Control Input')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('control_inputs_racecar.png')
    plt.close(fig)  # Close the figure to free up memory

    # Save states plot to a file instead of displaying it
    state_array = np.array(xs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(state_array[:, 0], label='X position')
    ax1.plot(state_array[:, 1], label='Y position')
    ax1.plot(state_array[:, 2], label='Yaw (orientation)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Position (m) / Orientation (rad)')
    ax1.set_title('Race Car Position and Orientation')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(state_array[:, 3], label='Velocity')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Race Car Velocity')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('race_car_states.png')
    plt.close(fig)  # Close the figure to free up memory

print("Simulation complete. Results saved to video and image files.")